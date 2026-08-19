#!/usr/bin/env python3
"""Static scan for names a module reads but never binds, and for keyword-argument mismatches.

WHY THIS EXISTS
    The 2026-08-19 refactor landed eight production bugs, and every one of them was invisible to
    `import`:

      * `agent/nodes/curate.py` called `hashlib.sha256` after the `import hashlib` was removed with
        an unrelated block — so the EVAL FIREWALL raised on the first row it blocked, meaning the
        safety mechanism killed the run at the exact moment it caught a leak;
      * `data/loaders/web_acquire.py` read `_spec` and `_closed_label_space` after their assignment
        moved into a different function;
      * `agent/checkpoint.py` imported a constant that had been deleted, inside a function called at
        module scope by the runner — every run died before the graph was built;
      * two encoders read `EvalSet` fields that no longer existed, so no checkpoint could be written;
      * two call sites passed keyword arguments their callee had dropped.

    A smoke test that imports every module catches none of these: a lazy import inside a function, an
    undefined name on a branch, and a keyword mismatch are all runtime failures. They are, however,
    trivially visible to the AST — which is what this does.

WHAT IT CHECKS
    1. Every `Name` load in a module resolves to something bound in an enclosing scope, a builtin, or
       an import.
    2. Every call to a function defined in the SAME file passes only keyword arguments that function
       accepts.

    Deliberately conservative: it reports only names it is confident about, because a scan that cries
    wolf gets switched off. Run it before committing a refactor that moves code between functions.

USAGE
    python scripts/check_unresolved_names.py [path ...]      # defaults to the production packages
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

DEFAULT_ROOTS = ("agent", "config", "data", "eval", "tasks", "training")
_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__"}


class _Scope:
    def __init__(self, parent: "_Scope | None" = None, *, transparent: bool = False):
        self.parent = parent
        self.names: set[str] = set()
        # A comprehension/lambda scope can see the enclosing function's locals; a nested `def`
        # can too. Class bodies cannot be seen from methods, which is why they are not transparent.
        self.transparent = transparent

    def bind(self, name: str) -> None:
        self.names.add(name)

    def resolves(self, name: str) -> bool:
        if name in self.names or name in _BUILTINS:
            return True
        return self.parent.resolves(name) if self.parent else False


def _bind_target(node: ast.AST, scope: _Scope) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            scope.bind(child.id)


class _Checker(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.problems: list[str] = []
        self.module = _Scope()
        self.scope = self.module
        self.functions: dict[str, ast.arguments] = {}

    # -- scope helpers ----------------------------------------------------------
    def _in_scope(self, scope: _Scope, body) -> None:
        previous, self.scope = self.scope, scope
        for node in body:
            self.visit(node)
        self.scope = previous

    def _hoist(self, body) -> None:
        """Bind every name a block assigns before visiting it, so forward references inside
        functions (the normal case for mutually recursive helpers) are not reported."""
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.scope.bind(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    self.scope.bind((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    _bind_target(target, self.scope)
            elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    self._hoist(getattr(node, attr, []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    self._hoist(handler.body)

    # -- visitors ---------------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        self._hoist(node.body)
        for child in node.body:
            self.visit(child)

    def visit_FunctionDef(self, node) -> None:
        self.scope.bind(node.name)
        self.functions.setdefault(node.name, node.args)
        for decorator in node.decorator_list:
            self.visit(decorator)
        inner = _Scope(self.scope, transparent=True)
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            inner.bind(arg.arg)
        if args.vararg:
            inner.bind(args.vararg.arg)
        if args.kwarg:
            inner.bind(args.kwarg.arg)
        for default in [*args.defaults, *(d for d in args.kw_defaults if d)]:
            self.visit(default)
        previous, self.scope = self.scope, inner
        self._hoist(node.body)
        for child in node.body:
            self.visit(child)
        self.scope = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        inner = _Scope(self.scope, transparent=True)
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            inner.bind(arg.arg)
        self._in_scope(inner, [ast.Expr(value=node.body)])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.bind(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        inner = _Scope(self.scope)
        self._hoist(node.body)
        self._in_scope(inner, node.body)

    def _comprehension(self, node) -> None:
        inner = _Scope(self.scope, transparent=True)
        for generator in node.generators:
            _bind_target(generator.target, inner)
        previous, self.scope = self.scope, inner
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for attr in ("elt", "key", "value"):
            child = getattr(node, attr, None)
            if isinstance(child, ast.AST):
                self.visit(child)
        self.scope = previous

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comprehension

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.scope.bind(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self.scope.bind(name)

    visit_Nonlocal = visit_Global

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.scope.bind(node.id)
            return
        if not self.scope.resolves(node.id):
            self.problems.append(
                f"{self.path}:{node.lineno}: name {node.id!r} is read but never bound"
            )

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return
        args = self.functions.get(node.func.id)
        if args is None or args.kwarg is not None:
            return
        accepted = {
            arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        }
        for keyword in node.keywords:
            if keyword.arg and keyword.arg not in accepted:
                self.problems.append(
                    f"{self.path}:{node.lineno}: {node.func.id}() got keyword "
                    f"{keyword.arg!r}, which it does not accept"
                )


def check(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        return [f"{path}: could not parse ({error})"]
    checker = _Checker(path)
    checker.visit(tree)
    return checker.problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path(r) for r in DEFAULT_ROOTS]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    problems: list[str] = []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        problems.extend(check(path))
    for problem in problems:
        print(problem)
    print(f"\nscanned {len(files)} file(s); {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
