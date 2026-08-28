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

    The CROSS-MODULE checks were added on 2026-08-19 after verification run 38656655 died 27 minutes
    in — past the loader, the task brief and the teacher-fitness gate, i.e. at the most expensive
    possible moment — on two faults this scan could see and was not looking for:

      * four call sites passed `task=` / `task_type=` to `eval.harness.run_eval`, whose signature had
        dropped it (the task now comes from the EvalSet). All four are on GPU-only paths, which is why
        1,377 CPU tests were green;
      * `tests/pipeline/run.py` did `from eval.harness import TASK_METRIC_NAMES`, a constant deleted
        with the task channels — and it sat 26 lines ABOVE the final report, so the crash also took
        the whole report with it.

WHAT IT CHECKS
    1. Every `Name` load in a module resolves to something bound in an enclosing scope, a builtin, or
       an import.
    2. Every call to a function defined in the SAME file passes only keyword arguments that function
       accepts.
    3. Every `from <project module> import <name>` names something that module actually defines.
    4. Every call to a function IMPORTED from another project module passes only keyword arguments
       that function accepts.

    Deliberately conservative: it reports only names it is confident about, because a scan that cries
    wolf gets switched off. Checks 3 and 4 skip anything whose definition it cannot see plainly — a
    re-export, a conditional definition, a decorated function, a `**kwargs` signature — so a clean run
    is not proof of correctness, only the absence of the mistakes it can prove.

USAGE
    python scripts/check_unresolved_names.py [path ...]      # defaults to the production packages
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

# `tests/pipeline/run.py` and `hardware_eval` are in here because both carried one of the faults that
# killed run 38656655, and neither is imported by anything the earlier roots covered.
DEFAULT_ROOTS = (
    "agent", "config", "data", "eval", "tasks", "training", "hardware_eval",
    "tests/pipeline/run.py",
)
# Package roots a `from X import Y` is checked against. An import from anything else (stdlib, a
# third-party package) is left alone: this scan knows about THIS project's files, nothing more.
PROJECT_PACKAGES = ("agent", "config", "data", "eval", "tasks", "training", "hardware_eval")
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
    def __init__(self, path: Path, *, index=None):
        self.path = path
        self.problems: list[str] = []
        self.index = index
        # local name -> (module, original name), for every `from <module> import <name>` this file
        # performs at ANY depth. Lazy imports inside functions are the norm in this codebase (they
        # keep torch out of CPU-only paths), so a check that only looked at module scope would miss
        # most of them — including three of the four that killed run 38656655.
        self.imported_from: dict[str, tuple[str, str]] = {}
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

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.generic_visit(node)
        # Relative imports are resolved against the file's own package; `level` chases the dots.
        module = node.module or ""
        if node.level:
            parts = self.path.resolve().parent.parts
            try:
                anchor = parts[len(parts) - node.level:] if node.level > 1 else ()
            except Exception:  # noqa: BLE001
                return
            base = ".".join(p for p in parts if p in PROJECT_PACKAGES or anchor)
            if not base:
                return
            module = f"{base}.{module}" if module else base
        if not module or self.index is None:
            return
        if module.split(".")[0] not in PROJECT_PACKAGES:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self.imported_from[alias.asname or alias.name] = (module, alias.name)
            if self.index.defines(module, alias.name) is False:
                self.problems.append(
                    f"{self.path}:{node.lineno}: {module!r} does not define {alias.name!r}"
                )

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
        name = node.func.id
        # Same file first: a local definition shadows an import of the same name.
        args = self.functions.get(name)
        where = "" if args is not None else None
        if args is None and self.index is not None and name in self.imported_from:
            module, original = self.imported_from[name]
            args = self.index.signature(module, original)
            where = f" (defined in {module})"
        if args is None or args.kwarg is not None:
            return
        accepted = {
            arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]
        }
        for keyword in node.keywords:
            if keyword.arg and keyword.arg not in accepted:
                self.problems.append(
                    f"{self.path}:{node.lineno}: {name}() got keyword "
                    f"{keyword.arg!r}, which it does not accept{where or ''}"
                )


class _ModuleIndex:
    """What each project module defines at top level, and the signature of each plain function.

    Built by parsing, never by importing: importing `training.lora_trainer` to ask what it exports
    would load torch, and a lint that needs a GPU is a lint nobody runs.
    """

    def __init__(self, root: Path):
        self.root = root
        self._defined: dict[str, set[str] | None] = {}
        self._signatures: dict[str, ast.arguments | None] = {}

    def _path_for(self, module: str) -> Path | None:
        if not module.split(".")[0] in PROJECT_PACKAGES:
            return None
        base = self.root.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def _load(self, module: str) -> set[str] | None:
        if module in self._defined:
            return self._defined[module]
        self._defined[module] = None
        path = self._path_for(module)
        if path is None:
            return None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            return None
        names: set[str] = set()
        star_import = False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                if isinstance(node, ast.FunctionDef) and not node.decorator_list:
                    self._signatures[f"{module}.{node.name}"] = node.args
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        names.update(e.id for e in target.elts if isinstance(e, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Import):
                names.update((a.asname or a.name.split(".")[0]) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "*" for a in node.names):
                    # A star import means this module's surface is not knowable from its own text.
                    star_import = True
                names.update((a.asname or a.name) for a in node.names if a.name != "*")
            elif isinstance(node, (ast.If, ast.Try)):
                # Conditional top-level definitions (TYPE_CHECKING blocks, optional dependencies).
                # Collected but never used for signatures, since which branch ran is not knowable.
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(sub.name)
                    elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
                    elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                        names.update((a.asname or a.name.split(".")[0]) for a in sub.names
                                     if a.name != "*")
        # A submodule is importable from its package without the package naming it.
        package_dir = self._path_for(module)
        if package_dir is not None and package_dir.name == "__init__.py":
            names.update(child.stem for child in package_dir.parent.glob("*.py"))
            names.update(child.name for child in package_dir.parent.iterdir() if child.is_dir())
        self._defined[module] = None if star_import else names
        return self._defined[module]

    def defines(self, module: str, name: str) -> bool | None:
        """True / False, or None when the module's surface is not knowable from its text."""
        names = self._load(module)
        return None if names is None else name in names

    def signature(self, module: str, name: str) -> ast.arguments | None:
        self._load(module)
        return self._signatures.get(f"{module}.{name}")


def check(path: Path, index: "_ModuleIndex | None" = None) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        return [f"{path}: could not parse ({error})"]
    checker = _Checker(path, index=index)
    checker.visit(tree)
    return checker.problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path(r) for r in DEFAULT_ROOTS]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    index = _ModuleIndex(Path(__file__).resolve().parent.parent)
    problems: list[str] = []
    for path in files:
        if "__pycache__" in path.parts:
            continue
        problems.extend(check(path, index))
    for problem in problems:
        print(problem)
    print(f"\nscanned {len(files)} file(s); {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
