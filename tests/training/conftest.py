"""
Ensure the project root's `training` package takes precedence over
`tests/training/`, which pytest can shadow due to its package insertion order.
"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remove stale cache entries that point into tests/training
for key in list(sys.modules.keys()):
    mod = sys.modules[key]
    file_attr = getattr(mod, "__file__", None) or ""
    if file_attr and os.path.join("tests", "training") in file_attr.replace("\\", "/"):
        del sys.modules[key]

# Ensure project root is first on sys.path
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
elif sys.path[0] != _project_root:
    sys.path.remove(_project_root)
    sys.path.insert(0, _project_root)
