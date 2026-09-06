"""Adds the (vendored) michelleblom/AZUL framework to sys.path.

Import this from any script that needs `model`, `utils`, etc.
Looks for ./framework first (vendored copy for deploy), then ../framework
(sibling layout used during local development before vendoring).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    os.path.join(_HERE, "framework"),
    os.path.normpath(os.path.join(_HERE, "..", "framework")),
):
    if os.path.isdir(candidate):
        if candidate in sys.path:
            sys.path.remove(candidate)
        sys.path.insert(0, candidate)
        break
