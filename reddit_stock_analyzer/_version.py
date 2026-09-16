"""The package version, in one place.

Deliberately a module of its own containing nothing but a literal. Both
`pyproject.toml` (via ``[tool.setuptools.dynamic]``) and the package read from
here, so a release is one edit rather than two values that can drift apart.

Kept out of ``__init__.py`` for two reasons: importing the package pulls in
``httpx``, which a build backend should not need, and ``config`` would
otherwise import the package it is part of. A literal in a leaf module is read
statically by setuptools — no import, no dependencies, no cycle.
"""

__version__ = "0.1.0"
