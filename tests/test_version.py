"""The version lives in exactly one place.

A release used to mean editing `pyproject.toml` and `__init__.py` and hoping
they matched — the failure mode that left ticker-price-data shipping 0.1.4
while reporting 0.1.3. These pin the single-source arrangement so it cannot
quietly come apart.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tomllib

import reddit_stock_analyzer
from reddit_stock_analyzer import _version
from reddit_stock_analyzer.config import USER_AGENT

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_package_exports_the_version():
    assert reddit_stock_analyzer.__version__ == _version.__version__


def test_version_looks_like_a_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", _version.__version__)


def test_pyproject_reads_the_version_from_the_package():
    # If someone re-adds a literal `version` here, the drift is back.
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert "version" not in pyproject["project"], (
        "pyproject declares a literal version again — it should stay dynamic so "
        "reddit_stock_analyzer/_version.py is the only place to bump"
    )
    assert "version" in pyproject["project"]["dynamic"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "reddit_stock_analyzer._version.__version__"
    )


def test_version_module_holds_nothing_but_the_literal():
    # setuptools reads this statically at build time. An import here would mean
    # the build backend needs the package's dependencies just to learn a string.
    tree = ast.parse((ROOT / "reddit_stock_analyzer" / "_version.py").read_text())
    assert not [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ], "_version.py must not import anything"

    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)


def test_user_agent_carries_the_current_version():
    # Reddit asks clients to identify themselves; a stale number here is the
    # third place the version used to be written by hand.
    assert f"reddit-stock-analyzer/{_version.__version__}" in USER_AGENT


def test_user_agent_is_overridable(monkeypatch):
    monkeypatch.setenv("REDDIT_USER_AGENT_OVERRIDE", "custom/1.0")
    import importlib

    from reddit_stock_analyzer import config

    importlib.reload(config)
    try:
        assert config.USER_AGENT == "custom/1.0"
    finally:
        monkeypatch.delenv("REDDIT_USER_AGENT_OVERRIDE")
        importlib.reload(config)
