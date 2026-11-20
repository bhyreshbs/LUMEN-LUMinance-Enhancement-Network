"""Shared pytest configuration.

The model checkpoints are deliberately not in the repository (licence uncertain, see weights/README.md), so a clean
clone - and CI - has none. A test that fails ONLY because a checkpoint file is absent is reported as a skip with that
reason instead of a failure. Every other failure stays a failure, including a checkpoint that exists but is corrupt
or fails its SHA-256 check.
"""
from __future__ import annotations

import pytest

from lumen.deep.infer import CheckpointError

_MISSING = " checkpoint found (looked for:"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None:
        exc = call.excinfo.value
        if isinstance(exc, CheckpointError) and _MISSING in str(exc):
            rep.outcome = "skipped"
            rep.longrepr = (str(item.fspath), item.location[1] or 0, f"Skipped: needs a model checkpoint that is not present ({str(exc).split(' (looked')[0]})")
