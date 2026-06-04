"""Root conftest.py — ensures src/ is on sys.path for all tests.

With hatchling's import-hook editable install, ``commerce_ml`` may be
registered in ``sys.modules`` before any test runs, but with an ``__path__``
that doesn't expose the raw ``src/`` file tree.  This means subpackages such
as ``commerce_ml.data`` and ``commerce_ml.features`` raise
``ModuleNotFoundError`` even though the files exist.

By inserting ``src/`` at the front of ``sys.path`` here — and evicting any
stale ``commerce_ml`` cache entries — we guarantee that every test file
imports the package directly from the filesystem, regardless of how the
editable install is configured.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).parent / "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Drop any commerce_ml entries that the editable install hook may have
# pre-loaded with an incorrect __path__ so they get re-imported from src/.
_stale = [k for k in sys.modules if k == "commerce_ml" or k.startswith("commerce_ml.")]
for _k in _stale:
    del sys.modules[_k]
