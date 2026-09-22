"""Reuse of the Semantic Fusion fixture engine.

Semantic Mapping is tested against *real* fused evidence, built by the baseline accumulation from
coherent observations and perception results, instead of hand-made records that could drift from
what Semantic Fusion actually produces. The engine lives in the fusion tests, so its directory
is put on the import path for this package only.
"""

import sys
from pathlib import Path

_FUSION_TESTS = str(Path(__file__).resolve().parents[1] / "semantic_fusion")
if _FUSION_TESTS not in sys.path:
    sys.path.insert(0, _FUSION_TESTS)
