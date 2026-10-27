"""Load the per-domain check/switch keyword sets used by hidden_analysis.py.

Each domain keeps its set at extraction/<domain>_extraction/<domain>_keyword/keywords.json
({"check_words": [...], "check_prefix": [...], "switch_words": [...], "switch_prefix": [...]}).
The folder name before ``_keyword`` is the name accepted by ``--keywords``.
"""
import glob
import json
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_EXTRACTION_DIR = os.path.dirname(_SRC_DIR)
_REQUIRED = ("check_words", "check_prefix", "switch_words", "switch_prefix")


def load_keyword_sets(extraction_dir=_EXTRACTION_DIR):
    sets = {}
    for path in sorted(glob.glob(os.path.join(extraction_dir, "*_extraction", "*_keyword", "keywords.json"))):
        folder = os.path.basename(os.path.dirname(path))
        name = folder[: -len("_keyword")]
        with open(path) as fin:
            kw = json.load(fin)
        missing = [k for k in _REQUIRED if k not in kw]
        if missing:
            raise ValueError(f"{path} is missing keys {missing}")
        sets[name] = {k: list(kw[k]) for k in _REQUIRED}
    if not sets:
        raise RuntimeError(f"no keyword sets found under {extraction_dir}/*_extraction/*_keyword/keywords.json")
    return sets


KEYWORD_SETS = load_keyword_sets()
