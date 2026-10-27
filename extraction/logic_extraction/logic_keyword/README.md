# logic_keyword — check/switch keyword set for the `logic` domain

`keywords.json` is the keyword set `extraction/src/hidden_analysis.py --keywords logic` uses to
tag each `\n\n`-delimited reasoning step inside `<think>` as **check** (reflection),
**switch** (transition) or **other** (execution). It is loaded at import time by
`extraction/src/keyword_sets.py`, which scans `extraction/*_extraction/*_keyword/keywords.json`;
the folder name before `_keyword` is the `--keywords` name.

Fields: `check_words` / `switch_words` are case-insensitive substring matches; `check_prefix` /
`switch_prefix` are case-insensitive step-prefix matches. Check is tested before switch.

Notes: The set that built the shipped logiqa2_v_logic.pt. Derived from real LogiQA reasoning traces (see discover_logic_keywords.sh + logic_keyword_coverage.py in this folder), not invented up front. check_words extends math/code's checking vocabulary with phrasing this model uses for LogiQA ("not sure", "let me re-examine", "making/made a mistake"); switch reuses math's vocabulary. This is the version consistent with the committed vector.

Changing this file changes which boundary activations enter the contrast vector, so bump the
`--save_prefix` (or regenerate the vector) rather than silently rebuilding over an old one.
