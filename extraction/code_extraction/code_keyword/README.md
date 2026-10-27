# code_keyword — check/switch keyword set for the `code` domain

`keywords.json` is the keyword set `extraction/src/hidden_analysis.py --keywords code` uses to
tag each `\n\n`-delimited reasoning step inside `<think>` as **check** (reflection),
**switch** (transition) or **other** (execution). It is loaded at import time by
`extraction/src/keyword_sets.py`, which scans `extraction/*_extraction/*_keyword/keywords.json`;
the folder name before `_keyword` is the `--keywords` name.

Fields: `check_words` / `switch_words` are case-insensitive substring matches; `check_prefix` /
`switch_prefix` are case-insensitive step-prefix matches. Check is tested before switch.

Notes: Code-adapted lists (v_code): all "contains" matching (wait/alternatively promoted from prefix to contains), plus code-specific cues.

Changing this file changes which boundary activations enter the contrast vector, so bump the
`--save_prefix` (or regenerate the vector) rather than silently rebuilding over an old one.
