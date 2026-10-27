# math_keyword — check/switch keyword set for the `math` domain

`keywords.json` is the keyword set `extraction/src/hidden_analysis.py --keywords math` uses to
tag each `\n\n`-delimited reasoning step inside `<think>` as **check** (reflection),
**switch** (transition) or **other** (execution). It is loaded at import time by
`extraction/src/keyword_sets.py`, which scans `extraction/*_extraction/*_keyword/keywords.json`;
the folder name before `_keyword` is the `--keywords` name.

Fields: `check_words` / `switch_words` are case-insensitive substring matches; `check_prefix` /
`switch_prefix` are case-insensitive step-prefix matches. Check is tested before switch.

Notes: Upstream SEAL's lists (v_math), except the dead "think differenly" typo is fixed to "think differently".

Changing this file changes which boundary activations enter the contrast vector, so bump the
`--save_prefix` (or regenerate the vector) rather than silently rebuilding over an old one.
