"""Run the pre-GPU checklist on the R_iso control draws.

Every check here is CPU-only and runs in seconds, which is the point: a bad
control vector is much cheaper to catch now than after three MATH-500 arms have
been evaluated against it. The checklist is §9 of
``docs/random_vector_build_plan.html``.

What each check is actually guarding against:

``shape/dtype``
    The eval loads the tensor and adds it straight into the residual stream. A
    ``[1, 1536]`` tensor would broadcast rather than fail, silently steering
    something other than what was intended.

``norm``
    ``coef`` is fixed at −1.0, so the norm *is* the intervention strength. A raw
    ``randn(1536)`` has norm ~39.2; an unnormalized draw would be a ~0.7x
    intervention, and a unit draw a ~0.02x one. Both would read as "random
    steering does nothing" for entirely uninteresting reasons.

``protocol: norm-matched to S_combo_<size>``
    The 1.5B protocol sets the target norm with ``--like S_combo``, i.e. the
    mean of the three domain-vector norms (57.06 at 1.5B, 85.04 at 7B). A draw
    matched to anything else — e.g. the mean of only two domain vectors — is
    not the same control and is flagged here.

``protocol: screened against every packaged vector of its size``
    The build records what it screened against in ``references``. The 1.5B
    draws were screened against S_math, S_code, S_logic, S_general and S_combo;
    a draw whose meta lists fewer than the packaged vectors that exist for its
    size was built to a weaker protocol and is flagged.

``cosine vs the SEAL vectors``
    Confirms the control is not accidentally a weak copy of the thing it is
    controlling for. The reference set is size-aware: every packaged vector of
    the draw's model size that exists on disk, and a FAIL for any required one
    (S_math, S_code, S_logic, S_combo) that is missing.

``pairwise cosine between draws``
    Three draws that happened to be similar would not be three independent
    samples, and the whole argument for building three collapses.

``reproducibility``
    Rebuilds each vector from its recorded seed and compares. Bit-identical
    is required when the running torch version matches the one recorded in
    the meta (``torch_version``) — that combination means the meta really is
    a sufficient record to regenerate the exact tensor. Across different
    torch versions, CPU RNG output for the same seed is not guaranteed
    bit-identical by PyTorch, so a mismatch there is downgraded to an
    informational skip as long as the rebuild still matches direction and
    norm; this is expected drift, not a defect in the ``.pt`` file. Either
    way, the eval scripts consume the committed bytes directly via
    ``torch.load`` and never re-derive the vector from its seed, so this
    check is an audit-trail property, not a correctness requirement for
    steering.

``meta agreement``
    Catches a ``.pt`` and a ``.meta.json`` that have drifted apart — e.g. a
    vector rebuilt with different flags but the old meta left in place.

Exits nonzero if any check fails, so it can gate a run script.

Usage::

    python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py
    python extraction/control_extraction/control_extraction_1.5B/pipeline/verify_random_vectors.py \\
        --dir extraction/control_extraction/control_extraction_7B/result --prefix R_iso

The model size is inferred from ``--dir`` / ``--prefix`` (``control_extraction_7B`` or a
``_7b`` prefix means 7B, otherwise 1.5B); pass ``--model-size`` to override.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "extraction", "src"))  # build_general_vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # build_random_vector (sibling)

from build_general_vector import cosine  # noqa: E402
from build_random_vector import draw_iso, load_reference  # noqa: E402

NORM_TOL = 1e-3
COSINE_TOL = 1e-4  # ~3 orders of magnitude tighter than the 0.08 redraw threshold

PASS = "  ok  "
FAIL = " FAIL "
SKIP = " skip "

MODEL_SIZES = ("1.5B", "7B")

# The packaged vectors of each size, in the order the 1.5B builds screened them.
# ``required`` ones must exist (FAIL otherwise); S_general is optional because
# it has not been built at 7B. Paths are repo-root-relative. The same table
# drives build_random_vectors.sh — keep the two in step.
# (name, path, required)
REFERENCE_VECTORS: Dict[str, List[Tuple[str, str, bool]]] = {
    "1.5B": [
        ("S_math", "extraction/math_extraction/math_extraction_1.5B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt", True),
        ("S_code", "extraction/code_extraction/code_extraction_1.5B/result/apps_v_code.pt", True),
        ("S_logic", "extraction/logic_extraction/logic_extraction_1.5B/result/logiqa2_v_logic.pt", True),
        ("S_general", "extraction/general_extraction/general_extraction_1.5B/result/S_general_math_apps_logic_phase1.pt", False),
        ("S_combo", "extraction/combo_extraction/combo_extraction_1.5B/result/S_combo.pt", True),
    ],
    "7B": [
        ("S_math", "extraction/math_extraction/math_extraction_7B/result/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt", True),
        ("S_code", "extraction/code_extraction/code_extraction_7B/result/apps_v_code.pt", True),
        ("S_logic", "extraction/logic_extraction/logic_extraction_7B/result/logiqa2_v_logic.pt", True),
        ("S_general", "extraction/general_extraction/general_extraction_7B/result/S_general_7b.pt", False),  # not built
        ("S_combo", "extraction/combo_extraction/combo_extraction_7B/result/S_combo.pt", True),
    ],
}

# S_combo_<size> is the norm reference: ‖S_combo‖ = mean of the three domain norms.
NORM_REFERENCE = "S_combo"

# Transitional: the packaged 1.5B logic vector is being renamed
# logiqa_v_logic.pt -> logiqa2_v_logic.pt (same bytes). Accept the old name
# until the rename lands.
LEGACY_NAMES = {"logiqa2_v_logic.pt": "logiqa_v_logic.pt"}


class Checker:
    """Collects pass/fail lines so every check runs before anything exits."""

    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(f"[{PASS if ok else FAIL}] {label}" + (f"  — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)

    def skip(self, label: str, detail: str = "") -> None:
        """Record an informational skip: not a failure, not a pass."""
        print(f"[{SKIP}] {label}" + (f"  — {detail}" if detail else ""))


def load_meta(vector_path: str) -> Dict[str, object]:
    """Read the sidecar meta for a vector, or {} if it is missing."""
    meta_path = os.path.splitext(vector_path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as handle:
        return json.load(handle)


def infer_model_size(vector_dir: str, prefix: str) -> str:
    """Infer 1.5B / 7B from ``--dir`` / ``--prefix``.

    ``control_extraction_7B`` anywhere in the directory, or a prefix ending in
    ``_7b``, means 7B; everything else is the 1.5B default.
    """
    parts = os.path.normpath(vector_dir).lower().split(os.sep)
    if prefix.lower().endswith("_7b") or any(p.endswith("_7b") for p in parts):
        return "7B"
    return "1.5B"


def resolve_reference_path(path: str) -> str:
    """Return ``path`` if it exists, else its legacy name if that exists."""
    if os.path.exists(path):
        return path
    base = os.path.basename(path)
    legacy = LEGACY_NAMES.get(base)
    if legacy:
        candidate = os.path.join(os.path.dirname(path), legacy)
        if os.path.exists(candidate):
            return candidate
    return path


def canonical_ref_name(name: str) -> str:
    """``S_math_7b`` / ``S_math_1.5B`` / ``S_math`` -> ``s_math``."""
    name = name.strip().lower()
    for suffix in ("_7b", "_1.5b", "_1_5b"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_size_references(
    model_size: str, checker: Checker
) -> Tuple[Dict[str, torch.Tensor], Optional[float]]:
    """Load every packaged vector of ``model_size`` that exists.

    Records a FAIL for each required vector that is missing, a skip for a
    missing optional one, and returns ``(references, norm_of_S_combo)``.
    """
    refs: Dict[str, torch.Tensor] = {}
    combo_norm: Optional[float] = None
    for name, path, required in REFERENCE_VECTORS[model_size]:
        resolved = resolve_reference_path(path)
        if os.path.exists(resolved):
            refs[name] = load_reference(resolved)
            note = "" if resolved == path else f"  (legacy name; expected {os.path.basename(path)})"
            print(f"  {name:<10} {resolved}{note}")
            if name == NORM_REFERENCE:
                combo_norm = float(refs[name].norm())
        elif required:
            checker.check(False, f"{model_size} reference {name} present", f"missing: {path}")
        else:
            checker.skip(f"{model_size} reference {name} present", f"not built ({path})")
    return refs, combo_norm


def rebuild_from_meta(meta: Dict[str, object], dim: int) -> torch.Tensor:
    """Regenerate a vector from its recorded seed, replaying any rejections.

    References are not reloaded — the rebuild does not re-run the cosine screen,
    it replays the generator. A rejected draw still advances the generator, so
    reaching the accepted draw means discarding ``attempts - 1`` draws first.
    Skipping that replay would silently rebuild the *first* draw whenever a
    rejection had occurred, and the check would fail for a reason that has
    nothing to do with reproducibility.

    Uses ``target_norm``, not ``vector_norm``: the latter is rounded for
    readability, and multiplying by a rounded scale reproduces a nearly
    identical vector rather than a bit-identical one.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(meta["seed"]))
    for _ in range(int(meta.get("attempts", 1)) - 1):
        draw_iso(dim, generator)
    return draw_iso(dim, generator) * float(meta["target_norm"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the R_iso control draws.")
    parser.add_argument("--dir", default="extraction/control_extraction/control_extraction_1.5B/result", help="Vector directory.")
    parser.add_argument("--prefix", default="R_iso", help="Vector filename prefix.")
    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZES,
        default=None,
        help="Which size's packaged vectors to check against; inferred from --dir/--prefix if omitted.",
    )
    parser.add_argument(
        "--max-cos",
        type=float,
        default=None,
        help="Override the threshold; defaults to each vector's recorded max_cos.",
    )
    args = parser.parse_args()

    model_size = args.model_size or infer_model_size(args.dir, args.prefix)

    paths = sorted(glob.glob(os.path.join(args.dir, f"{args.prefix}_seed*.pt")))
    if not paths:
        print(f"No vectors matching {args.prefix}_seed*.pt in {args.dir}")
        sys.exit(1)

    print("=" * 70)
    print(f" Verifying {len(paths)} control vector(s) in {args.dir}")
    print(f" model size {model_size}" + ("" if args.model_size else "  (inferred from --dir/--prefix)"))
    print("=" * 70)

    checker = Checker()
    print(f"\n--- packaged {model_size} reference vectors ---")
    seal, combo_norm = load_size_references(model_size, checker)
    # Names of the packaged vectors that exist: every one of these must appear
    # in a draw's recorded screen (the 1.5B protocol).
    expected_screen = {canonical_ref_name(n): n for n in seal}

    drawn: Dict[str, torch.Tensor] = {}

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n--- {name} ---")
        meta = load_meta(path)
        checker.check(bool(meta), f"{name}: meta present")

        vector = torch.load(path, map_location="cpu", weights_only=True)
        checker.check(
            vector.dim() == 1, f"{name}: is 1-D", f"shape {tuple(vector.shape)}"
        )
        checker.check(
            vector.dtype == torch.float32, f"{name}: float32", str(vector.dtype)
        )

        vector = vector.float().flatten()
        drawn[name] = vector

        if meta:
            target = float(meta["vector_norm"])
            got = float(vector.norm())
            checker.check(
                abs(got - target) <= NORM_TOL,
                f"{name}: norm matches meta",
                f"{got:.6f} vs {target:.6f}",
            )
            checker.check(
                int(meta["vector_dim"]) == int(vector.numel()),
                f"{name}: dim matches meta",
            )
            checker.check(
                float(meta["apply_coef"]) == -1.0 and int(meta["apply_layer"]) == 20,
                f"{name}: records layer 20 / coef -1.0",
            )

        # Protocol: the target norm is ‖S_combo_<size>‖ (= mean of the three
        # domain-vector norms), set with --like, never a typed number.
        if combo_norm is not None:
            got = float(vector.norm())
            checker.check(
                abs(got - combo_norm) <= NORM_TOL,
                f"{name}: protocol — norm-matched to {NORM_REFERENCE} ({model_size}, mean of the three domain norms)",
                f"{got:.4f} vs ‖{NORM_REFERENCE}‖ {combo_norm:.4f}"
                + (f", built with norm_mode={meta.get('norm_mode')!r}" if meta else ""),
            )
        else:
            checker.check(
                False,
                f"{name}: protocol — norm-matched to {NORM_REFERENCE} ({model_size})",
                f"{NORM_REFERENCE} for {model_size} is missing, cannot check",
            )

        # Protocol: the build screened against every packaged vector of its size.
        if meta:
            recorded = {canonical_ref_name(n) for n in (meta.get("references") or {})}
            missing_screen = sorted(
                display for key, display in expected_screen.items() if key not in recorded
            )
            checker.check(
                not missing_screen,
                f"{name}: protocol — build screened against every packaged {model_size} vector",
                (
                    f"recorded {sorted(meta.get('references') or {})}"
                    + (f"; not screened against {missing_screen}" if missing_screen else "")
                ),
            )

        max_cos = args.max_cos if args.max_cos is not None else float(
            meta.get("max_cos", 0.08)
        )
        # Compare only against references in the same hidden dimension (the
        # size table guarantees that; the filter is a belt-and-braces guard).
        # Any extra reference path recorded in the draw's own meta that still
        # exists on disk is added too. A cross-dim cosine is not defined.
        refs = {n: v for n, v in seal.items() if v.numel() == vector.numel()}
        for ref_name, ref_path in (meta.get("references") or {}).items():
            if canonical_ref_name(ref_name) in {canonical_ref_name(n) for n in refs}:
                continue
            if os.path.exists(ref_path):
                ref_vec = load_reference(ref_path)
                if ref_vec.numel() == vector.numel():
                    refs[ref_name] = ref_vec
        if refs:
            worst_name, worst = "", 0.0
            for seal_name, seal_vec in refs.items():
                c = abs(cosine(vector, seal_vec))
                if c > worst:
                    worst_name, worst = seal_name, c
            checker.check(
                worst <= max_cos,
                f"{name}: near-orthogonal to all {len(refs)} same-dim reference vectors",
                f"worst |cos| {worst:.4f} vs {worst_name}, threshold {max_cos}",
            )
        else:
            checker.check(
                False,
                f"{name}: no same-dim reference vectors found to check against",
            )

        if meta:
            rebuilt = rebuild_from_meta(meta, int(vector.numel()))
            identical = torch.equal(rebuilt, vector)
            built_version = meta.get("torch_version")
            same_version = built_version == torch.__version__

            if identical:
                checker.check(True, f"{name}: reproducible from seed {meta['seed']}", "bit-identical")
            elif same_version:
                # Same torch version, same seed, different output: a real bug
                # (e.g. the replay didn't account for a rejected draw), not an
                # RNG quirk. This must fail loudly.
                checker.check(
                    False,
                    f"{name}: reproducible from seed {meta['seed']}",
                    f"DIFFERS despite matching torch {torch.__version__} — investigate",
                )
            else:
                # PyTorch does not guarantee bit-identical CPU RNG output
                # across versions for the same seed, only within one version.
                # A vector built on 2.13.0 and checked under 2.5.1 (this
                # repo's pinned version, from vllm) can legitimately diverge
                # here with no defect in the vector itself. Fall back to
                # confirming the rebuild is at least the *same direction and
                # magnitude* the meta claims, which is what actually matters
                # for steering: eval scripts torch.load() the committed .pt
                # bytes directly, they never re-derive it from the seed.
                close = (
                    abs(cosine(rebuilt, vector) - 1.0) <= COSINE_TOL
                    and abs(float(rebuilt.norm()) - float(vector.norm())) <= NORM_TOL
                )
                checker.skip(
                    f"{name}: reproducible from seed {meta['seed']}",
                    f"torch {built_version} built it, {torch.__version__} is checking it — "
                    f"bit-identity isn't guaranteed across versions; "
                    + (
                        f"rebuild matches direction/norm (cos={cosine(rebuilt, vector):.8f})"
                        if close
                        else "WARNING: rebuild also fails the looser direction/norm check"
                    ),
                )
                if not close:
                    # The version excuse doesn't cover this: something is
                    # actually wrong, not just RNG drift.
                    checker.check(
                        False,
                        f"{name}: rebuild direction/norm (cross-version fallback)",
                        f"cos={cosine(rebuilt, vector):.8f}",
                    )

    print("\n--- pairwise between draws ---")
    names = sorted(drawn)
    pairs: List[Tuple[str, str, float]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((a, b, cosine(drawn[a], drawn[b])))
    for a, b, c in pairs:
        checker.check(abs(c) <= 0.08, f"{a} vs {b}", f"cos {c:+.4f}")
    if len(names) > 1:
        checker.check(
            all(not torch.equal(drawn[a], drawn[b]) for a, b, _ in pairs),
            "all draws are distinct tensors",
        )

    print("\n" + "=" * 70)
    if checker.failures:
        print(f" {len(checker.failures)} CHECK(S) FAILED — do not spend GPU time")
        for failure in checker.failures:
            print(f"   - {failure}")
        if model_size == "7B":
            print(
                " Rebuild with the 1.5B protocol (after the 7B S_logic + S_combo_7b rebuilds):\n"
                "   MODEL_SIZE=7B bash extraction/control_extraction/control_extraction_1.5B/pipeline/build_random_vectors.sh"
            )
        print("=" * 70)
        sys.exit(1)
    print(" All checks passed. Vectors are ready to evaluate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
