"""Diff two golden-metric recordings: numeric drift and timing change.

    python scripts/compare_golden_metrics.py <before_dir> <after_dir> [rel_tol]

Exits non-zero on drift, so it can gate a migration phase.
"""

import json
import os
import sys

GATE_RTOL = 1e-9
DEFAULT_RTOL = 1e-6

#: Below this a value is numerically zero and relative drift is meaningless.
ATOL = 1e-12


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def _drift(before, after, rtol):
    """Worst tolerance breach, scored as numpy does: ``ATOL + rtol * |expected|``."""
    a = dict(_flatten(before))
    b = dict(_flatten(after))
    missing = set(a) ^ set(b)
    worst_ratio, worst_at, worst_abs = 0.0, "", 0.0
    for key in set(a) & set(b):
        x, y = a[key], b[key]
        gap = abs(x - y)
        ratio = gap / (ATOL + rtol * abs(y))
        if gap > worst_abs:
            worst_abs = gap
        if ratio > worst_ratio:
            worst_ratio, worst_at = ratio, key
    return worst_ratio, worst_at, worst_abs, sorted(missing)


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) < 3:
        raise SystemExit(
            "usage: compare_golden_metrics.py <before_dir> <after_dir> [rel_tol]"
        )
    before_dir, after_dir = argv[1], argv[2]
    tol = float(argv[3]) if len(argv) > 3 else DEFAULT_RTOL

    failures = 0
    for name in sorted(os.listdir(before_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(before_dir, name)) as handle:
            before = json.load(handle)
        after_path = os.path.join(after_dir, name)
        if not os.path.exists(after_path):
            print(f"{name}: MISSING in {after_dir}")
            failures += 1
            continue
        with open(after_path) as handle:
            after = json.load(handle)

        print(f"\n{name}  (n={before['nodes']}, m={before['edges']})")
        for run, before_run in sorted(before["runs"].items()):
            after_run = after["runs"][run]

            gate_ratio, gate_at, gate_abs, _ = _drift(
                before_run["error_features"], after_run["error_features"], GATE_RTOL
            )
            gate_ok = gate_ratio <= 1.0
            print(
                f"  {run}"
                f"\n    gate scalars   max |diff| {gate_abs:.3e}  "
                f"{'OK' if gate_ok else 'FAIL'}"
                f"{'' if gate_ok else f' at {gate_at} ({gate_ratio:.1f}x tolerance)'}"
            )
            failures += 0 if gate_ok else 1

            worst, worst_at, worst_abs, missing = _drift(
                before_run["analysis"], after_run["analysis"], tol
            )
            ok = worst <= 1.0 and not missing
            print(
                f"    analysis       max |diff| {worst_abs:.3e}  "
                f"{'OK' if ok else 'DRIFT'}"
                f"{'' if ok else f' at {worst_at} ({worst:.1f}x tolerance)'}"
            )
            if missing:
                print(f"    keys only on one side: {missing[:5]}")
            failures += 0 if ok else 1

            for label, seconds in sorted(before_run["seconds"].items()):
                new = after_run["seconds"][label]
                ratio = new / seconds if seconds else float("inf")
                verdict = "faster" if ratio < 1 else "slower"
                print(
                    f"    {label:24} {seconds:7.2f}s -> {new:7.2f}s  "
                    f"{ratio:5.2f}x {verdict}"
                )

    print(f"\n{'PASS' if failures == 0 else f'{failures} check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
