"""Check that a retrained calibration matches the committed one.

    python scripts/check_calibration.py committed.json retrained.json

The published image is trained from source in CI, so it can differ slightly from the model that
was evaluated on the test set. This fails if the calibrator, the threshold or the Brier score moved
by more than a small tolerance, so a materially different model is never published.
"""

import json
import sys

THRESHOLD_TOLERANCE = 0.03
BRIER_RELATIVE_TOLERANCE = 0.05


def compare(committed: dict, retrained: dict) -> list[str]:  # type: ignore[type-arg]
    """Reasons the retrained calibration differs materially (empty when it matches)."""
    problems = []
    if committed["method"] != retrained["method"]:
        problems.append(f"calibrator changed: {committed['method']} -> {retrained['method']}")
    old, new = committed["decision"]["threshold"], retrained["decision"]["threshold"]
    if abs(old - new) > THRESHOLD_TOLERANCE:
        problems.append(f"threshold moved from {old} to {new}")
    method = committed["method"]
    old_b = committed["calibrators"][method]["brier"]
    new_b = retrained["calibrators"][method]["brier"]
    if abs(new_b - old_b) / old_b > BRIER_RELATIVE_TOLERANCE:
        problems.append(f"Brier score moved from {old_b} to {new_b}")
    return problems


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        committed = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        retrained = json.load(f)
    problems = compare(committed, retrained)
    print(
        f"committed:  {committed['method']}, threshold {committed['decision']['threshold']}, "
        f"Brier {committed['calibrators'][committed['method']]['brier']}"
    )
    print(
        f"retrained:  {retrained['method']}, threshold {retrained['decision']['threshold']}, "
        f"Brier {retrained['calibrators'][retrained['method']]['brier']}"
    )
    if problems:
        print("MISMATCH: " + "; ".join(problems))
        sys.exit(1)
    print("OK: the retrained calibration matches the committed one")


if __name__ == "__main__":
    main()
