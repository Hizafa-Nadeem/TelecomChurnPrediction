"""Accept or reject the model against the pass criteria fixed in `params.yaml` (`gate` section).

    python -m churn.gate --config configs/params.yaml \
        --result reports/evaluation/test_evaluation.json

Exits with code 1 if any check fails, so it can guard a merge.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from churn.config import GateParams, load_params


def check(result: dict[str, Any], gate: GateParams) -> list[tuple[str, float, str, bool]]:
    """(name, value, requirement, passed) for each criterion."""
    m = result["metrics"]
    logreg = result["baselines"]["logreg_pr_auc"]
    return [
        (
            "PR-AUC",
            m["pr_auc"]["value"],
            f">= {gate.min_pr_auc}",
            m["pr_auc"]["value"] >= gate.min_pr_auc,
        ),
        (
            "ROC-AUC",
            m["roc_auc"]["value"],
            f">= {gate.min_roc_auc}",
            m["roc_auc"]["value"] >= gate.min_roc_auc,
        ),
        (
            "Recall at threshold",
            m["recall"]["value"],
            f">= {gate.min_recall}",
            m["recall"]["value"] >= gate.min_recall,
        ),
        (
            "PR-AUC vs logistic regression",
            m["pr_auc"]["value"],
            f"> {logreg:.4f}",
            m["pr_auc"]["value"] > logreg,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = check(result, load_params(args.config).gate)
    for name, value, requirement, passed in rows:
        print(f"{name:32s} {value:7.4f}  {requirement:>10s}  {'PASS' if passed else 'FAIL'}")
    ok = all(passed for *_, passed in rows)
    print(f"GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
