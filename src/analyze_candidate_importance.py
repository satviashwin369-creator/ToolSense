
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "toolsense_model_candidate.joblib"
OUTPUT_PATH = ROOT / "reports" / "candidate_feature_importance.csv"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            "The saved model does not expose feature_importances_."
        )

    importances = model.feature_importances_

    if len(features) != len(importances):
        raise ValueError(
            "Feature names and importance values have different lengths."
        )

    results = pd.DataFrame({
        "feature": features,
        "importance": importances,
    })

    results = results.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1),
    )

    results["importance_percent"] = (
        results["importance"] * 100
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("=" * 64)
    print("TOOLSENSE - CANDIDATE FEATURE IMPORTANCE")
    print("=" * 64)
    print(f"Model: {artifact.get('model_type')}")
    print(f"Feature count: {len(features)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nTOP 15 FEATURES")
    print(
        results.head(15).to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format,
                "importance_percent": "{:.3f}".format,
            },
        )
    )

    print("\nSENSOR-GROUP SUMMARY")
    sensor_prefixes = [
        "smcAC",
        "smcDC",
        "vib_table",
        "vib_spindle",
        "AE_table",
        "AE_spindle",
    ]

    grouped = []

    for sensor in sensor_prefixes:
        total = results.loc[
            results["feature"].str.startswith(sensor + "_"),
            "importance",
        ].sum()

        grouped.append({
            "sensor": sensor,
            "importance": total,
            "importance_percent": total * 100,
        })

    grouped.append({
        "sensor": "Machining conditions",
        "importance": results.loc[
            results["feature"].isin(["DOC", "feed"]),
            "importance",
        ].sum(),
        "importance_percent": results.loc[
            results["feature"].isin(["DOC", "feed"]),
            "importance",
        ].sum() * 100,
    })

    grouped_df = pd.DataFrame(grouped).sort_values(
        "importance",
        ascending=False,
    )

    print(
        grouped_df.to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format,
                "importance_percent": "{:.3f}".format,
            },
        )
    )

    print("\nNote:")
    print(
        "These are impurity-based Extra Trees importances. "
        "They describe model reliance, not causal effects."
    )


if __name__ == "__main__":
    main()
