"""Model governance report: calibration diagnostics, discrimination, and population stability.

Regulatory context (Basel / IFRS 9): lenders must demonstrate that PD estimates are
well-calibrated, that discrimination (Gini / KS) holds up out-of-time, and that neither the
model's predictions nor the underlying credit features have drifted materially across
origination vintages since model development (Population Stability Index).

Reuses the exact chronological train/calibration/OOT splits from train_pd.py and train_lgd.py
so these diagnostics are computed on the same out-of-time population those scripts evaluated.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score, root_mean_squared_error

from src.models import train_lgd, train_pd
from src.models.features import LGD_FEATURES, PD_FEATURES
from src.models.governance_utils import (
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    classify_psi,
    compute_ks_statistic,
    compute_psi,
    reliability_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
CALIBRATION_PLOT_PATH = REPORTS_DIR / "calibration_curve.png"
PSI_REPORT_PATH = REPORTS_DIR / "psi_report.csv"
MODEL_RISK_SUMMARY_PATH = REPORTS_DIR / "model_risk_summary.json"

# A curated subset of "key credit features" to monitor for drift, rather than all model inputs.
KEY_MONITORING_FEATURES = ["int_rate", "dti", "revol_util", "annual_inc", "loan_to_income", "credit_utilization"]


def _add_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """Derive an 'issue_year'Q'issue_quarter' vintage label from issue_date, for PSI grouping."""
    df = df.copy()
    df["vintage"] = df["issue_date"].dt.year.astype(str) + "Q" + df["issue_date"].dt.quarter.astype(str)
    return df


def plot_calibration_curve(curve: pd.DataFrame, output_path: Path) -> None:
    """Plot the reliability diagram: mean predicted PD vs empirical default rate, per decile."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curve["predicted_pd"], curve["empirical_default_rate"], marker="o", label="Model (deciles)")
    upper = float(max(curve["predicted_pd"].max(), curve["empirical_default_rate"].max()) * 1.05)
    ax.plot([0, upper], [0, upper], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted PD (decile)")
    ax.set_ylabel("Empirical Default Rate (decile)")
    ax.set_title("PD Calibration Reliability Diagram (Out-of-Time Test)")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compute_vintage_psi(baseline: pd.Series, monitoring_df: pd.DataFrame, column: str) -> list[dict]:
    """PSI of `column` for each vintage in `monitoring_df`, against a fixed development-sample baseline."""
    results = []
    for vintage, group in monitoring_df.groupby("vintage"):
        psi = compute_psi(baseline.to_numpy(), group[column].to_numpy())
        results.append({"vintage": vintage, "n": len(group), "psi": round(psi, 4), "flag": classify_psi(psi)})
    return sorted(results, key=lambda r: r["vintage"])


def main() -> None:
    pd_model = joblib.load(train_pd.PD_MODEL_PATH)
    lgd_model = joblib.load(train_lgd.LGD_MODEL_PATH)
    logger.info("Loaded PD model from %s and LGD model from %s", train_pd.PD_MODEL_PATH, train_lgd.LGD_MODEL_PATH)

    pd_df = _add_vintage(train_pd.load_labeled_data())
    pd_train, pd_calib, pd_oot = train_pd.time_based_split(pd_df)

    lgd_df = train_lgd.load_defaulted_loans()
    lgd_train, lgd_oot = train_lgd.time_based_split(lgd_df)

    # --- Calibration diagnostics (reliability diagram + Brier score) ---
    oot_scores = pd_model.predict_proba(pd_oot[PD_FEATURES])[:, 1]
    curve = reliability_curve(pd_oot[train_pd.TARGET].to_numpy(), oot_scores, n_bins=10)
    plot_calibration_curve(curve, CALIBRATION_PLOT_PATH)
    brier = brier_score_loss(pd_oot[train_pd.TARGET], oot_scores)
    logger.info("Wrote calibration reliability diagram to %s", CALIBRATION_PLOT_PATH)

    # --- Discrimination metrics: ROC-AUC, Gini, KS ---
    roc_auc = roc_auc_score(pd_oot[train_pd.TARGET], oot_scores)
    gini = 2 * roc_auc - 1
    ks = compute_ks_statistic(pd_oot[train_pd.TARGET].to_numpy(), oot_scores)
    logger.info("OOT ROC-AUC=%.4f  Gini=%.4f  KS=%.4f  Brier=%.4f", roc_auc, gini, ks, brier)

    # --- LGD metrics ---
    lgd_preds = lgd_model.predict(lgd_oot[LGD_FEATURES]).clip(0.0, 1.0)
    lgd_rmse = root_mean_squared_error(lgd_oot[train_lgd.TARGET], lgd_preds)
    lgd_mae = mean_absolute_error(lgd_oot[train_lgd.TARGET], lgd_preds)
    logger.info("LGD OOT RMSE=%.4f  MAE=%.4f", lgd_rmse, lgd_mae)

    # --- PSI: predicted PD across vintages, baseline = training (development) sample ---
    train_pd_scores = pd.Series(pd_model.predict_proba(pd_train[PD_FEATURES])[:, 1])
    monitoring_df = pd.concat([pd_calib, pd_oot], ignore_index=True)
    monitoring_df["predicted_pd"] = pd_model.predict_proba(monitoring_df[PD_FEATURES])[:, 1]
    pd_psi_rows = compute_vintage_psi(train_pd_scores, monitoring_df, "predicted_pd")

    # --- PSI: key credit features across vintages, same baseline population ---
    feature_psi_rows: dict[str, list[dict]] = {
        feature: compute_vintage_psi(pd_train[feature], monitoring_df, feature)
        for feature in KEY_MONITORING_FEATURES
    }

    psi_records = [{"metric": "predicted_pd", **row} for row in pd_psi_rows]
    for feature, rows in feature_psi_rows.items():
        psi_records.extend({"metric": feature, **row} for row in rows)
    psi_df = pd.DataFrame(psi_records)
    PSI_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    psi_df.to_csv(PSI_REPORT_PATH, index=False)
    logger.info("Wrote PSI report (%s rows) to %s", len(psi_df), PSI_REPORT_PATH)

    flagged = psi_df[psi_df["flag"] != "stable"]
    if len(flagged):
        logger.warning("PSI flags raised on %s (metric, vintage) pairs:\n%s", len(flagged), flagged.to_string(index=False))

    # --- Governance summary artifact ---
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oot_test_period": {
            "start": str(pd_oot["issue_date"].min().date()),
            "end": str(pd_oot["issue_date"].max().date()),
        },
        "pd_model": {
            "roc_auc": round(float(roc_auc), 4),
            "gini_coefficient": round(float(gini), 4),
            "ks_statistic": round(float(ks), 4),
            "brier_score": round(float(brier), 4),
        },
        "lgd_model": {
            "rmse": round(float(lgd_rmse), 4),
            "mae": round(float(lgd_mae), 4),
        },
        "population_stability": {
            "moderate_threshold": PSI_MODERATE_THRESHOLD,
            "significant_threshold": PSI_SIGNIFICANT_THRESHOLD,
            "n_flagged_moderate_or_worse": int(len(flagged)),
            "report_csv": str(PSI_REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "artifacts": {
            "calibration_curve_png": str(CALIBRATION_PLOT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    MODEL_RISK_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote model risk summary to %s", MODEL_RISK_SUMMARY_PATH)


if __name__ == "__main__":
    main()
