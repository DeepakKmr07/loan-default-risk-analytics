"""Macroeconomic stress testing engine (IFRS 9 / CCAR-style scenarios).

Simulates portfolio-level Expected Loss under three scenarios by shocking PD, LGD, and EAD:
    Baseline:          unadjusted model predictions (PD x LGD x EAD)
    Adverse:           PD x1.25, LGD x1.15 (10% collateral haircut), EAD unchanged
    Severely Adverse:  PD x1.60, LGD x1.25 (25% collateral haircut), EAD + 20% CCF on undrawn exposure

Note on the CCF / undrawn-facility shock: LendingClub originates fixed-amortization installment
loans that are fully disbursed at origination, so there is no revolving/undrawn commitment in
this dataset (unlike a credit card or HELOC book). The EAD formula below is written generally as
`funded_ead + ccf * undrawn_amount` so the mechanism is correct if ever applied to a portfolio
with revolving exposure; here `undrawn_amount` is 0 for every loan, so EAD is unaffected by the
CCF term in practice for this dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACT_PORTFOLIO_PATH = PROJECT_ROOT / "data" / "curated" / "Fact_Loan_Risk_Portfolio.parquet"
STRESS_OUTPUT_PATH = PROJECT_ROOT / "data" / "curated" / "Fact_Stress_Test_Scenarios.parquet"

SCENARIOS: dict[str, dict[str, float]] = {
    "Baseline": {"pd_multiplier": 1.00, "lgd_multiplier": 1.00, "ccf_undrawn": 0.00},
    "Adverse": {"pd_multiplier": 1.25, "lgd_multiplier": 1.15, "ccf_undrawn": 0.00},
    "Severely Adverse": {"pd_multiplier": 1.60, "lgd_multiplier": 1.25, "ccf_undrawn": 0.20},
}


def load_portfolio(path: Path = FACT_PORTFOLIO_PATH) -> pd.DataFrame:
    """Load the scored fact table needed to re-derive stressed PD/LGD/EAD/EL."""
    con = duckdb.connect()
    try:
        df = con.execute(
            f"""
            SELECT loan_id, borrower_key, vintage_key, credit_grade_key, ead, pd, lgd
            FROM read_parquet('{path.as_posix()}')
            """
        ).df()
    finally:
        con.close()
    df["undrawn_amount"] = 0.0  # see module docstring: no revolving exposure in this dataset
    return df


def apply_scenario(df: pd.DataFrame, scenario_name: str, params: dict[str, float]) -> pd.DataFrame:
    """Recompute stressed PD, LGD, EAD, and Expected Loss for one scenario."""
    out = df.copy()
    out["scenario"] = scenario_name
    out["pd"] = (out["pd"] * params["pd_multiplier"]).clip(upper=1.0)
    out["lgd"] = (out["lgd"] * params["lgd_multiplier"]).clip(upper=1.0)
    out["ead"] = out["ead"] + params["ccf_undrawn"] * out["undrawn_amount"]
    out["expected_loss"] = out["pd"] * out["lgd"] * out["ead"]
    return out.drop(columns=["undrawn_amount"])


def run_stress_test(
    input_path: Path = FACT_PORTFOLIO_PATH, output_path: Path = STRESS_OUTPUT_PATH
) -> pd.DataFrame:
    """Run all scenarios and write the combined long-format result to Parquet for Power BI."""
    portfolio = load_portfolio(input_path)
    logger.info("Running %s scenarios over %s loans", len(SCENARIOS), f"{len(portfolio):,}")

    results = [apply_scenario(portfolio, name, params) for name, params in SCENARIOS.items()]
    combined = pd.concat(results, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("stress_results", combined)
        con.execute(f"COPY stress_results TO '{output_path.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    logger.info("Wrote %s stress-test rows to %s", f"{len(combined):,}", output_path)

    for scenario, group in combined.groupby("scenario", sort=False):
        total_ead = group["ead"].sum()
        weighted_pd = (group["pd"] * group["ead"]).sum() / total_ead
        total_el = group["expected_loss"].sum()
        logger.info(
            "[%s] Total EAD=$%s  Weighted PD=%.2f%%  Total EL=$%s",
            scenario, f"{total_ead:,.0f}", weighted_pd * 100, f"{total_el:,.0f}",
        )
    return combined


if __name__ == "__main__":
    run_stress_test()
