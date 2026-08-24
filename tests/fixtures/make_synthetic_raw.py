"""Generate a small synthetic raw loans dataset matching the LendingClub accepted-loans schema
(all VARCHAR, exactly like download.py's real output) so the full pipeline can be exercised in
CI without needing the actual multi-gigabyte, Kaggle-gated dataset.

Run standalone or via CI: writes directly to data/raw/loans_raw.parquet, the same path
etl_duckdb.py expects.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "loans_raw.parquet"
N_ROWS = 6000
RANDOM_SEED = 42


def generate_synthetic_raw(n: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a synthetic DataFrame with realistic-enough values across LendingClub's raw columns,
    including a genuine grade-to-default-rate risk gradient so downstream model tests see real signal.
    """
    rng = np.random.default_rng(seed)

    grades = np.array(list("ABCDEFG"))
    grade = rng.choice(grades, size=n, p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.06, 0.04])
    sub_grade = [g + str(rng.integers(1, 6)) for g in grade]

    years = rng.integers(2015, 2019, size=n)
    quarters = rng.integers(1, 5, size=n)
    months_by_q = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    issue_d = [
        f"{month_names[months_by_q[q][rng.integers(0, 3)] - 1]}-{y}"
        for y, q in zip(years, quarters)
    ]
    earliest_cr_line = [f"Jan-{y - rng.integers(2, 20)}" for y in years]

    grade_risk = {"A": 0.03, "B": 0.07, "C": 0.13, "D": 0.20, "E": 0.28, "F": 0.35, "G": 0.45}
    default_prob = np.array([grade_risk[g] for g in grade])

    is_resolved = rng.random(n) < 0.7
    is_default = is_resolved & (rng.random(n) < default_prob)
    loan_status = np.where(~is_resolved, "Current", np.where(is_default, "Charged Off", "Fully Paid"))

    loan_amnt = rng.integers(2000, 35000, size=n).astype(float)
    funded_amnt = loan_amnt
    term = rng.choice([" 36 months", " 60 months"], size=n)
    int_rate = np.round(5 + default_prob * 40 + rng.normal(0, 1.5, size=n), 2)
    installment = np.round(funded_amnt / 24 + rng.normal(0, 20, size=n), 2)
    annual_inc = np.round(rng.lognormal(mean=11, sigma=0.4, size=n), 2)
    dti = np.round(rng.uniform(2, 35, size=n), 2)
    revol_util = np.round(rng.uniform(0, 100, size=n), 1)
    emp_length = rng.choice(["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years", "n/a"], size=n)

    recoveries = np.where(is_default, np.round(funded_amnt * rng.uniform(0.0, 0.4, size=n), 2), 0.0)
    collection_recovery_fee = np.round(recoveries * 0.15, 2)
    total_rec_prncp = np.where(is_default, np.round(funded_amnt * rng.uniform(0.05, 0.5, size=n), 2), funded_amnt)

    df = pd.DataFrame({
        "id": np.arange(1, n + 1).astype(str),
        "loan_amnt": loan_amnt,
        "funded_amnt": funded_amnt,
        "term": term,
        "int_rate": [f"{v}%" for v in int_rate],
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_title": "Engineer",
        "emp_length": emp_length,
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], size=n),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], size=n),
        "issue_d": issue_d,
        "loan_status": loan_status,
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other"], size=n),
        "title": "loan",
        "zip_code": "123xx",
        "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], size=n),
        "dti": dti,
        "delinq_2yrs": rng.integers(0, 3, size=n),
        "earliest_cr_line": earliest_cr_line,
        "inq_last_6mths": rng.integers(0, 4, size=n),
        "open_acc": rng.integers(2, 20, size=n),
        "pub_rec": rng.integers(0, 2, size=n),
        "revol_bal": np.round(rng.uniform(0, 40000, size=n), 2),
        "revol_util": [f"{v}%" for v in revol_util],
        "total_acc": rng.integers(5, 40, size=n),
        "initial_list_status": rng.choice(["w", "f"], size=n),
        "application_type": "Individual",
        "mort_acc": rng.integers(0, 5, size=n),
        "pub_rec_bankruptcies": rng.integers(0, 2, size=n),
        "tot_cur_bal": np.round(rng.uniform(0, 200000, size=n), 2),
        "total_bal_ex_mort": np.round(rng.uniform(0, 100000, size=n), 2),
        "total_bc_limit": np.round(rng.uniform(1000, 50000, size=n), 2),
        "tot_hi_cred_lim": np.round(rng.uniform(10000, 300000, size=n), 2),
        "avg_cur_bal": np.round(rng.uniform(0, 30000, size=n), 2),
        "bc_util": [f"{v}" for v in np.round(rng.uniform(0, 100, size=n), 1)],
        "bc_open_to_buy": np.round(rng.uniform(0, 20000, size=n), 2),
        "num_actv_bc_tl": rng.integers(0, 10, size=n),
        "num_tl_90g_dpd_24m": rng.integers(0, 3, size=n),
        "mo_sin_old_rev_tl_op": rng.integers(12, 300, size=n),
        "percent_bc_gt_75": np.round(rng.uniform(0, 100, size=n), 1),
        "acc_open_past_24mths": rng.integers(0, 10, size=n),
        "out_prncp": np.where(is_resolved, 0.0, np.round(funded_amnt * rng.uniform(0.3, 0.9, size=n), 2)),
        "total_pymnt": np.round(funded_amnt * rng.uniform(0.1, 1.2, size=n), 2),
        "total_rec_prncp": total_rec_prncp,
        "total_rec_int": np.round(funded_amnt * rng.uniform(0.02, 0.2, size=n), 2),
        "total_rec_late_fee": np.round(rng.uniform(0, 50, size=n), 2),
        "recoveries": recoveries,
        "collection_recovery_fee": collection_recovery_fee,
        "last_pymnt_amnt": np.round(rng.uniform(50, 1000, size=n), 2),
    })
    return df.astype(str)  # mimic ALL_VARCHAR raw parquet from download.py


def main() -> None:
    df = generate_synthetic_raw()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df", df)
        con.execute(f"COPY df TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    print(f"Wrote {len(df)} synthetic rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
