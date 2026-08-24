"""Shared feature lists for PD/LGD training and inference, so all three stay in sync."""

NUMERIC_FEATURES: list[str] = [
    "loan_amnt",
    "funded_amnt",
    "term_months",
    "int_rate",
    "installment",
    "emp_length_years",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tot_cur_bal",
    "total_bal_ex_mort",
    "total_bc_limit",
    "tot_hi_cred_lim",
    "avg_cur_bal",
    "bc_util",
    "bc_open_to_buy",
    "num_actv_bc_tl",
    "num_tl_90g_dpd_24m",
    "mo_sin_old_rev_tl_op",
    "percent_bc_gt_75",
    "acc_open_past_24mths",
    "credit_history_months",
    "installment_to_income",
    "loan_to_income",
    "revol_bal_to_income",
    "credit_utilization",
    "bankcard_utilization",
]

CATEGORICAL_FEATURES: list[str] = [
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
]

PD_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LGD_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
