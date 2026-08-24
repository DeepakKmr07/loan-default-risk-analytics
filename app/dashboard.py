"""Interactive Streamlit dashboard for the credit risk intelligence platform.

Reads directly from the curated star schema Parquet files in data/curated/ — DuckDB queries
the Parquet files in place for every view, so there's no separate app-specific data pipeline.
Run via `streamlit run app/dashboard.py` (or `python run_pipeline.py --dashboard`).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
FACT_PATH = CURATED_DIR / "Fact_Loan_Risk_Portfolio.parquet"
STRESS_PATH = CURATED_DIR / "Fact_Stress_Test_Scenarios.parquet"
DIM_VINTAGE_PATH = CURATED_DIR / "Dim_Vintage.parquet"
DIM_GRADE_PATH = CURATED_DIR / "Dim_Credit_Grade.parquet"

SCATTER_SAMPLE_SIZE = 5000

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="\U0001F3E6",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _sql_list(values: list[str]) -> str:
    """Render a list of strings as a valid SQL `(...)` literal list for an IN clause."""
    escaped = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"({escaped})"


@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    """Open one DuckDB connection and register the star schema as SQL views over Parquet."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW dim_vintage AS SELECT * FROM read_parquet('{DIM_VINTAGE_PATH.as_posix()}')")
    con.execute(f"CREATE VIEW dim_grade AS SELECT * FROM read_parquet('{DIM_GRADE_PATH.as_posix()}')")
    con.execute(
        f"""
        CREATE VIEW loans AS
        SELECT f.*, v.vintage, v.issue_year, v.issue_quarter, g.grade, g.sub_grade
        FROM read_parquet('{FACT_PATH.as_posix()}') f
        JOIN dim_vintage v USING (vintage_key)
        JOIN dim_grade g USING (credit_grade_key)
        """
    )
    if STRESS_PATH.exists():
        con.execute(
            f"""
            CREATE VIEW stress_scenarios AS
            SELECT s.*, v.vintage, g.grade, g.sub_grade
            FROM read_parquet('{STRESS_PATH.as_posix()}') s
            JOIN dim_vintage v USING (vintage_key)
            JOIN dim_grade g USING (credit_grade_key)
            """
        )
    return con


@st.cache_data
def load_filter_options(_con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str]]:
    """Distinct sub-grade and vintage values for the sidebar filters (leading underscore on
    `_con` tells Streamlit's cache not to try hashing the DB connection object)."""
    sub_grades = _con.execute("SELECT DISTINCT sub_grade FROM loans ORDER BY sub_grade").df()["sub_grade"].tolist()
    vintages = _con.execute("SELECT DISTINCT vintage FROM loans ORDER BY vintage").df()["vintage"].tolist()
    return sub_grades, vintages


def build_where_clause(sub_grades: list[str], vintage_start: str, vintage_end: str) -> str:
    return f"sub_grade IN {_sql_list(sub_grades)} AND vintage BETWEEN '{vintage_start}' AND '{vintage_end}'"


# --- Bootstrap ---
if not FACT_PATH.exists():
    st.error(
        f"Curated star schema not found at `{FACT_PATH}`. "
        "Run `python run_pipeline.py` first to generate it."
    )
    st.stop()

con = get_connection()
all_sub_grades, all_vintages = load_filter_options(con)

st.title("\U0001F3E6 Credit Risk Intelligence Platform")
st.caption("LendingClub accepted-loans portfolio — PD × LGD × EAD Expected Loss engine")

st.sidebar.header("Filters")
selected_sub_grades = st.sidebar.multiselect(
    "Credit Sub-Grade", options=all_sub_grades, default=all_sub_grades
)
vintage_range = st.sidebar.select_slider(
    "Vintage Quarter Range", options=all_vintages, value=(all_vintages[0], all_vintages[-1])
)

if not selected_sub_grades:
    st.warning("Select at least one credit sub-grade in the sidebar to see results.")
    st.stop()

where_clause = build_where_clause(selected_sub_grades, vintage_range[0], vintage_range[1])

tab1, tab2, tab3 = st.tabs(
    ["\U0001F4CA Executive Portfolio Health", "\U0001F4C8 Vintage Analysis & Migration", "\U0001F32A️ Dynamic Macro Stress Simulator"]
)

# ============================== TAB 1: Executive Portfolio Health ==============================
with tab1:
    total_ead, weighted_pd, total_el = con.execute(
        f"""
        SELECT
            SUM(ead) AS total_ead,
            SUM(pd * ead) / NULLIF(SUM(ead), 0) AS weighted_pd,
            SUM(pd * lgd * ead) AS total_el
        FROM loans WHERE {where_clause}
        """
    ).fetchone()
    el_rate = (total_el / total_ead) if total_ead else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Exposure (EAD)", f"${total_ead:,.0f}")
    c2.metric("Weighted PD", f"{weighted_pd:.2%}")
    c3.metric("Aggregate Expected Loss", f"${total_el:,.0f}")
    c4.metric("Portfolio EL Rate", f"{el_rate:.2%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        grade_df = con.execute(
            f"""
            SELECT grade, SUM(pd * lgd * ead) AS expected_loss
            FROM loans WHERE {where_clause}
            GROUP BY grade ORDER BY grade
            """
        ).df()
        fig = px.bar(
            grade_df, x="grade", y="expected_loss",
            title="Expected Loss Distribution Across Credit Grades",
            labels={"grade": "Credit Grade", "expected_loss": "Expected Loss ($)"},
            color="grade", color_discrete_sequence=px.colors.sequential.YlOrRd,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        scatter_df = con.execute(
            f"""
            SELECT int_rate, pd, grade FROM (
                SELECT int_rate, pd, grade FROM loans WHERE {where_clause}
            ) filtered
            USING SAMPLE {SCATTER_SAMPLE_SIZE} (reservoir)
            """
        ).df()
        fig = px.scatter(
            scatter_df, x="int_rate", y="pd", color="grade",
            title=f"Interest Rate Yield vs Predicted PD ({len(scatter_df):,}-loan sample)",
            labels={"int_rate": "Interest Rate (%)", "pd": "Predicted PD"},
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================== TAB 2: Vintage Analysis & Migration ==============================
with tab2:
    st.subheader("Cumulative Default Rate by Origination Vintage")
    st.caption(
        "Realized default rate among resolved loans per origination quarter — a vintage-seasoning "
        "proxy, not a month-by-month decay curve (this dataset has no loan-level payment-history panel)."
    )
    vintage_df = con.execute(
        f"""
        SELECT
            vintage,
            SUM(CASE WHEN default_flag = 1 THEN 1 ELSE 0 END) AS n_default,
            SUM(CASE WHEN default_flag IS NOT NULL THEN 1 ELSE 0 END) AS n_resolved
        FROM loans WHERE {where_clause}
        GROUP BY vintage ORDER BY vintage
        """
    ).df()
    vintage_df["cumulative_default_rate"] = vintage_df["n_default"] / vintage_df["n_resolved"].replace(0, pd.NA)
    fig = px.line(
        vintage_df, x="vintage", y="cumulative_default_rate", markers=True,
        title="Cumulative Default Rate by Origination Vintage",
        labels={"vintage": "Origination Quarter", "cumulative_default_rate": "Cumulative Default Rate"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Loan Status Composition by Credit Grade")
    st.caption(
        "Current status mix per grade (%) — a composition snapshot, not a month-over-month "
        "delinquency transition matrix (no loan-performance panel exists in this dataset)."
    )
    status_df = con.execute(
        f"""
        SELECT grade, loan_status, COUNT(*) AS n
        FROM loans WHERE {where_clause}
        GROUP BY grade, loan_status
        """
    ).df()
    pivot = status_df.pivot(index="grade", columns="loan_status", values="n").fillna(0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig = px.imshow(
        pivot_pct, text_auto=".1f", aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="Loan Status", y="Credit Grade", color="% of Grade"),
        title="Loan Status Mix by Credit Grade (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================== TAB 3: Dynamic Macro Stress Simulator ==============================
with tab3:
    st.subheader("Dynamic Macro Stress Simulator")
    st.caption("Shocks apply to the currently filtered population (sidebar sub-grade/vintage selection).")

    s1, s2 = st.columns(2)
    pd_multiplier = s1.slider("PD Multiplier Shock", 1.0, 2.5, 1.0, 0.05)
    lgd_haircut_pct = s2.slider("Collateral Haircut / LGD Adjustment (%)", 0, 50, 0, 5)
    lgd_multiplier = 1.0 + lgd_haircut_pct / 100.0

    base_ead, base_wpd, base_el = con.execute(
        f"""
        SELECT SUM(ead) AS ead, SUM(pd * ead) / NULLIF(SUM(ead), 0) AS wpd, SUM(pd * lgd * ead) AS el
        FROM loans WHERE {where_clause}
        """
    ).fetchone()
    stress_ead, stress_wpd, stress_el = con.execute(
        f"""
        SELECT
            SUM(ead) AS ead,
            SUM(LEAST(pd * {pd_multiplier}, 1.0) * ead) / NULLIF(SUM(ead), 0) AS wpd,
            SUM(LEAST(pd * {pd_multiplier}, 1.0) * LEAST(lgd * {lgd_multiplier}, 1.0) * ead) AS el
        FROM loans WHERE {where_clause}
        """
    ).fetchone()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total EAD", f"${stress_ead:,.0f}", delta=f"${stress_ead - base_ead:,.0f}")
    c2.metric("Weighted PD", f"{stress_wpd:.2%}", delta=f"{(stress_wpd - base_wpd) * 100:+.2f} pp")
    c3.metric(
        "Expected Loss (Capital Impact)", f"${stress_el:,.0f}",
        delta=f"${stress_el - base_el:,.0f}", delta_color="inverse",
    )

    compare_df = pd.DataFrame(
        {"scenario": ["Before (Baseline)", "After (Custom Shock)"], "expected_loss": [base_el, stress_el]}
    )
    fig = px.bar(
        compare_df, x="scenario", y="expected_loss",
        title="Expected Loss: Before vs After Custom Shock",
        labels={"scenario": "", "expected_loss": "Expected Loss ($)"},
        color="scenario", color_discrete_sequence=["#4C78A8", "#D4AF37"],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Reference: Pre-Computed CCAR-Style Scenarios")
    if STRESS_PATH.exists():
        scenario_df = con.execute(
            f"""
            SELECT
                scenario,
                SUM(ead) AS total_ead,
                SUM(pd * ead) / NULLIF(SUM(ead), 0) AS weighted_pd,
                SUM(pd * lgd * ead) AS total_expected_loss
            FROM stress_scenarios WHERE {where_clause}
            GROUP BY scenario
            """
        ).df()
        scenario_order = {"Baseline": 0, "Adverse": 1, "Severely Adverse": 2}
        scenario_df["_order"] = scenario_df["scenario"].map(scenario_order)
        scenario_df = scenario_df.sort_values("_order").drop(columns="_order")
        scenario_df["weighted_pd"] = scenario_df["weighted_pd"].map(lambda v: f"{v:.2%}")
        scenario_df["total_ead"] = scenario_df["total_ead"].map(lambda v: f"${v:,.0f}")
        scenario_df["total_expected_loss"] = scenario_df["total_expected_loss"].map(lambda v: f"${v:,.0f}")
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No pre-computed scenarios found. Run `python run_pipeline.py --stress-test` "
            "to generate `Fact_Stress_Test_Scenarios.parquet` for a side-by-side reference."
        )
