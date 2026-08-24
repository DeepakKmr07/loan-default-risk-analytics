"""Keep categorical feature encodings consistent across training and inference.

LightGBM encodes pandas 'category' columns as integer codes internally. If the training
DataFrame and the inference DataFrame don't share the exact same category vocabulary (in the
same order), the same code can mean two different categories in each — a silent correctness
bug. Deriving the vocabulary from the full population (not just a training slice) once and
reusing it everywhere avoids that.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_categorical_dtypes(
    con: duckdb.DuckDBPyConnection, parquet_path: Path, categorical_features: list[str]
) -> dict[str, pd.CategoricalDtype]:
    """Derive a fixed category vocabulary per column from the full population Parquet file."""
    dtypes: dict[str, pd.CategoricalDtype] = {}
    for col in categorical_features:
        values = (
            con.execute(
                f"""
                SELECT DISTINCT {col} FROM read_parquet('{parquet_path.as_posix()}')
                WHERE {col} IS NOT NULL ORDER BY {col}
                """
            )
            .df()[col]
            .tolist()
        )
        dtypes[col] = pd.CategoricalDtype(categories=values)
    return dtypes


def apply_categorical_dtypes(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """Cast each categorical column of `df` to its fixed, shared dtype in place."""
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    return df
