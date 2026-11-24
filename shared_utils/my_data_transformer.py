
"""
data_transformation.py

Reusable functions for transforming data with Pandas.

Cleaning and validation functions are maintained
in separate modules.
"""

import pandas as pd
import numpy as np


# ==================================================
# 1. RENAME COLUMNS
# ==================================================

def rename_columns(df, column_mapping):
    """Rename columns using a dictionary."""
    return df.rename(columns=column_mapping).copy()


# ==================================================
# 2. CONVERT DATA TYPE
# ==================================================

def convert_data_type(df, column, dtype):
    """Convert a column to a specified data type."""
    df = df.copy()
    df[column] = df[column].astype(dtype)
    return df


# ==================================================
# 3. CONVERT TO NUMERIC
# ==================================================

def convert_to_numeric(df, column):
    """Convert a column to numeric values."""
    df = df.copy()
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )
    return df


# ==================================================
# 4. CONVERT TO DATETIME
# ==================================================

def convert_to_datetime(df, column, date_format=None):
    """Convert a column to datetime."""
    df = df.copy()
    df[column] = pd.to_datetime(
        df[column],
        format=date_format,
        errors="coerce"
    )
    return df


# ==================================================
# 5. EXTRACT DATE COMPONENTS
# ==================================================

def extract_date_parts(df, date_column):
    """Extract year, month, day and quarter."""
    df = convert_to_datetime(df, date_column)

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["month_name"] = df[date_column].dt.month_name()
    df["day"] = df[date_column].dt.day
    df["quarter"] = df[date_column].dt.quarter

    return df


# ==================================================
# 6. CREATE MONTH-YEAR COLUMN
# ==================================================

def create_month_year(
    df, date_column, new_column="month_year"
):
    """Create a YYYY-MM column."""
    df = convert_to_datetime(df, date_column)

    df[new_column] = (
        df[date_column].dt.strftime("%Y-%m")
    )
    return df


# ==================================================
# 7. CREATE CALCULATED COLUMN
# ==================================================

def create_calculated_column(
    df, column1, column2,
    new_column, operation="multiply"
):
    """Perform arithmetic on two columns."""
    df = df.copy()

    a = pd.to_numeric(df[column1], errors="coerce")
    b = pd.to_numeric(df[column2], errors="coerce")

    operations = {
        "add": lambda: a + b,
        "subtract": lambda: a - b,
        "multiply": lambda: a * b,
        "divide": lambda: a.div(b.replace(0, np.nan))
    }

    if operation not in operations:
        raise ValueError(
            "Use add, subtract, multiply or divide"
        )

    df[new_column] = operations[operation]()
    return df


# ==================================================
# 8. CREATE CONDITIONAL COLUMN
# ==================================================

def create_conditional_column(
    df, column, threshold, new_column,
    true_value="High", false_value="Low"
):
    """Categorize values based on a threshold."""
    df = df.copy()

    values = pd.to_numeric(
        df[column], errors="coerce"
    )

    df[new_column] = np.where(
        values > threshold,
        true_value,
        false_value
    )

    return df


# ==================================================
# 9. MAP VALUES TO NEW COLUMN
# ==================================================

def map_column(df, source_column, mapping, new_column):
    """Map existing values to a new column."""
    df = df.copy()

    df[new_column] = df[source_column].map(mapping)
    return df


# ==================================================
# 10. FILTER ROWS
# ==================================================

def filter_rows(df, column, operator, value):
    """Filter rows using comparison operators."""
    operators = {
        "==": lambda s: s == value,
        "!=": lambda s: s != value,
        ">": lambda s: s > value,
        "<": lambda s: s < value,
        ">=": lambda s: s >= value,
        "<=": lambda s: s <= value
    }

    if operator not in operators:
        raise ValueError("Unsupported operator")

    return df.loc[
        operators[operator](df[column])
    ].copy()


# ==================================================
# 11. SORT DATA
# ==================================================

def sort_data(df, columns, ascending=True):
    """Sort by one or multiple columns."""
    return df.sort_values(
        by=columns,
        ascending=ascending
    ).reset_index(drop=True)


# ==================================================
# 12. GROUP AND AGGREGATE
# ==================================================

def group_and_aggregate(
    df, group_columns, aggregations
):
    """Group data and calculate summary statistics."""
    return (
        df.groupby(group_columns, dropna=False)
        .agg(aggregations)
        .reset_index()
    )


# ==================================================
# 13. CREATE PIVOT TABLE
# ==================================================

def create_pivot_table(
    df, index, columns, values, aggfunc="sum"
):
    """Create a pivot table."""
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0
    )


# ==================================================
# 14. MERGE DATAFRAMES
# ==================================================

def merge_dataframes(
    left_df, right_df, on,
    how="inner", validate=None
):
    """Merge two DataFrames."""
    return pd.merge(
        left_df,
        right_df,
        on=on,
        how=how,
        validate=validate
    )


# ==================================================
# 15. CONCATENATE DATAFRAMES
# ==================================================

def concatenate_dataframes(
    dataframes, ignore_index=True
):
    """Combine multiple DataFrames vertically."""
    return pd.concat(
        dataframes,
        ignore_index=ignore_index
    )


# ==================================================
# 16. DROP COLUMNS
# ==================================================

def drop_columns(df, columns):
    """Remove selected columns."""
    return df.drop(columns=columns).copy()


# ==================================================
# 17. SELECT COLUMNS
# ==================================================

def select_columns(df, columns):
    """Select specific columns."""
    return df.loc[:, columns].copy()


# ==================================================
# 18. REORDER COLUMNS
# ==================================================

def reorder_columns(df, first_columns):
    """Move selected columns to the beginning."""
    if isinstance(first_columns, str):
        first_columns = [first_columns]

    missing = [
        col for col in first_columns
        if col not in df.columns
    ]

    if missing:
        raise KeyError(f"Missing columns: {missing}")

    if len(first_columns) != len(set(first_columns)):
        raise ValueError("Duplicate columns specified")

    remaining = [
        col for col in df.columns
        if col not in first_columns
    ]

    return df.loc[
        :, first_columns + remaining
    ].copy()


# ==================================================
# 19. CREATE BINS
# ==================================================

def create_bins(
    df, column, bins, labels, new_column
):
    """Group numeric values into categories."""
    df = df.copy()

    df[new_column] = pd.cut(
        pd.to_numeric(df[column], errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


# ==================================================
# 20. PERCENTAGE OF TOTAL
# ==================================================

def percentage_of_total(
    df, column, new_column="percentage"
):
    """Calculate each value as a percentage of total."""
    df = df.copy()

    values = pd.to_numeric(
        df[column], errors="coerce"
    )

    total = values.sum()

    if pd.isna(total) or total == 0:
        df[new_column] = np.nan
    else:
        df[new_column] = (
            values / total * 100
        ).round(2)

    return df


# ==================================================
# 21. ADD ROW NUMBER
# ==================================================

def add_row_number(
    df, new_column="row_number", start=1
):
    """Add sequential row numbers."""
    df = df.copy()

    df[new_column] = np.arange(
        start, start + len(df)
    )

    return df


# ==================================================
# 22. APPLY CUSTOM TRANSFORMATION
# ==================================================

def apply_transformation(df, column, function):
    """Apply a custom function to a column."""
    df = df.copy()

    df[column] = df[column].apply(function)

    return df
