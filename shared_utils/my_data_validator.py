
import pandas as pd
import numpy as np
import re


# ==========================================
# 1. DATASET OVERVIEW
# ==========================================

def dataset_overview(df):
    """Return basic information about the dataset."""
    return pd.DataFrame({
        "metric": [
            "Total Rows",
            "Total Columns",
            "Duplicate Rows",
            "Missing Values",
            "Memory Usage (MB)"
        ],
        "value": [
            len(df),
            len(df.columns),
            int(df.duplicated().sum()),
            int(df.isna().sum().sum()),
            round(
                df.memory_usage(deep=True).sum() / 1024**2,
                2
            )
        ]
    })


# ==========================================
# 2. CHECK MISSING VALUES
# ==========================================

def check_missing_values(df):
    """Count missing values and percentages."""
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (
            df.isna().mean() * 100
        ).round(2)
    })


# ==========================================
# 3. CHECK DUPLICATES
# ==========================================

def check_duplicates(df, subset=None):
    """Return duplicate rows, keeping the first occurrence."""
    return df[df.duplicated(subset=subset, keep="first")]


# ==========================================
# 4. CHECK DATA TYPES
# ==========================================

def check_data_types(df):
    """Return each column's data type."""
    return pd.DataFrame({
        "column": df.columns,
        "data_type": [str(dtype) for dtype in df.dtypes]
    })


# ==========================================
# 5. CHECK UNIQUE VALUES
# ==========================================

def check_unique_values(df):
    """Count distinct non-null values in each column."""
    return df.nunique(dropna=True).to_frame("unique_count")


# ==========================================
# 6. SHOW UNIQUE VALUES IN A COLUMN
# ==========================================

def show_unique_values(df, column):
    """Return distinct values in a specified column."""
    return df[column].unique()


# ==========================================
# 7. CHECK VALUE FREQUENCIES
# ==========================================

def check_value_counts(df, column):
    """Count how often each value occurs."""
    return df[column].value_counts(dropna=False)


# ==========================================
# 8. CHECK EMPTY STRINGS
# ==========================================

def check_empty_strings(df):
    """Count empty or whitespace-only strings."""
    result = {}

    for col in df.columns:
        result[col] = int(
            df[col].map(
                lambda x: isinstance(x, str)
                and x.strip() == ""
            ).sum()
        )

    return pd.Series(result, name="empty_strings")


# ==========================================
# 9. CHECK LEADING/TRAILING SPACES
# ==========================================

def check_extra_spaces(df):
    """Count text values with leading/trailing spaces."""
    result = {}

    for col in df.columns:
        result[col] = int(
            df[col].map(
                lambda x: isinstance(x, str)
                and x != x.strip()
            ).sum()
        )

    return pd.Series(result, name="extra_spaces")


# ==========================================
# 10. CHECK HIDDEN CHARACTERS
# ==========================================

def check_hidden_characters(df):
    """Count text containing control or zero-width characters."""
    pattern = r"[\x00-\x1f\x7f\u200b-\u200d\ufeff]"

    result = {}

    for col in df.columns:
        result[col] = int(
            df[col].map(
                lambda x: bool(re.search(pattern, x))
                if isinstance(x, str) else False
            ).sum()
        )

    return pd.Series(result, name="hidden_characters")


# ==========================================
# 11. CHECK NEGATIVE VALUES
# ==========================================

def check_negative_values(df, column):
    """Return rows with negative numeric values."""
    values = pd.to_numeric(df[column], errors="coerce")
    return df[values < 0]


# ==========================================
# 12. CHECK ZERO VALUES
# ==========================================

def check_zero_values(df, column):
    """Return rows with zero numeric values."""
    values = pd.to_numeric(df[column], errors="coerce")
    return df[values == 0]


# ==========================================
# 13. CHECK OUTLIERS USING IQR
# ==========================================

def check_outliers_iqr(df, column):
    """Return rows outside the 1.5 IQR boundaries."""
    values = pd.to_numeric(df[column], errors="coerce")

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr):
        return df.iloc[0:0].copy()

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[(values < lower) | (values > upper)]


# ==========================================
# 14. CHECK OUTLIERS USING Z-SCORE
# ==========================================

def check_outliers_zscore(df, column, threshold=3):
    """Return rows with absolute z-score above threshold."""
    values = pd.to_numeric(df[column], errors="coerce")

    std = values.std()

    if pd.isna(std) or std == 0:
        return df.iloc[0:0].copy()

    z_scores = (values - values.mean()) / std

    return df[z_scores.abs() > threshold]


# ==========================================
# 15. CHECK VALUE RANGE
# ==========================================

def check_value_range(df, column, min_value=None, max_value=None):
    """Return numeric rows outside an allowed range."""
    values = pd.to_numeric(df[column], errors="coerce")
    invalid = pd.Series(False, index=df.index)

    if min_value is not None:
        invalid |= values < min_value

    if max_value is not None:
        invalid |= values > max_value

    return df[invalid]


# ==========================================
# 16. CHECK INVALID DATES
# ==========================================

def check_invalid_dates(df, column, date_format=None):
    """Return non-null values that cannot be parsed as dates."""
    converted = pd.to_datetime(
        df[column],
        format=date_format,
        errors="coerce"
    )

    invalid = df[column].notna() & converted.isna()

    return df[invalid]


# ==========================================
# 17. CHECK FUTURE DATES
# ==========================================

def check_future_dates(df, column):
    """Return dates later than the current date."""
    dates = pd.to_datetime(df[column], errors="coerce")
    today = pd.Timestamp.today().normalize()

    return df[dates > today]


# ==========================================
# 18. CHECK REQUIRED COLUMNS
# ==========================================

def check_required_columns(df, required_columns):
    """Return missing required column names."""
    return [
        col for col in required_columns
        if col not in df.columns
    ]


# ==========================================
# 19. CHECK CATEGORY VALUES
# ==========================================

def check_allowed_values(df, column, allowed_values):
    """Return non-null values outside an approved category list."""
    invalid = (
        df[column].notna()
        & ~df[column].isin(allowed_values)
    )

    return df[invalid]


# ==========================================
# 20. COMPLETE DATA QUALITY REPORT
# ==========================================

def data_quality_report(df):
    """Return an overall data-quality summary."""
    return {
        "overview": dataset_overview(df),
        "missing_values": check_missing_values(df),
        "data_types": check_data_types(df),
        "unique_values": check_unique_values(df),
        "empty_strings": check_empty_strings(df),
        "extra_spaces": check_extra_spaces(df),
        "hidden_characters": check_hidden_characters(df),
        "duplicate_rows": check_duplicates(df)
    }
