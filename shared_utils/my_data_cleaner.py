
import pandas as pd
import numpy as np
import re


# ==========================================
# 1. CLEAN COLUMN NAMES
# ==========================================

def clean_column_names(df):
    """Remove spaces, lowercase, and standardize column names."""
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    ) 

    return df


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


# ==========================================
# 3. REMOVE HIDDEN CHARACTERS
# ==========================================

def remove_hidden_characters(df):
    """Remove extra spaces, tabs, and newline characters."""
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(
            lambda x: " ".join(x.split())
            if isinstance(x, str) else x
        )

    return df


# ==========================================
# 4. FILL MISSING VALUES
# ==========================================

def fill_nulls(df):
    """Fill numeric nulls with median and text nulls with Unknown."""
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():
                df[col] = df[col].fillna(df[col].median())

        elif pd.api.types.is_object_dtype(df[col]) or \
             pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("Unknown")

    return df


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

def check_nulls(df):
    """Return missing value counts and percentages."""
    result = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (
            df.isnull().mean() * 100
        ).round(2)
    })

    return result


# ==========================================
# 6. STANDARDIZE CITY NAMES
# ==========================================

def standardize_city(df, column="city"):
    """Standardize common city name variations."""
    df = df.copy()

    city_mapping = {
        "nyc": "New York",
        "new york": "New York",
        "la": "Los Angeles",
        "los angeles": "Los Angeles",
        "san fran": "San Francisco",
        "san francisco": "San Francisco"
    }

    values = df[column].astype("string").str.strip()
    normalized = values.str.lower()

    df[column] = normalized.map(city_mapping).fillna(values)

    return df


# ==========================================
# 7. STANDARDIZE PAYMENT METHODS
# ==========================================

def standardize_payment(df, column="payment_method"):
    """Standardize payment method variations."""
    df = df.copy()

    payment_mapping = {
        "cc": "Credit Card",
        "credit": "Credit Card",
        "credit card": "Credit Card",
        "debit": "Debit Card",
        "debit card": "Debit Card",
        "cash": "Cash",
        "paypal": "PayPal"
    }

    values = df[column].astype("string").str.strip()
    normalized = values.str.lower()

    df[column] = normalized.map(payment_mapping).fillna(values)

    return df


# ==========================================
# 8. CONVERT DATE COLUMNS
# ==========================================

def convert_to_datetime(df, column, date_format=None):
    """Convert a column to datetime."""
    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        format=date_format,
        errors="coerce"
    )

    return df


# ==========================================
# 9. CONVERT NUMERIC COLUMNS
# ==========================================

def convert_to_numeric(df, column):
    """Convert a column to numeric values."""
    df = df.copy()

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return df


# ==========================================
# 10. REMOVE EMPTY ROWS
# ==========================================

def remove_empty_rows(df):
    """Remove rows where all values are missing."""
    return df.dropna(how="all").reset_index(drop=True)


# ==========================================
# 11. REMOVE EMPTY COLUMNS
# ==========================================

def remove_empty_columns(df):
    """Remove columns where all values are missing."""
    return df.dropna(axis=1, how="all")


# ==========================================
# 12. FIND OUTLIERS USING IQR
# ==========================================

def find_outliers(df, column):
    """Return rows containing IQR outliers."""
    data = pd.to_numeric(df[column], errors="coerce")

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[(data < lower) | (data > upper)]


# ==========================================
# 13. DATASET SUMMARY
# ==========================================

def data_summary(df):
    """Display basic dataset information."""
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Duplicates:", df.duplicated().sum())
    print("Missing values:", df.isnull().sum().sum())

    return df.dtypes


# ==========================================
# 14. REMOVE LEADING/TRAILING SPACES
# ==========================================

def strip_all_strings(df):
    """Strip spaces from every text column."""
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    return df


# ==========================================
# 15. MASTER CLEANING FUNCTION
# ==========================================

def clean_data(df, fill_missing=False):
    """Apply common cleaning steps."""
    df = clean_column_names(df)
    df = remove_hidden_characters(df)
    df = remove_empty_rows(df)
    df = remove_empty_columns(df)
    df = remove_duplicates(df)

    if fill_missing:
        df = fill_nulls(df)

    return df   
print ("\nData cleaning module loaded successfully.")