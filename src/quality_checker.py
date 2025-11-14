import pandas as pd
import numpy as np

expected_schema = {
    "client_id": "int",
    "full_name": "string",
    "email": "string",
    "age": "int",
    "country": "string",
    "signup_date": "date",
    "last_purchase_date": "date",
    "total_spent": "float",
    "is_active": "bool",
    "segment": "string",
    "phone_number": "string"
}

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def check_missing_values(df):

    missing_count = df.isnull().sum()
    
    missing_percentage = (missing_count / len(df)) * 100

    missing_summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    return missing_summary

def check_duplicates(df, subset_columns=None):

    # Case 1: Duplicates considereing the entire row
    total_duplicates = df.duplicated().sum()

    # Case 2; Duplicates based on specific columns
    if subset_columns is not None:
        subset_duplicates = df.duplicated(subset=subset_columns).sum()
    else:
        subset_duplicates = None
    
    # Extract actual duplicated rows
    duplicated_rows = df[df.duplicated(keep=False)]

    # Build result dictionary
    result = {
        "total_duplicates": total_duplicates,
        "subset_duplicates": subset_duplicates,
        "duplicated_rows_sample": duplicated_rows.head()
    }

    return result

def check_schema(df, expected_schema):

    report_rows = []

    # Iterate over each column in the expected schema
    for column, expected_type in expected_schema.items():

        # Detect actual pandas dtype
        actual_dtype = df[column].dtype

        # Analyze invalid values for each expected type
        invalid_values = []

        for value in df[column].sample(min(20, len(df))):
            if expected_type == "int":
                try:
                    int(value)
                except:
                    invalid_values.append(value)

            elif expected_type == "float":
                try:
                    float(value)
                except:
                    invalid_values.append(value)
            
            elif expected_type == "bool":
                if value not in [True, False, "true", "false", "TRUE", "FALSE", 1, 0]:
                    invalid_values.append(value)
            
            elif expected_type == "date":
                try:
                    pd.to_datetime(value)
                except:
                    invalid_values.append(value)
            
            else:
                if not isinstance(value, str):
                    invalid_values.append(value)
        
        report_rows.append({
            "column": column,
            "expected_type": expected_type,
            "actual_dtype": str(actual_dtype),
            "invalid_sample_values": invalid_values[:5]  # Limit to first 5 invalid values
        })

    return pd.DataFrame(report_rows)

def clean_whitespace(df):

    cleaned_df = df.copy()
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].str.strip().astype(str).str.strip()
    return cleaned_df

def normalize_case(df, columns):

    cleaned_df = df.copy()
    for col in columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.capitalize()
    return cleaned_df

def normalize_countries(df):

    cleaned_df = df.copy()
    replacements = {
        "panama": "Panama",
        "panamá": "Panama",
        "panamá ": "Panama",
        "panama ": "Panama",
        "usa": "United States",
        "us": "United States",
        "canada": "Canada"
    }

    cleaned_df["country"] = (
        cleaned_df["country"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace(replacements)
        .str.capitalize()
    )
    return cleaned_df

def fix_booleans(df):

    cleaned_df = df.copy()

    true_vals = ["true", "True", "TRUE", "1", 1]
    false_vals = ["false", "False", "FALSE", "0", 0]

    def convert_to_bool(val):
        if val in true_vals:
            return True
        elif val in false_vals:
            return False
        else:
            return np.nan
    cleaned_df["is_active"] = cleaned_df["is_active"].apply(convert_to_bool)
    return cleaned_df

def fix_unknowns(df):

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.replace("unknown", np.nan)
    return cleaned_df

def normalize_segment(df):

    cleaned_df = df.copy()

    cleaned_df["segment"] = (
        cleaned_df["segment"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
        .str.capitalize()
    )

    segment_map = {
        "Vip": "VIP",
        "V i p": "VIP",
        "V i.p": "VIP",
        "Basic": "Basic",
        "Premium": "Premium",
        "None": np.nan,
        "Nan": np.nan
    }

    cleaned_df["segment"] = cleaned_df["segment"].replace(segment_map)

    return cleaned_df

def format_cleaner(df):

    original_df = df.copy()

    # Step 1 - whitespace
    df= clean_whitespace(df)

    # Step 2 - normalize alphabetic case
    df = normalize_case(df, columns=["country", "segment"])

    # Step 3 - fix countries
    df = normalize_countries(df)

    # Step 4 - fix booleans incosistencies
    df = fix_booleans(df)

    # Step 5 - replace 'unknown' with NaN
    df = fix_unknowns(df)

    # Step 6 - Standardize segment categories
    df = normalize_segment(df)

    # Compare original vs cleaned data
    changes = (original_df != df).sum()

    report = pd.DataFrame({
        "column": changes.index,
        "values_changed": changes.values
    })

    return df, report

def get_numeric_columns(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return numeric_cols

def detect_outliers_iqr(series):
   
    series = pd.Series(series).astype(float)

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (series < lower_bound) | (series > upper_bound)

    return pd.Series(mask).reset_index(drop=True)


def detect_outliers_zscore(series):
 
    series = pd.Series(series).astype(float)

    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series([False] * len(series))

    z_scores = (series - mean) / std
    mask = (z_scores > 3) | (z_scores < -3)

    return pd.Series(mask).reset_index(drop=True)

def outlier_detector(df):

    numeric_cols = get_numeric_columns(df)
    report_rows = []

    for col in numeric_cols:

        if col == "client_id":
            continue
        series = df[col].dropna().reset_index(drop=True)

    iqr_mask = detect_outliers_iqr(series)
    z_mask = detect_outliers_zscore(series)

    iqr_examples = series[iqr_mask.values].head(5).tolist()
    z_examples = series[z_mask.values].head(5).tolist()


    report_rows.append({
        "column": col,
        "outliers_iqr": int(iqr_mask.sum()),
        "outliers_zscore": int(z_mask.sum()),
        "iqr_percentage": round((iqr_mask.sum() / len(series)) * 100, 2),
        "zscore_percentage": round((z_mask.sum() / len(series)) * 100, 2),
        "iqr_example": iqr_examples,
        "zscore_example": z_examples
    })

    return pd.DataFrame(report_rows)

def validate_age(df):

    mask =(df["age"] < 18) | (df["age"] > 100)
    return df.loc[mask, ["client_id", "full_name", "age"]]


def validate_total_spent(df):

    values = pd.to_numeric(df["total_spent"], errors="coerce")
    mask = (values < 0)
    return df.loc[mask, ["client_id", "full_name", "total_spent"]]


def validate_date_order(df):

    df_dates = df.copy()
    df_dates["signup_date"] = pd.to_datetime(df_dates["signup_date"], errors="coerce")
    df_dates["last_purchase_date"] = pd.to_datetime(df_dates["last_purchase_date"], errors="coerce")

    mask = df_dates["last_purchase_date"] < df_dates["signup_date"]
    return df.loc[mask, ["client_id", "signup_date", "last_purchase_date"]]


def validate_email(df):

    mask = ~df["email"].str.contains("@") | ~df["email"].str.contains(".", regex=False)
    return df.loc[mask, ["client_id", "email"]]


def validate_segment(df):

    allowed = ["Basic", "Premium", "VIP"]
    mask = ~df["segment"].isin(allowed)
    return df.loc[mask, ["client_id", "segment"]]


def validate_country(df):
    allowed = ["Panama", "United States", "Canada"]
    mask = ~df["country"].isin(allowed)
    return df.loc[mask, ["client_id", "country"]]


def validate_client_id(df):
    duplicates = df[df["client_id"].duplicated(keep=False)]
    return duplicates[["client_id", "full_name"]]


def validate_phone(df):
    mask = ~df["phone_number"].str.contains(r"\d", regex=True)
    return df.loc[mask, ["client_id", "phone_number"]]

def validate_business_rules(df):

    checks = {
        "age_range": validate_age(df),
        "total_spent_non_negative": validate_total_spent(df),
        "date_order": validate_date_order(df),
        "invalid_email": validate_email(df),
        "invalid_segment": validate_segment(df),
        "invalid_country": validate_country(df),
        "duplicate_client_id": validate_client_id(df),
        "invalid_phone": validate_phone(df)
    }

    report = []

    for check_name, result in checks.items():
        report.append({
            "rule": check_name,
            "violations": len(result),
            "examples": result.head(3).to_dict(orient="records")
        })

    return pd.DataFrame(report), checks

def calculate_data_quality_score(df, missing_report, duplicate_rows, schema_report,
                                 format_report, outlier_report, business_report):
    total_rows = len(df)
    total_cells = df.shape[0] * df.shape[1]


    total_missing = missing_report["missing_count"].sum()
    missing_pct = (total_missing / total_cells) * 100
    missing_score = max(0, 100 - missing_pct)


    duplicate_pct = (duplicate_rows / total_rows) * 100
    duplicate_score = max(0, 100 - duplicate_pct)


    schema_issues = sum(len(x) for x in schema_report["invalid_sample_values"])
    schema_pct = (schema_issues / total_rows) * 100
    schema_score = max(0, 100 - schema_pct)


    format_issues = format_report["values_changed"].sum()
    format_pct = (format_issues / total_cells) * 100
    format_score = max(0, 100 - format_pct)


    outlier_total = outlier_report["outliers_iqr"].sum() + outlier_report["outliers_zscore"].sum()
    outlier_pct = (outlier_total / total_rows) * 100
    outlier_score = max(0, 100 - outlier_pct)


    business_violations = business_report["violations"].sum()
    business_pct = (business_violations / total_rows) * 100
    business_score = max(0, 100 - business_pct)


    final_score = (
        missing_score * 0.20 +
        duplicate_score * 0.20 +
        business_score * 0.20 +
        schema_score * 0.15 +
        format_score * 0.15 +
        outlier_score * 0.10
    )

    scores = pd.DataFrame({
        "metric": [
            "missing_score",
            "duplicate_score",
            "schema_score",
            "format_score",
            "outlier_score",
            "business_rules_score",
            "FINAL_DATA_QUALITY_SCORE"
        ],
        "score": [
            missing_score,
            duplicate_score,
            schema_score,
            format_score,
            outlier_score,
            business_score,
            final_score
        ]
    })

    return scores, final_score

def main():

    file_path = "data/clients.csv"

    df = load_data(file_path)

    print("Data loaded successfully.\n")

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nMissing values summary:")
    missing_summary = check_missing_values(df)
    print(missing_summary)

    print("\nDuplicate rows check:")
    duplicates = check_duplicates(df)
    print(duplicates)

    duplicate_rows_result = duplicates["total_duplicates"]

    print("\nDuplicate emails check:")
    print(check_duplicates(df, subset_columns=["email"]))

    print("\nDuplicate client_id check:")
    print(check_duplicates(df, subset_columns=["client_id"]))

    print("\nSchema & Data Type Validation:")
    schema_report = check_schema(df, expected_schema)
    print(schema_report)

    print("\nFormat cleaning:")
    cleaned_df, format_report = format_cleaner(df)
    print(format_report)

    print("\nOutlier detection:")
    outlier_report = outlier_detector(cleaned_df)
    print(outlier_report)

    print("\nBusiness Rules Validation:")
    rules_report, details = validate_business_rules(cleaned_df)
    print(rules_report)

    print("\nData Quality Score:")
    dq_scores, final_score = calculate_data_quality_score(
        cleaned_df,
        missing_summary,
        duplicate_rows_result,
        schema_report,
        format_report,
        outlier_report,
        rules_report
    )

    print(dq_scores)
    print(f"\nFINAL DATA QUALITY SCORE: {final_score:.2f}/100")

if __name__ == "__main__":
    main()