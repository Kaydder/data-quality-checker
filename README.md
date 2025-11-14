📊 Data Quality Tool (Python)

A complete Python-based data quality evaluation tool designed to validate, clean, detect anomalies, and score the overall quality of any dataset.
This project replicates several real-world Data Quality & Governance practices used in Data Analytics, Data Engineering and Data Stewardship roles.

🚀 Features
✔ 1. Missing Values Analysis

Counts missing values per column

Calculates missing percentage

Generates a summary table

✔ 2. Duplicate Detection

Detects full-row duplicates

Detects duplicates by key columns (e.g., email, client_id)

Shows examples of duplicated records

✔ 3. Schema Validation

Checks whether each column matches the expected schema:

int

float

string

bool

date

Includes examples of invalid values found.

✔ 4. Format Cleaning & Normalization

Automatically fixes:

Inconsistent casing

Whitespace

Country name variations

Boolean values ("TRUE", "false", "0", etc.)

Replacement of "unknown" with NaN

Standardization of segment categories ("V.I.P", "vip", etc.)

✔ 5. Outlier Detection

Supports two statistical methods:

IQR (Interquartile Range)

Z-Score

Reports:

Outlier count

Percentage

Example outlier values

✔ 6. Business Rules Validation

Applies domain rules such as:

Age must be between 18 and 100

last_purchase_date must be after signup_date

Emails must contain "@" and "."

Allowed country list

Non-negative total_spent

Valid phone number structure

Detection of duplicate client_id

✔ 7. Data Quality Score

Weighted scoring system:

Metric	Weight
Missing values	20%
Duplicates	20%
Business rules	20%
Schema	15%
Format	15%
Outliers	10%

Produces:

Metrics table

Final score /100

📁 Project Structure
data-quality-tool/
│
├── data/
│   └── clients.csv
│
├── src/
│   └── quality_checker.py
│
├── README.md
└── requirements.txt


🧪 Dataset Included

The repository includes a synthetic dataset with realistic data quality issues:

Missing values

Invalid dates

Broken emails

Duplicated IDs

Negative or impossible ages

Inconsistent country formatting

Corrupted phone numbers

Unstandardized categories

This dataset fully tests every validation feature.

Author

Kayder Murillo

Contributions

Pull requests and suggestions are welcome!

 License

MIT License