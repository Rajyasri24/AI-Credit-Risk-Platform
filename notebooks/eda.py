import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import ANALYTICAL_DATA


df = pd.read_parquet(ANALYTICAL_DATA)


print("\nDATASET SUMMARY")
print("=" * 60)

print("Applicants:", len(df))
print("Features:", len(df.columns))
print(
    "Duplicate applicants:",
    df["SK_ID_CURR"]
    .duplicated()
    .sum(),
)

print(
    "Observed default rate:",
    round(
        df["TARGET"].mean() * 100,
        2,
    ),
    "%",
)


numeric_cols = (
    df.select_dtypes(include="number")
    .columns
)

categorical_cols = (
    df.select_dtypes(exclude="number")
    .columns
)

print(
    "Numeric features:",
    len(numeric_cols),
)

print(
    "Categorical features:",
    len(categorical_cols),
)


print("\nDATA QUALITY")
print("=" * 60)

missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missing
    .head(15)
    .round(2)
)


print("\nINSIGHT 1 — DEFAULT RATE BY INCOME TYPE")
print("=" * 60)

income_analysis = (
    df.groupby(
        "NAME_INCOME_TYPE"
    )
    .agg(
        Applicants=("TARGET", "size"),
        Default_Rate=("TARGET", "mean"),
    )
    .sort_values(
        "Default_Rate",
        ascending=False,
    )
)

income_analysis["Default_Rate"] *= 100

print(
    income_analysis.round(2)
)


print("\nINSIGHT 2 — CREDIT BURDEN")
print("=" * 60)

print(
    df.groupby("TARGET")[
        "CREDIT_INCOME_RATIO"
    ]
    .median()
    .round(3)
)


print("\nINSIGHT 3 — EXTERNAL CREDIT INDICATORS")
print("=" * 60)

external_sources = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

print(
    df.groupby("TARGET")[
        external_sources
    ]
    .median()
    .round(3)
)


print("\nINSIGHT 4 — EMPLOYMENT")
print("=" * 60)

print(
    df.groupby("TARGET")[
        "EMPLOYMENT_YEARS"
    ]
    .median()
    .round(2)
)


print("\nINSIGHT 5 — REPAYMENT BEHAVIOUR")
print("=" * 60)

print(
    df.groupby("TARGET")[
        [
            "AVG_PAYMENT_DELAY",
            "LATE_PAYMENT_RATE",
        ]
    ]
    .median()
    .round(3)
)


# Chart 1
df["TARGET"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title(
    "Applicant Default Distribution"
)
plt.xlabel(
    "Default Status"
)
plt.ylabel(
    "Applicants"
)

plt.tight_layout()
plt.show()


# Chart 2
income_analysis[
    "Default_Rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Observed Default Rate by Income Type"
)
plt.xlabel(
    "Default Rate (%)"
)

plt.tight_layout()
plt.show()


# Chart 3
df.boxplot(
    column="CREDIT_INCOME_RATIO",
    by="TARGET",
    showfliers=False,
)

plt.title(
    "Credit Burden by Default Status"
)
plt.suptitle("")
plt.xlabel(
    "Default Status"
)
plt.ylabel(
    "Credit / Income"
)

plt.tight_layout()
plt.show()


# Chart 4
df.boxplot(
    column="EXT_SOURCE_2",
    by="TARGET",
    showfliers=False,
)

plt.title(
    "External Credit Indicator by Default Status"
)
plt.suptitle("")
plt.xlabel(
    "Default Status"
)
plt.ylabel(
    "EXT_SOURCE_2"
)

plt.tight_layout()
plt.show()


# Chart 5
late_payment = (
    df.groupby("TARGET")[
        "LATE_PAYMENT_RATE"
    ]
    .mean()
)

late_payment.plot(
    kind="bar"
)

plt.title(
    "Historical Late-Payment Rate by Default Status"
)
plt.xlabel(
    "Default Status"
)
plt.ylabel(
    "Average Late-Payment Rate"
)

plt.tight_layout()
plt.show()
