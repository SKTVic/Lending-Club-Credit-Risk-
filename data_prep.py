"""Load and clean the Kaggle Lending Club file, and split it into dev / test / holdout.

Sample: 36-month loans issued 2012-2014 that have finished (fully paid or charged off),
so every loan's outcome is known. Run time: ~20 seconds the first time, then cached.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "accepted_2007_to_2018Q4.csv"     # from Kaggle: wordsforthewise/lending-club
CACHE = ROOT / "data" / "work" / "loans_clean.pkl"
SEED = 2024
MIN_DEV_ROWS = 200      # rarer loan purposes are grouped into "other"

COLS = ["id", "loan_status", "issue_d", "last_pymnt_d", "term", "int_rate", "grade", "sub_grade",
        "loan_amnt", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
        "fico_range_low", "fico_range_high", "dti", "annual_inc", "verification_status",
        "emp_length", "home_ownership", "purpose", "revol_util", "inq_last_6mths",
        "delinq_2yrs", "earliest_cr_line", "addr_state"]
NUMERIC = ["int_rate", "loan_amnt", "funded_amnt", "total_rec_prncp", "recoveries",
           "collection_recovery_fee", "fico_range_low", "fico_range_high", "dti", "annual_inc",
           "revol_util", "inq_last_6mths", "delinq_2yrs"]


def read_raw():
    """Read only the rows and columns we need, as text."""
    if not RAW.exists():
        raise FileNotFoundError(f"Download accepted_2007_to_2018Q4.csv from Kaggle into {RAW.parent}")
    return duckdb.sql(f"""
        SELECT {', '.join(COLS)}
        FROM read_csv_auto('{RAW}', all_varchar=true)
        WHERE trim(term) = '36 months'
          AND right(issue_d, 4) IN ('2012', '2013', '2014')
          AND loan_status IN ('Fully Paid', 'Charged Off', 'Default')
    """).df()


def clean(raw):
    """Types, derived features, simple filters. Returns (clean data, log of rows removed)."""
    d = raw.copy()
    log = [("36-month loans issued 2012-2014, finished", len(d))]

    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c].str.replace("%", "", regex=False).str.strip(), errors="coerce")
    for c in ["issue_d", "last_pymnt_d", "earliest_cr_line"]:
        d[c] = pd.to_datetime(d[c], format="%b-%Y", errors="coerce")
    d["emp_length"] = pd.to_numeric(
        d["emp_length"].replace({"< 1 year": "0"}).str.extract(r"(\d+)", expand=False), errors="coerce")

    # Derived features
    d["default"] = d["loan_status"].isin(["Charged Off", "Default"]).astype(int)
    d["fico"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
    d["log_income"] = np.log(d["annual_inc"].where(d["annual_inc"] > 0))
    d["log_amount"] = np.log(d["loan_amnt"])
    d["history_years"] = (d["issue_d"] - d["earliest_cr_line"]).dt.days / 365.25
    d["emp_missing"] = d["emp_length"].isna().astype(int)      # blank job tenure is itself a signal
    d["emp_length"] = d["emp_length"].fillna(0)

    # Simple filters
    needed = ["fico", "dti", "log_income", "revol_util", "inq_last_6mths", "delinq_2yrs",
              "history_years", "funded_amnt", "total_rec_prncp"]
    before = len(d)
    d = d.dropna(subset=needed)
    log.append(("removed: a key field is blank", before - len(d)))
    before = len(d)
    d = d[d["dti"].between(0, 60) & d["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"])]
    log.append(("removed: dti outside 0-60 or rare home ownership type", before - len(d)))

    # Split: sort by id so the split is reproducible, then 60 / 20 / 20
    d = d.sort_values("id").reset_index(drop=True)
    u = np.random.default_rng(SEED).random(len(d))
    d["split"] = np.where(u < 0.6, "dev", np.where(u < 0.8, "test", "holdout"))

    # Rare purposes -> "other" (decided on dev only)
    counts = d.loc[d["split"] == "dev", "purpose"].value_counts()
    d["purpose"] = d["purpose"].where(d["purpose"].isin(counts[counts >= MIN_DEV_ROWS].index), "other")

    log.append(("loans used", len(d)))
    return d, pd.DataFrame(log, columns=["step", "loans"])


def load_loans():
    """Clean loans (cached after the first run) and the cleaning log."""
    if CACHE.exists():
        try:
            return pd.read_pickle(CACHE)
        except Exception:          # cache made by another pandas version: rebuild it
            pass
    out = clean(read_raw())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, CACHE)
    return out


if __name__ == "__main__":
    loans, log = load_loans()
    print(log.to_string(index=False))
    print(loans["split"].value_counts().to_dict(), f"default rate {loans['default'].mean():.1%}")
