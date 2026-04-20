import pandas as pd
import numpy as np

# ============================================================
# Project: Merge and clean raw healthcare data files
# Output : One patient-level analysis table
# Files  : patients.csv, diagnoses.csv, medications.csv,
#          lab_results.csv, outcomes.csv
# ============================================================

# -----------------------------
# 1. File paths
# -----------------------------
PATIENTS_FILE = "/mnt/data/patients.csv"
DIAGNOSES_FILE = "/mnt/data/diagnoses.csv"
MEDICATIONS_FILE = "/mnt/data/medications.csv"
LAB_RESULTS_FILE = "/mnt/data/lab_results.csv"
OUTCOMES_FILE = "/mnt/data/outcomes.csv"
OUTPUT_FILE = "/mnt/data/cleaned_patient_analysis_table.csv"


# -----------------------------
# 2. Helper functions
# -----------------------------
def clean_binary_columns(df, columns):
    """Ensure binary columns are 0/1 integers."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def safe_mode(series):
    """Return the mode if it exists; otherwise return NaN."""
    mode_vals = series.mode(dropna=True)
    return mode_vals.iloc[0] if not mode_vals.empty else np.nan


# -----------------------------
# 3. Load and clean patients table
# -----------------------------
patients = pd.read_csv(PATIENTS_FILE)
patients = patients.drop_duplicates(subset=["patient_id"]).copy()

# Standardize simple categorical text columns
categorical_cols = [
    "sex", "smoking_status", "alcohol_use", "exercise_level", "insurance_type"
]
for col in categorical_cols:
    patients[col] = patients[col].astype(str).str.strip().str.lower()

# Clean diagnosis indicator columns already present in patients.csv
patient_dx_cols = [c for c in patients.columns if c.startswith("dx_")]
patients = clean_binary_columns(patients, patient_dx_cols)

# Optional: basic sanity filtering for obviously invalid physiological values
# We keep rows but replace impossible values with NaN so they do not distort analysis.
patients.loc[~patients["age"].between(0, 120), "age"] = np.nan
patients.loc[~patients["bmi"].between(10, 100), "bmi"] = np.nan
patients.loc[~patients["systolic_bp"].between(50, 300), "systolic_bp"] = np.nan
patients.loc[~patients["diastolic_bp"].between(30, 200), "diastolic_bp"] = np.nan
patients.loc[~patients["heart_rate"].between(20, 250), "heart_rate"] = np.nan
patients.loc[~patients["temperature_f"].between(90, 110), "temperature_f"] = np.nan


# -----------------------------
# 4. Load and aggregate diagnoses
# -----------------------------
diagnoses = pd.read_csv(DIAGNOSES_FILE)
diagnoses = diagnoses.drop_duplicates().copy()
diagnoses["visit_date"] = pd.to_datetime(diagnoses["visit_date"], errors="coerce")

for col in ["visit_type", "primary_diagnosis", "primary_icd10", "provider_specialty"]:
    diagnoses[col] = diagnoses[col].astype(str).str.strip().str.lower()

# Count useful patient-level utilization features
visit_counts = diagnoses.groupby("patient_id").size().rename("n_diagnosis_visits")
unique_primary_dx = diagnoses.groupby("patient_id")["primary_diagnosis"].nunique().rename("n_unique_primary_diagnoses")
unique_icd10 = diagnoses.groupby("patient_id")["primary_icd10"].nunique().rename("n_unique_primary_icd10")
unique_specialty = diagnoses.groupby("patient_id")["provider_specialty"].nunique().rename("n_provider_specialties")

# Visit-type counts
visit_type_counts = (
    diagnoses.pivot_table(
        index="patient_id",
        columns="visit_type",
        values="primary_icd10",
        aggfunc="count",
        fill_value=0,
    )
    .add_prefix("visit_count_")
)

# Most recent diagnosis information
latest_dx = (
    diagnoses.sort_values(["patient_id", "visit_date"])
    .groupby("patient_id", as_index=False)
    .tail(1)
    [["patient_id", "visit_date", "primary_diagnosis", "primary_icd10", "provider_specialty", "visit_type"]]
    .rename(
        columns={
            "visit_date": "latest_dx_date",
            "primary_diagnosis": "latest_primary_diagnosis",
            "primary_icd10": "latest_primary_icd10",
            "provider_specialty": "latest_provider_specialty",
            "visit_type": "latest_visit_type",
        }
    )
)

# Combine diagnosis-level aggregates
agg_diagnoses = pd.concat(
    [visit_counts, unique_primary_dx, unique_icd10, unique_specialty], axis=1
).reset_index()
agg_diagnoses = agg_diagnoses.merge(visit_type_counts.reset_index(), on="patient_id", how="left")
agg_diagnoses = agg_diagnoses.merge(latest_dx, on="patient_id", how="left")


# -----------------------------
# 5. Load and aggregate medications
# -----------------------------
medications = pd.read_csv(MEDICATIONS_FILE)
medications = medications.drop_duplicates().copy()
medications["start_date"] = pd.to_datetime(medications["start_date"], errors="coerce")

for col in ["medication", "unit", "frequency", "indication"]:
    medications[col] = medications[col].astype(str).str.strip().str.lower()

medications["is_generic"] = medications["is_generic"].fillna(0).astype(int)
medications.loc[~medications["adherence_pct"].between(0, 100), "adherence_pct"] = np.nan
medications.loc[medications["duration_days"] < 0, "duration_days"] = np.nan
medications.loc[medications["dose"] < 0, "dose"] = np.nan

med_agg = medications.groupby("patient_id").agg(
    n_medication_records=("medication", "size"),
    n_unique_medications=("medication", "nunique"),
    n_unique_indications=("indication", "nunique"),
    avg_med_adherence_pct=("adherence_pct", "mean"),
    median_med_adherence_pct=("adherence_pct", "median"),
    avg_med_duration_days=("duration_days", "mean"),
    pct_generic_meds=("is_generic", "mean"),
    latest_med_start_date=("start_date", "max"),
).reset_index()

# Frequency counts are useful and compact
med_freq_counts = (
    medications.pivot_table(
        index="patient_id",
        columns="frequency",
        values="medication",
        aggfunc="count",
        fill_value=0,
    )
    .add_prefix("med_freq_count_")
    .reset_index()
)

# Most recent medication started for each patient
latest_med = (
    medications.sort_values(["patient_id", "start_date"])
    .groupby("patient_id", as_index=False)
    .tail(1)
    [["patient_id", "medication", "indication", "dose", "unit", "frequency", "start_date"]]
    .rename(
        columns={
            "medication": "latest_medication",
            "indication": "latest_medication_indication",
            "dose": "latest_medication_dose",
            "unit": "latest_medication_unit",
            "frequency": "latest_medication_frequency",
            "start_date": "latest_medication_start_date",
        }
    )
)

agg_medications = med_agg.merge(med_freq_counts, on="patient_id", how="left")
agg_medications = agg_medications.merge(latest_med, on="patient_id", how="left")


# -----------------------------
# 6. Load and aggregate lab results
# -----------------------------
lab_results = pd.read_csv(LAB_RESULTS_FILE)
lab_results = lab_results.drop_duplicates().copy()
lab_results["test_date"] = pd.to_datetime(lab_results["test_date"], errors="coerce")
lab_results["test_name"] = lab_results["test_name"].astype(str).str.strip().str.lower()
lab_results["flag"] = lab_results["flag"].astype(str).str.strip().str.lower()
lab_results["is_abnormal"] = lab_results["is_abnormal"].fillna(0).astype(int)

# Basic lab sanity checks: impossible negative values become NaN.
lab_results.loc[lab_results["value"] < 0, "value"] = np.nan

# Aggregate each patient x test_name into compact analysis features.
labs_by_test = (
    lab_results.groupby(["patient_id", "test_name"]).agg(
        n_results=("value", "size"),
        latest_test_date=("test_date", "max"),
        latest_value=("value", "last"),
        mean_value=("value", "mean"),
        median_value=("value", "median"),
        min_value=("value", "min"),
        max_value=("value", "max"),
        abnormal_rate=("is_abnormal", "mean"),
    )
    .reset_index()
)

# To guarantee latest_value truly corresponds to the latest date, recompute from sorted rows.
latest_lab_rows = (
    lab_results.sort_values(["patient_id", "test_name", "test_date"])
    .groupby(["patient_id", "test_name"], as_index=False)
    .tail(1)
    [["patient_id", "test_name", "test_date", "value", "flag", "is_abnormal"]]
    .rename(
        columns={
            "test_date": "true_latest_test_date",
            "value": "true_latest_value",
            "flag": "latest_flag",
            "is_abnormal": "latest_is_abnormal",
        }
    )
)

labs_by_test = labs_by_test.drop(columns=["latest_test_date", "latest_value"])
labs_by_test = labs_by_test.merge(latest_lab_rows, on=["patient_id", "test_name"], how="left")

# Pivot into a single wide patient-level table.
lab_features = labs_by_test.pivot(index="patient_id", columns="test_name")
lab_features.columns = [f"lab_{test}_{metric}" for metric, test in lab_features.columns]
lab_features = lab_features.reset_index()

# Overall lab utilization features
lab_overall = lab_results.groupby("patient_id").agg(
    n_lab_records=("test_name", "size"),
    n_unique_lab_tests=("test_name", "nunique"),
    overall_lab_abnormal_rate=("is_abnormal", "mean"),
    first_lab_date=("test_date", "min"),
    last_lab_date=("test_date", "max"),
).reset_index()

agg_labs = lab_overall.merge(lab_features, on="patient_id", how="left")


# -----------------------------
# 7. Load and aggregate outcomes
# -----------------------------
outcomes = pd.read_csv(OUTCOMES_FILE)
outcomes = outcomes.drop_duplicates().copy()
outcomes["admission_date"] = pd.to_datetime(outcomes["admission_date"], errors="coerce")
outcomes["discharge_date"] = pd.to_datetime(outcomes["discharge_date"], errors="coerce")
outcomes["discharge_disposition"] = outcomes["discharge_disposition"].astype(str).str.strip().str.lower()

binary_outcome_cols = ["icu_admission", "in_hospital_death", "readmitted_30d"]
outcomes = clean_binary_columns(outcomes, binary_outcome_cols)

outcome_agg = outcomes.groupby("patient_id").agg(
    n_hospitalizations=("admission_date", "size"),
    total_los_days=("length_of_stay_days", "sum"),
    avg_los_days=("length_of_stay_days", "mean"),
    max_los_days=("length_of_stay_days", "max"),
    any_icu_admission=("icu_admission", "max"),
    total_icu_days=("icu_days", "sum"),
    any_in_hospital_death=("in_hospital_death", "max"),
    any_readmitted_30d=("readmitted_30d", "max"),
    avg_days_to_readmission=("days_to_readmission", "mean"),
    total_charges_usd_sum=("total_charges_usd", "sum"),
    total_charges_usd_mean=("total_charges_usd", "mean"),
    latest_admission_date=("admission_date", "max"),
).reset_index()

latest_outcome = (
    outcomes.sort_values(["patient_id", "admission_date"])
    .groupby("patient_id", as_index=False)
    .tail(1)
    [["patient_id", "admission_date", "discharge_date", "discharge_disposition", "primary_drg"]]
    .rename(
        columns={
            "admission_date": "latest_hosp_admission_date",
            "discharge_date": "latest_hosp_discharge_date",
            "discharge_disposition": "latest_discharge_disposition",
            "primary_drg": "latest_primary_drg",
        }
    )
)

agg_outcomes = outcome_agg.merge(latest_outcome, on="patient_id", how="left")


# -----------------------------
# 8. Merge all cleaned patient-level tables
# -----------------------------
final_df = patients.copy()
final_df = final_df.merge(agg_diagnoses, on="patient_id", how="left")
final_df = final_df.merge(agg_medications, on="patient_id", how="left")
final_df = final_df.merge(agg_labs, on="patient_id", how="left")
final_df = final_df.merge(agg_outcomes, on="patient_id", how="left")

# Fill count-like fields with 0 where no record exists in downstream tables.
count_like_keywords = [
    "n_", "count_", "total_", "any_", "pct_generic_meds", "overall_lab_abnormal_rate"
]
count_like_cols = [
    c for c in final_df.columns
    if any(keyword in c for keyword in count_like_keywords)
]
for col in count_like_cols:
    if final_df[col].dtype.kind in "biufc":
        final_df[col] = final_df[col].fillna(0)

# Convert date columns to ISO strings for easier export/use in many tools.
date_cols = [c for c in final_df.columns if "date" in c]
for col in date_cols:
    final_df[col] = pd.to_datetime(final_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

# Remove columns that are mostly redundant or unusable for standard analysis.
# Here we drop raw free-text secondary diagnosis strings because they are sparse,
# very high-cardinality, and not patient-level after aggregation.
# (They were not merged into the final table in the first place.)
# No further drop is needed because all remaining columns are already aggregated
# or baseline patient-level features.

# Final deduplication and sort
final_df = final_df.drop_duplicates(subset=["patient_id"]).sort_values("patient_id").reset_index(drop=True)

# Save final table
final_df.to_csv(OUTPUT_FILE, index=False)

print("Final analysis table created successfully.")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Shape: {final_df.shape}")
