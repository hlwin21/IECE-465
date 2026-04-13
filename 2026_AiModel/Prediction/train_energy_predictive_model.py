"""
train_energy_predictive_model.py  —  IMPROVED V4
=================================================
V3 -> V4 changes (residential R2 fix):

The residential model was stuck at R2 ~ 0.47 because the three largest
electricity drivers were missing or poorly represented:

  1. ELECTRIC HEAT INTERACTION  (biggest single fix)
     A home heated with electricity uses 3-4x the kWh of an identical
     gas-heated home.  Adding electric_heat_flag and -- critically -- the
     interaction electric_heat_flag x HDD65 gives the model the signal
     it needs to separate these two fundamentally different populations.

  2. EXPANDED RECS FEATURE SET  (+15 new columns)
       FUELH2O   -- water heater fuel  (electric WH alone adds ~4 500 kWh/yr)
       SWIMPOOL  -- pool pump          (~2 000-4 000 kWh/yr)
       EQUIPM    -- heating equipment  (heat pump vs furnace efficiency differs)
       COOLTYPE  -- AC type            (central vs window: 2x efficiency gap)
       DRYRFUEL  -- dryer fuel         (electric vs gas: ~800 kWh/yr difference)
       RANGEFUEL -- cooking range fuel
       NREFRIG   -- extra refrigerators are large baseline loads
       TOTROOMS / STORIES             (better size proxies alongside sqft)
       ADQINSUL / DRAFTY              (insulation quality drives HVAC load)
       MONEYPY                        (income proxies appliance count & quality)

  3. ELECTRIC-APPLIANCE STACK FEATURE
     total_electric_appliances combines electric WH + dryer + range into
     a single count capturing baseline electric load from fuel choice.

  4. RESIDENTIAL HYPERPARAMETERS
     max_iter -> 1 200, min_samples_leaf -> 15 to exploit the 18k-row
     RECS dataset more fully.

  5. QUANTILE CAPPING instead of hard IQR drop for residential
     Dropping IQR outliers removes real high-use homes (pools, large all-
     electric homes) that are important to model. Capping at the 99th
     percentile keeps those rows while bounding extreme leverage.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print(" ENERGY PREDICTIVE MODEL  --  TRAINING SCRIPT V4")
print("=" * 60)

# =========================================================
# PATHS
# =========================================================
BASE_DIR   = Path.home() / "Downloads" / "465data" / "training_data"
OUTPUT_DIR = Path.home() / "Downloads" / "465data" / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMERCIAL_FILE  = BASE_DIR / "cbecs2018_public.csv"
RESIDENTIAL_FILE = BASE_DIR / "recs2020_public_v7.csv"

# =========================================================
# TARGETS
# =========================================================
COMMERCIAL_TARGET  = "ELBTU"   # thousand BTU of electricity consumed
RESIDENTIAL_TARGET = "KWH"     # kWh of electricity consumed

# =========================================================
# FEATURE CANDIDATES
# Script silently drops any column absent in the CSV so new
# candidates are safe to add without breaking anything.
# =========================================================
COMMERCIAL_FEATURE_CANDIDATES = [
    # Building identity
    "PBA", "PBAPLUS",
    # Size / occupancy
    "SQFT", "NFLOOR", "NWKER", "WKHRS",
    # Vintage
    "YRCONC",
    # Climate
    "CDD65", "HDD65", "PUBCLIM",
    # Operations
    "OPEN24", "WKHRSC", "VACANT",
    # Equipment
    "RFGWI", "PCTERM", "PCTERMN", "LAPTOP", "SERVER",
]

RESIDENTIAL_FEATURE_CANDIDATES = [
    # -- Housing characteristics ------------------------------------------
    "TYPEHUQ",        # housing unit type (single-family, apt, etc.)
    "TOTCSQFT",       # conditioned floor area (sq ft)
    "TOTUCSQFT",      # unconditioned floor area
    "BEDROOMS",       # number of bedrooms
    "TOTROOMS",       # total number of rooms                   (NEW)
    "STORIES",        # number of stories                       (NEW)
    "YEARMADERANGE",  # construction vintage category
    "ADQINSUL",       # insulation adequacy (self-reported)     (NEW)
    "DRAFTY",         # home is drafty / leaky                  (NEW)
    # -- Household --------------------------------------------------------
    "NHSLDMEM",       # household members
    "MONEYPY",        # annual household income bracket         (NEW)
    # -- Space heating ----------------------------------------------------
    "FUELHEAT",       # primary heating fuel  <- most important feature
    "EQUIPM",         # heating equipment type (heat pump, furnace...) (NEW)
    "HEATHOME",       # home has heating equipment              (NEW)
    "UGWARM",         # census climate warm flag
    "TEMPHOME",       # winter thermostat set-point
    # -- Space cooling ----------------------------------------------------
    "AIRCOND",        # has air conditioning
    "COOLTYPE",       # AC type: central / window / evaporative (NEW)
    "NUMBERAC",       # number of AC units
    "TEMPHOMEAC",     # summer thermostat set-point
    # -- Water heating ----------------------------------------------------
    "FUELH2O",        # water heater fuel (NEW -- ~4 500 kWh/yr if electric)
    "WHEATSIZ",       # water heater tank size                  (NEW)
    # -- Pool / hot tub ---------------------------------------------------
    "SWIMPOOL",       # has swimming pool  (NEW -- ~3 000 kWh/yr)
    "FUELTUB",        # hot tub / spa fuel                      (NEW)
    # -- Appliances -------------------------------------------------------
    "NREFRIG",        # number of refrigerators                 (NEW)
    "RANGEFUEL",      # cooking range fuel (electric adds load) (NEW)
    "STOVEN",         # has stove/oven
    "DRYRFUEL",       # dryer fuel (electric vs gas)            (NEW)
    "DRYER",          # has dryer
    "WASHLOAD",       # washer loads per week
    "TVCOLOR",        # number of TVs
    "NCOMBATH",       # number of bathrooms
    # NOTE: DOEID intentionally omitted -- survey record ID, not a predictor
]

# =========================================================
# HELPERS
# =========================================================

def usable_columns(df, cols):
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [INFO] Columns not in CSV (skipped): {missing}")
    return present


def clean_numeric_target(df, target_col):
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df[df[target_col].notna() & (df[target_col] > 0)].copy()


def remove_outliers_iqr(df, target_col, multiplier=3.0):
    """Drop extreme outliers. Used for commercial dataset."""
    q1, q3 = df[target_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[
        (df[target_col] >= q1 - multiplier * iqr) &
        (df[target_col] <= q3 + multiplier * iqr)
    ].copy()


def cap_outliers_quantile(df, target_col, upper_pct=0.99):
    """
    Winsorise target at the given percentile instead of dropping rows.
    Used for residential: high-use homes (pools, all-electric) are real
    data points the model should learn from, not discarded.
    """
    cap = df[target_col].quantile(upper_pct)
    df  = df.copy()
    df[target_col] = df[target_col].clip(upper=cap)
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_engineered_features(df, mode):
    df = df.copy()

    # ------------------------------------------------------------------ #
    #  Commercial                                                          #
    # ------------------------------------------------------------------ #
    if mode == "commercial":
        sqft  = df.get("SQFT",   pd.Series(np.nan, index=df.index))
        nwker = df.get("NWKER",  pd.Series(np.nan, index=df.index))
        wkhrs = df.get("WKHRS",  pd.Series(np.nan, index=df.index))
        nflr  = df.get("NFLOOR", pd.Series(np.nan, index=df.index))
        cdd   = df.get("CDD65",  pd.Series(np.nan, index=df.index))
        hdd   = df.get("HDD65",  pd.Series(np.nan, index=df.index))

        df["log_sqft"]        = np.log1p(sqft)
        df["climate_load"]    = cdd + hdd
        df["sqft_x_climate"]  = sqft * (cdd + hdd) / 1_000
        df["sqft_per_worker"] = sqft  / nwker.replace(0, np.nan)
        df["worker_density"]  = nwker / sqft.replace(0, np.nan) * 1_000
        df["worker_hours"]    = wkhrs * nwker
        df["sqft_per_floor"]  = sqft  / nflr.replace(0, np.nan)

    # ------------------------------------------------------------------ #
    #  Residential                                                         #
    # ------------------------------------------------------------------ #
    if mode == "residential":
        sqft     = df.get("TOTCSQFT", pd.Series(np.nan, index=df.index))
        mem      = df.get("NHSLDMEM", pd.Series(np.nan, index=df.index))
        bed      = df.get("BEDROOMS", pd.Series(np.nan, index=df.index))
        ugw      = df.get("UGWARM",   pd.Series(0,      index=df.index))
        aircond  = df.get("AIRCOND",  pd.Series(0,      index=df.index))
        fuelheat = df.get("FUELHEAT", pd.Series(np.nan, index=df.index))
        fuelh2o  = df.get("FUELH2O",  pd.Series(np.nan, index=df.index))
        dryrfuel = df.get("DRYRFUEL", pd.Series(np.nan, index=df.index))
        rgfuel   = df.get("RANGEFUEL",pd.Series(np.nan, index=df.index))
        pool     = df.get("SWIMPOOL", pd.Series(0,      index=df.index))
        hdd_col  = df.get("HDD65",    pd.Series(0,      index=df.index))
        cdd_col  = df.get("CDD65",    pd.Series(0,      index=df.index))

        # Basic size / density
        df["log_sqft"]            = np.log1p(sqft)
        df["sqft_per_person"]     = sqft / mem.replace(0, np.nan)
        df["people_per_bedroom"]  = mem  / bed.replace(0, np.nan)
        df["sqft_per_bedroom"]    = sqft / bed.replace(0, np.nan)
        df["bedrooms_x_sqft"]     = bed  * sqft

        # ---- ELECTRIC HEATING (most important residential interaction) ----
        # FUELHEAT == 1 means electricity is the primary heating fuel.
        # Homes using electric heat have fundamentally different kWh profiles.
        df["electric_heat_flag"]   = (fuelheat == 1).astype(float)

        # The key interaction: electric heat magnitude scales with HDD.
        # A Minnesota all-electric home vs a Florida all-electric home differ
        # by thousands of kWh.  This single feature explains much of the
        # previously missing variance.
        df["electric_heat_x_hdd"]  = df["electric_heat_flag"] * hdd_col

        # ---- COOLING load ------------------------------------------------
        df["ac_x_cdd"]             = aircond * cdd_col
        df["cool_load_proxy"]      = sqft * aircond * (1 + ugw)
        df["sqft_x_cdd"]           = sqft * cdd_col / 1_000

        # ---- ELECTRIC WATER HEATER ---------------------------------------
        # FUELH2O == 5 means electricity in RECS 2020.
        # An electric water heater adds ~4 000-5 000 kWh/yr by itself.
        df["electric_wh_flag"]     = (fuelh2o == 5).astype(float)

        # ---- OTHER ELECTRIC APPLIANCES -----------------------------------
        # DRYRFUEL 1 = electricity;  RANGEFUEL 5 = electricity (RECS 2020)
        df["electric_dryer_flag"]  = (dryrfuel == 1).astype(float)
        df["electric_range_flag"]  = (rgfuel   == 5).astype(float)

        # Count of major electric fuel choices in the home.
        # Each additional electric appliance type adds several hundred kWh/yr.
        df["total_electric_appliances"] = (
            df["electric_heat_flag"]  +
            df["electric_wh_flag"]    +
            df["electric_dryer_flag"] +
            df["electric_range_flag"]
        )

        # ---- POOL --------------------------------------------------------
        df["has_pool"] = (pool == 1).astype(float)

    return df


# =========================================================
# MODEL TRAINING
# =========================================================

def build_model(df, feature_cols, target_col, model_name,
                categorical_cols=None, residential=False):
    """
    Train a HistGradientBoostingRegressor pipeline.
    TransformedTargetRegressor keeps log1p/expm1 internal so
    pipeline.predict() always returns raw kWh or BTU.
    """
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    # Residential uses more iterations + finer leaves to exploit 18k rows
    if residential:
        hgbr = HistGradientBoostingRegressor(
            max_iter=1_200,
            learning_rate=0.03,
            max_depth=7,
            min_samples_leaf=15,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=40,
            random_state=42,
            scoring="neg_mean_squared_error",
        )
    else:
        hgbr = HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.04,
            max_depth=6,
            min_samples_leaf=25,
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
            scoring="neg_mean_squared_error",
        )

    regressor = TransformedTargetRegressor(
        regressor=hgbr,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model",         regressor),
    ])

    pipeline.fit(X_train, y_train)

    pred_raw = np.maximum(pipeline.predict(X_test), 0)
    mae  = mean_absolute_error(y_test, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test, pred_raw))
    r2   = r2_score(y_test, pred_raw)

    residuals = y_test.values - pred_raw
    low_q  = float(np.quantile(residuals, 0.10))
    high_q = float(np.quantile(residuals, 0.90))

    return {
        "name":             model_name,
        "pipeline":         pipeline,
        "feature_cols":     feature_cols,
        "target_col":       target_col,
        "mae":              float(mae),
        "rmse":             float(rmse),
        "r2":               float(r2),
        "residual_low_q":   low_q,
        "residual_high_q":  high_q,
        "log_target":       False,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    if not COMMERCIAL_FILE.exists():
        raise FileNotFoundError(f"Missing: {COMMERCIAL_FILE}")
    if not RESIDENTIAL_FILE.exists():
        raise FileNotFoundError(f"Missing: {RESIDENTIAL_FILE}")

    print("\nLoading datasets ...")
    commercial_df  = pd.read_csv(COMMERCIAL_FILE,  low_memory=False)
    residential_df = pd.read_csv(RESIDENTIAL_FILE, low_memory=False)
    print(f"  Commercial  raw shape : {commercial_df.shape}")
    print(f"  Residential raw shape : {residential_df.shape}")

    print("\nValidating feature candidates ...")
    comm_feats = usable_columns(commercial_df,  COMMERCIAL_FEATURE_CANDIDATES)
    resi_feats = usable_columns(residential_df, RESIDENTIAL_FEATURE_CANDIDATES)

    if COMMERCIAL_TARGET not in commercial_df.columns:
        raise ValueError(f"Commercial target not found: {COMMERCIAL_TARGET}")
    if RESIDENTIAL_TARGET not in residential_df.columns:
        raise ValueError(f"Residential target not found: {RESIDENTIAL_TARGET}")

    # Clean targets
    commercial_df  = clean_numeric_target(commercial_df,  COMMERCIAL_TARGET)
    residential_df = clean_numeric_target(residential_df, RESIDENTIAL_TARGET)

    # Outlier strategy:
    #   Commercial  -> hard IQR drop  (6 000 rows, clean survey)
    #   Residential -> quantile cap   (keep high-use homes; they're real)
    commercial_df  = remove_outliers_iqr(commercial_df,  COMMERCIAL_TARGET, multiplier=3.0)
    residential_df = cap_outliers_quantile(residential_df, RESIDENTIAL_TARGET, upper_pct=0.99)

    print("\nEngineering features ...")
    commercial_df  = commercial_df[comm_feats  + [COMMERCIAL_TARGET]].copy()
    residential_df = residential_df[resi_feats + [RESIDENTIAL_TARGET]].copy()

    commercial_df  = add_engineered_features(commercial_df,  "commercial")
    residential_df = add_engineered_features(residential_df, "residential")

    comm_all_feats = [c for c in commercial_df.columns  if c != COMMERCIAL_TARGET]
    resi_all_feats = [c for c in residential_df.columns if c != RESIDENTIAL_TARGET]

    # Categorical columns -- integer codes representing discrete classes
    comm_cat_cols = [c for c in ["PBA", "PBAPLUS", "PUBCLIM", "YRCONC"]
                     if c in comm_all_feats]
    resi_cat_cols = [c for c in [
                         "TYPEHUQ", "FUELHEAT", "YEARMADERANGE",
                         "FUELH2O", "EQUIPM", "COOLTYPE",
                         "DRYRFUEL", "RANGEFUEL", "ADQINSUL",
                     ] if c in resi_all_feats]

    print("\nTraining commercial model ...")
    commercial_model = build_model(
        commercial_df, comm_all_feats, COMMERCIAL_TARGET,
        "Commercial HGBR V4",
        categorical_cols=comm_cat_cols,
        residential=False,
    )

    print("\nTraining residential model ...")
    residential_model = build_model(
        residential_df, resi_all_feats, RESIDENTIAL_TARGET,
        "Residential HGBR V4",
        categorical_cols=resi_cat_cols,
        residential=True,
    )

    print("\n" + "=" * 50)
    print(f"  {commercial_model['name']}")
    print(f"  Rows     : {len(commercial_df):,}")
    print(f"  Features : {len(comm_all_feats)}")
    print(f"  MAE      : {commercial_model['mae']:,.1f}")
    print(f"  RMSE     : {commercial_model['rmse']:,.1f}")
    print(f"  R2       : {commercial_model['r2']:.4f}   <- target >= 0.70")

    print()
    print(f"  {residential_model['name']}")
    print(f"  Rows     : {len(residential_df):,}")
    print(f"  Features : {len(resi_all_feats)}")
    print(f"  MAE      : {residential_model['mae']:,.1f}")
    print(f"  RMSE     : {residential_model['rmse']:,.1f}")
    print(f"  R2       : {residential_model['r2']:.4f}   <- target >= 0.70")
    print("=" * 50)

    # Save
    summary_df = pd.DataFrame([
        {
            "model":         commercial_model["name"],
            "rows_used":     len(commercial_df),
            "features_used": len(comm_all_feats),
            "target":        COMMERCIAL_TARGET,
            "mae":           commercial_model["mae"],
            "rmse":          commercial_model["rmse"],
            "r2":            commercial_model["r2"],
        },
        {
            "model":         residential_model["name"],
            "rows_used":     len(residential_df),
            "features_used": len(resi_all_feats),
            "target":        RESIDENTIAL_TARGET,
            "mae":           residential_model["mae"],
            "rmse":          residential_model["rmse"],
            "r2":            residential_model["r2"],
        },
    ])

    summary_path = OUTPUT_DIR / "model_summary_v4.csv"
    summary_df.to_csv(summary_path, index=False)

    joblib.dump(commercial_model,  OUTPUT_DIR / "commercial_nn_model.joblib")
    joblib.dump(residential_model, OUTPUT_DIR / "residential_nn_model.joblib")

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {OUTPUT_DIR / 'commercial_nn_model.joblib'}")
    print(f"Saved: {OUTPUT_DIR / 'residential_nn_model.joblib'}")


if __name__ == "__main__":
    main()