"""
run_energy_prediction.py  --  V4
==================================
Updated to collect the new residential inputs added in V4:
  - Water heater fuel (FUELH2O)
  - Swimming pool     (SWIMPOOL)
  - Dryer fuel        (DRYRFUEL)
  - Cooking range fuel (RANGEFUEL)

All new engineered features (electric_heat_flag, electric_heat_x_hdd,
electric_wh_flag, etc.) are computed here before calling predict(),
matching exactly what train_energy_predictive_model.py computed during
training.  pipeline.predict() returns raw-scale kWh / BTU directly
because TransformedTargetRegressor handles log/exp internally.
"""

import os
import numpy as np
import joblib
import pandas as pd

# =========================================================
# MODEL PATHS
# =========================================================
MODEL_FOLDER = os.path.join(
    os.path.expanduser("~"), "Downloads", "465data", "model_outputs"
)
COMMERCIAL_MODEL_FILE  = os.path.join(MODEL_FOLDER, "commercial_nn_model.joblib")
RESIDENTIAL_MODEL_FILE = os.path.join(MODEL_FOLDER, "residential_nn_model.joblib")

if not os.path.exists(COMMERCIAL_MODEL_FILE):
    raise FileNotFoundError(f"Commercial model not found: {COMMERCIAL_MODEL_FILE}")
if not os.path.exists(RESIDENTIAL_MODEL_FILE):
    raise FileNotFoundError(f"Residential model not found: {RESIDENTIAL_MODEL_FILE}")

commercial_model  = joblib.load(COMMERCIAL_MODEL_FILE)
residential_model = joblib.load(RESIDENTIAL_MODEL_FILE)

# =========================================================
# INPUT HELPERS
# =========================================================
def ask_text(prompt):
    return input(prompt).strip()

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Please enter a number.")

def ask_int(prompt):
    while True:
        try:
            return int(float(input(prompt).strip()))
        except ValueError:
            print("  Please enter a whole number.")

# =========================================================
# STATE / CLIMATE LOOKUP
# =========================================================
STATE_CLIMATE = {
    "alabama":        {"CDD65": 2200, "HDD65": 1800, "PUBCLIM": 4, "UGWARM": 1},
    "alaska":         {"CDD65": 50,   "HDD65": 10000,"PUBCLIM": 1, "UGWARM": 0},
    "arizona":        {"CDD65": 3500, "HDD65": 1200, "PUBCLIM": 5, "UGWARM": 1},
    "arkansas":       {"CDD65": 2200, "HDD65": 2800, "PUBCLIM": 4, "UGWARM": 1},
    "california":     {"CDD65": 900,  "HDD65": 2200, "PUBCLIM": 3, "UGWARM": 1},
    "colorado":       {"CDD65": 500,  "HDD65": 6000, "PUBCLIM": 2, "UGWARM": 0},
    "connecticut":    {"CDD65": 700,  "HDD65": 5600, "PUBCLIM": 2, "UGWARM": 0},
    "delaware":       {"CDD65": 1100, "HDD65": 4500, "PUBCLIM": 3, "UGWARM": 0},
    "florida":        {"CDD65": 4000, "HDD65": 500,  "PUBCLIM": 5, "UGWARM": 1},
    "georgia":        {"CDD65": 2500, "HDD65": 2500, "PUBCLIM": 4, "UGWARM": 1},
    "hawaii":         {"CDD65": 2500, "HDD65": 0,    "PUBCLIM": 5, "UGWARM": 1},
    "idaho":          {"CDD65": 400,  "HDD65": 7000, "PUBCLIM": 2, "UGWARM": 0},
    "illinois":       {"CDD65": 900,  "HDD65": 6200, "PUBCLIM": 2, "UGWARM": 0},
    "indiana":        {"CDD65": 1000, "HDD65": 5600, "PUBCLIM": 2, "UGWARM": 0},
    "iowa":           {"CDD65": 800,  "HDD65": 6800, "PUBCLIM": 2, "UGWARM": 0},
    "kansas":         {"CDD65": 1400, "HDD65": 5000, "PUBCLIM": 3, "UGWARM": 0},
    "kentucky":       {"CDD65": 1300, "HDD65": 4700, "PUBCLIM": 3, "UGWARM": 0},
    "louisiana":      {"CDD65": 3200, "HDD65": 1200, "PUBCLIM": 5, "UGWARM": 1},
    "maine":          {"CDD65": 500,  "HDD65": 7600, "PUBCLIM": 1, "UGWARM": 0},
    "maryland":       {"CDD65": 1100, "HDD65": 4300, "PUBCLIM": 3, "UGWARM": 0},
    "massachusetts":  {"CDD65": 650,  "HDD65": 6000, "PUBCLIM": 2, "UGWARM": 0},
    "michigan":       {"CDD65": 650,  "HDD65": 6700, "PUBCLIM": 2, "UGWARM": 0},
    "minnesota":      {"CDD65": 600,  "HDD65": 8000, "PUBCLIM": 1, "UGWARM": 0},
    "mississippi":    {"CDD65": 2800, "HDD65": 2000, "PUBCLIM": 4, "UGWARM": 1},
    "missouri":       {"CDD65": 1400, "HDD65": 4800, "PUBCLIM": 3, "UGWARM": 0},
    "montana":        {"CDD65": 250,  "HDD65": 8000, "PUBCLIM": 1, "UGWARM": 0},
    "nebraska":       {"CDD65": 900,  "HDD65": 6200, "PUBCLIM": 2, "UGWARM": 0},
    "nevada":         {"CDD65": 1800, "HDD65": 3200, "PUBCLIM": 4, "UGWARM": 1},
    "new hampshire":  {"CDD65": 500,  "HDD65": 7000, "PUBCLIM": 1, "UGWARM": 0},
    "new jersey":     {"CDD65": 900,  "HDD65": 5000, "PUBCLIM": 3, "UGWARM": 0},
    "new mexico":     {"CDD65": 1400, "HDD65": 4200, "PUBCLIM": 3, "UGWARM": 0},
    "new york":       {"CDD65": 800,  "HDD65": 5800, "PUBCLIM": 2, "UGWARM": 0},
    "north carolina": {"CDD65": 1800, "HDD65": 3200, "PUBCLIM": 4, "UGWARM": 1},
    "north dakota":   {"CDD65": 500,  "HDD65": 9000, "PUBCLIM": 1, "UGWARM": 0},
    "ohio":           {"CDD65": 900,  "HDD65": 5800, "PUBCLIM": 2, "UGWARM": 0},
    "oklahoma":       {"CDD65": 2000, "HDD65": 3800, "PUBCLIM": 4, "UGWARM": 1},
    "oregon":         {"CDD65": 300,  "HDD65": 5000, "PUBCLIM": 2, "UGWARM": 0},
    "pennsylvania":   {"CDD65": 800,  "HDD65": 5600, "PUBCLIM": 2, "UGWARM": 0},
    "rhode island":   {"CDD65": 650,  "HDD65": 5600, "PUBCLIM": 2, "UGWARM": 0},
    "south carolina": {"CDD65": 2400, "HDD65": 2400, "PUBCLIM": 4, "UGWARM": 1},
    "south dakota":   {"CDD65": 700,  "HDD65": 7600, "PUBCLIM": 1, "UGWARM": 0},
    "tennessee":      {"CDD65": 1600, "HDD65": 3600, "PUBCLIM": 3, "UGWARM": 0},
    "texas":          {"CDD65": 2800, "HDD65": 2200, "PUBCLIM": 4, "UGWARM": 1},
    "utah":           {"CDD65": 700,  "HDD65": 5600, "PUBCLIM": 2, "UGWARM": 0},
    "vermont":        {"CDD65": 400,  "HDD65": 7600, "PUBCLIM": 1, "UGWARM": 0},
    "virginia":       {"CDD65": 1300, "HDD65": 4200, "PUBCLIM": 3, "UGWARM": 0},
    "washington":     {"CDD65": 250,  "HDD65": 4800, "PUBCLIM": 2, "UGWARM": 0},
    "west virginia":  {"CDD65": 900,  "HDD65": 5200, "PUBCLIM": 2, "UGWARM": 0},
    "wisconsin":      {"CDD65": 600,  "HDD65": 7200, "PUBCLIM": 1, "UGWARM": 0},
    "wyoming":        {"CDD65": 300,  "HDD65": 7800, "PUBCLIM": 1, "UGWARM": 0},
    "dc":             {"CDD65": 1200, "HDD65": 4300, "PUBCLIM": 3, "UGWARM": 0},
    "washington dc":  {"CDD65": 1200, "HDD65": 4300, "PUBCLIM": 3, "UGWARM": 0},
}
DEFAULT_CLIMATE = {"CDD65": 1000, "HDD65": 5000, "PUBCLIM": 3, "UGWARM": 0}

def get_climate_from_state(state_name):
    return STATE_CLIMATE.get(state_name.lower().strip(), DEFAULT_CLIMATE)

# =========================================================
# FRIENDLY CODE MAPPINGS
# =========================================================
COMMERCIAL_TYPE_TO_PBA = {
    "office": 2, "retail": 6, "education": 14, "school": 14,
    "healthcare": 16, "hospital": 18, "warehouse": 25,
    "restaurant": 8, "food service": 8, "hotel": 15, "lodging": 15,
    "gym": 13, "fitness": 13, "religious": 23, "grocery": 7,
    "public assembly": 13, "service": 5, "other": 91,
}

def map_year_built_commercial(year):
    if year < 1946: return 1
    elif year < 1960: return 2
    elif year < 1970: return 3
    elif year < 1980: return 4
    elif year < 1990: return 5
    elif year < 2000: return 6
    elif year < 2013: return 7
    else: return 8

RESIDENTIAL_HOME_TYPE = {
    "single family": 2, "apartment": 5, "multi family": 4,
    "townhouse": 3, "mobile home": 6, "duplex": 4, "other": 9,
}
RESIDENTIAL_AIRCOND = {"yes": 1, "no": 0}
RESIDENTIAL_HEAT = {
    "electric": 1, "natural gas": 2, "gas": 2,
    "propane": 3, "oil": 4, "wood": 5, "other": 9,
}
# FUELH2O: 1=natural gas, 2=propane, 3=fuel oil, 5=electricity, 7=solar
WATER_HEATER_FUEL = {
    "electric": 5, "electricity": 5,
    "natural gas": 1, "gas": 1,
    "propane": 2,
    "oil": 3,
    "solar": 7,
    "other": 9,
}
# DRYRFUEL: 1=electricity, 2=natural gas, 3=propane, 4=other
DRYER_FUEL = {
    "electric": 1, "electricity": 1,
    "natural gas": 2, "gas": 2,
    "propane": 3, "other": 4,
}
# RANGEFUEL: 1=natural gas, 2=propane, 3=fuel oil, 5=electricity
RANGE_FUEL = {
    "electric": 5, "electricity": 5,
    "natural gas": 1, "gas": 1,
    "propane": 2, "other": 9,
}

def map_year_built_residential(year):
    if year < 1950: return 1
    elif year < 1970: return 2
    elif year < 1990: return 3
    elif year < 2000: return 4
    elif year < 2010: return 5
    elif year < 2020: return 6
    else: return 7

# =========================================================
# FEATURE ENGINEERING  (must match train_energy_predictive_model.py)
# =========================================================

def add_commercial_engineered(d):
    sqft   = d.get("SQFT",   np.nan)
    nwker  = d.get("NWKER",  np.nan)
    wkhrs  = d.get("WKHRS",  np.nan)
    nfloor = d.get("NFLOOR", np.nan)
    cdd    = d.get("CDD65",  np.nan)
    hdd    = d.get("HDD65",  np.nan)

    d["log_sqft"]        = np.log1p(sqft)
    d["climate_load"]    = cdd + hdd
    d["sqft_x_climate"]  = sqft * (cdd + hdd) / 1_000
    d["sqft_per_worker"] = sqft / nwker    if nwker else np.nan
    d["worker_density"]  = nwker / sqft * 1_000 if sqft else np.nan
    d["worker_hours"]    = wkhrs * nwker
    d["sqft_per_floor"]  = sqft / nfloor   if nfloor else np.nan
    return d


def add_residential_engineered(d):
    sqft     = d.get("TOTCSQFT",  np.nan)
    mem      = d.get("NHSLDMEM",  np.nan)
    bed      = d.get("BEDROOMS",  np.nan)
    ugw      = d.get("UGWARM",    0)
    aircond  = d.get("AIRCOND",   0)
    fuelheat = d.get("FUELHEAT",  np.nan)
    fuelh2o  = d.get("FUELH2O",   np.nan)
    dryrfuel = d.get("DRYRFUEL",  np.nan)
    rgfuel   = d.get("RANGEFUEL", np.nan)
    pool     = d.get("SWIMPOOL",  0)
    hdd      = d.get("HDD65",     0)
    cdd      = d.get("CDD65",     0)

    # Size / density
    d["log_sqft"]             = np.log1p(sqft)
    d["sqft_per_person"]      = sqft / mem  if mem  else np.nan
    d["people_per_bedroom"]   = mem  / bed  if bed  else np.nan
    d["sqft_per_bedroom"]     = sqft / bed  if bed  else np.nan
    d["bedrooms_x_sqft"]      = bed  * sqft if (bed and sqft) else np.nan

    # Electric heating (most important residential interaction)
    d["electric_heat_flag"]   = 1.0 if fuelheat == 1 else 0.0
    d["electric_heat_x_hdd"]  = d["electric_heat_flag"] * hdd

    # Cooling
    d["ac_x_cdd"]             = aircond * cdd
    d["cool_load_proxy"]      = sqft * aircond * (1 + ugw) if sqft else 0.0
    d["sqft_x_cdd"]           = sqft * cdd / 1_000 if sqft else 0.0

    # Electric water heater (FUELH2O == 5)
    d["electric_wh_flag"]     = 1.0 if fuelh2o == 5  else 0.0

    # Other electric appliances
    d["electric_dryer_flag"]  = 1.0 if dryrfuel == 1 else 0.0
    d["electric_range_flag"]  = 1.0 if rgfuel   == 5 else 0.0

    d["total_electric_appliances"] = (
        d["electric_heat_flag"]  +
        d["electric_wh_flag"]    +
        d["electric_dryer_flag"] +
        d["electric_range_flag"]
    )

    d["has_pool"] = 1.0 if pool == 1 else 0.0
    return d

# =========================================================
# PREDICTION
# =========================================================

def predict_with_range(model_bundle, user_input_dict):
    X_new    = pd.DataFrame([user_input_dict])
    expected = float(np.maximum(model_bundle["pipeline"].predict(X_new)[0], 0))
    low      = max(expected + model_bundle["residual_low_q"],  0)
    high     = max(expected + model_bundle["residual_high_q"], 0)
    return {
        "low_estimate":      round(low,      2),
        "expected_estimate": round(expected, 2),
        "high_estimate":     round(high,     2),
        "reasons": [
            "weather variation",
            "occupancy behavior",
            "operating schedule changes",
            "HVAC and equipment efficiency",
            "seasonal demand differences",
        ]
    }

# =========================================================
# FRIENDLY INPUT COLLECTORS
# =========================================================

def get_commercial_input_friendly():
    print("\nEnter commercial building information.\n")
    building_type = ask_text(
        "Building type (office, retail, gym, school, healthcare, warehouse, "
        "restaurant, hotel, grocery, other): "
    ).lower()
    state   = ask_text("State (e.g. New York, Texas): ").lower()
    sqft    = ask_float("Square footage: ")
    floors  = ask_float("Number of floors: ")
    workers = ask_float("Number of workers: ")
    hours   = ask_float("Weekly operating hours: ")
    year_b  = ask_int("Year built (e.g. 2016): ")

    climate = get_climate_from_state(state)
    pba     = float(COMMERCIAL_TYPE_TO_PBA.get(building_type, 91))
    yrconc  = float(map_year_built_commercial(year_b))

    raw = {
        "PBA":     pba,
        "PBAPLUS": pba,
        "SQFT":    float(sqft),
        "NFLOOR":  float(floors),
        "NWKER":   float(workers),
        "WKHRS":   float(hours),
        "YRCONC":  yrconc,
        "CDD65":   float(climate["CDD65"]),
        "HDD65":   float(climate["HDD65"]),
        "PUBCLIM": float(climate["PUBCLIM"]),
    }
    return add_commercial_engineered(raw)


def get_residential_input_friendly():
    print("\nEnter residential household information.\n")
    home_type = ask_text(
        "Home type (single family, apartment, townhouse, multi family, "
        "mobile home, other): "
    ).lower()
    state    = ask_text("State (e.g. New York, Texas): ").lower()
    sqft     = ask_float("Total square feet: ")
    members  = ask_float("Number of household members: ")
    bedrooms = ask_float("Number of bedrooms: ")
    year_b   = ask_int("Year built (e.g. 2005): ")

    aircond  = ask_text("Air conditioning? (yes / no): ").lower()
    heat_fuel= ask_text(
        "Main heating fuel (electric, natural gas, propane, oil, wood, other): "
    ).lower()
    wh_fuel  = ask_text(
        "Water heater fuel (electric, natural gas, propane, oil, solar, other): "
    ).lower()
    dryr_fuel= ask_text(
        "Dryer fuel (electric, natural gas, propane, other — or 'none' if no dryer): "
    ).lower()
    rng_fuel = ask_text(
        "Cooking range fuel (electric, natural gas, propane, other): "
    ).lower()
    has_pool = ask_text("Swimming pool? (yes / no): ").lower()

    climate  = get_climate_from_state(state)

    raw = {
        "TYPEHUQ":       float(RESIDENTIAL_HOME_TYPE.get(home_type, 9)),
        "TOTCSQFT":      float(sqft),
        "NHSLDMEM":      float(members),
        "BEDROOMS":      float(bedrooms),
        "YEARMADERANGE": float(map_year_built_residential(year_b)),
        "AIRCOND":       float(RESIDENTIAL_AIRCOND.get(aircond, 0)),
        "FUELHEAT":      float(RESIDENTIAL_HEAT.get(heat_fuel, 9)),
        "FUELH2O":       float(WATER_HEATER_FUEL.get(wh_fuel, 9)),
        "DRYRFUEL":      float(DRYER_FUEL.get(dryr_fuel, 4)),
        "RANGEFUEL":     float(RANGE_FUEL.get(rng_fuel, 9)),
        "SWIMPOOL":      1.0 if has_pool == "yes" else 0.0,
        "UGWARM":        float(climate["UGWARM"]),
        "CDD65":         float(climate["CDD65"]),
        "HDD65":         float(climate["HDD65"]),
    }
    return add_residential_engineered(raw)

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 50)
    print("  Electricity Usage Prediction Tool  (V4)")
    print("=" * 50)
    print("  1.  Commercial building")
    print("  2.  Residential household")

    choice = input("\nChoose 1 or 2: ").strip()

    if choice == "1":
        user_input = get_commercial_input_friendly()
        result     = predict_with_range(commercial_model, user_input)
        print("\n  Commercial electricity prediction (thousand BTU / year)")
        print(f"  Low estimate     : {result['low_estimate']:>12,.1f}")
        print(f"  Expected estimate: {result['expected_estimate']:>12,.1f}")
        print(f"  High estimate    : {result['high_estimate']:>12,.1f}")
        print(f"  Range driven by  : {', '.join(result['reasons'])}")

    elif choice == "2":
        user_input = get_residential_input_friendly()
        result     = predict_with_range(residential_model, user_input)
        print("\n  Residential electricity prediction (kWh / year)")
        print(f"  Low estimate     : {result['low_estimate']:>10,.1f}")
        print(f"  Expected estimate: {result['expected_estimate']:>10,.1f}")
        print(f"  High estimate    : {result['high_estimate']:>10,.1f}")
        print(f"  Range driven by  : {', '.join(result['reasons'])}")

    else:
        print("\n  Invalid choice -- run again and enter 1 or 2.")


if __name__ == "__main__":
    main()