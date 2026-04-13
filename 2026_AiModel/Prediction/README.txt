# Energy Predictive Model Kit

This kit gives you everything you need to start a prediction-focused electricity model project.

## Files included
- `download_official_training_data.py`
- `train_energy_predictive_model.py`
- `requirements.txt`

## What this uses
- Commercial training data: 2018 CBECS public-use microdata CSV
- Residential training data: 2020 RECS public-use microdata CSV

## How to run

### 1) Install packages
```bash
pip install -r requirements.txt
```

### 2) Download the official data
```bash
python download_official_training_data.py
```

This saves the files into:
`~/Downloads/465data/training_data`

### 3) Train the model
```bash
python train_energy_predictive_model.py
```

## Output files
The training script will save:
- `~/Downloads/465data/model_outputs/model_summary.csv`
- `~/Downloads/465data/model_outputs/commercial_nn_model.joblib`
- `~/Downloads/465data/model_outputs/residential_nn_model.joblib`

## Important
The script uses starter column lists for both official datasets.
If EIA changes variable names or if you want different features, open
`train_energy_predictive_model.py` and edit:

- `COMMERCIAL_FEATURES`
- `RESIDENTIAL_FEATURES`
- `COMMERCIAL_TARGET`
- `RESIDENTIAL_TARGET`

The script will print any missing columns so you know what to change.
