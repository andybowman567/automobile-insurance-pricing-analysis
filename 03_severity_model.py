import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_gamma_deviance


#=========================================================
#STEP 1: LOAD THE DATA
#=========================================================


#Load the policy-level frequency data
print("Loading frequency data...")

freq = fetch_openml(
    data_id=41214,
    as_frame=True
).data


#Load the claim-level severity data
print("Loading severity data...")

sev = fetch_openml(
    data_id=41215,
    as_frame=True
).data


#Make policy IDs consistent between datasets
freq["IDpol"] = freq["IDpol"].astype(int)

sev["IDpol"] = sev["IDpol"].astype(int)


print("\nDatasets loaded successfully")


#=========================================================
#STEP 2: PREPARE CLAIM AMOUNT DATA
#=========================================================


#Sum claim amounts for policies with multiple claims
sev_by_policy = (
    sev.groupby(
        "IDpol",
        as_index=False
    )["ClaimAmount"]
    .sum()
)


#Merge claim amounts onto policy information
severity_df = freq.merge(
    sev_by_policy,
    on="IDpol",
    how="inner"
)


#Only keep policies with at least one reported claim
severity_df = severity_df[
    severity_df["ClaimNb"] > 0
].copy()


#Only keep positive claim amounts
severity_df = severity_df[
    severity_df["ClaimAmount"] > 0
].copy()


#Cap extreme total claim amounts at 200,000 euros
#This limits the influence of a small number of extreme losses
severity_df["ClaimAmount"] = (
    severity_df["ClaimAmount"]
    .clip(upper=200000)
)


#Calculate average claim severity for each policy
#Severity = Total Claim Amount / Number of Claims
severity_df["AvgClaimAmount"] = (
    severity_df["ClaimAmount"]
    / severity_df["ClaimNb"]
)


#Create log population density
severity_df["LogDensity"] = (
    np.log1p(
        severity_df["Density"]
    )
)


#=========================================================
#STEP 3: CREATE MODELING VARIABLES
#=========================================================


#---------------------------------------------------------
#Create Driver Age Groups
#---------------------------------------------------------

severity_df["DriverAgeGroup"] = pd.cut(
    severity_df["DrivAge"],
    bins=[
        17,
        24,
        29,
        39,
        49,
        59,
        69,
        100
    ],
    labels=[
        "18-24",
        "25-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+"
    ],
    include_lowest=True
)


#---------------------------------------------------------
#Create Vehicle Age Groups
#---------------------------------------------------------

severity_df["VehicleAgeGroup"] = pd.cut(
    severity_df["VehAge"],
    bins=[
        -1,
        1,
        5,
        10,
        15,
        20,
        100
    ],
    labels=[
        "0-1",
        "2-5",
        "6-10",
        "11-15",
        "16-20",
        "21+"
    ]
)


#---------------------------------------------------------
#Create Bonus-Malus Groups
#---------------------------------------------------------

severity_df["BonusMalusGroup"] = pd.cut(
    severity_df["BonusMalus"],
    bins=[
        49,
        50,
        60,
        70,
        80,
        100,
        230
    ],
    labels=[
        "50",
        "51-60",
        "61-70",
        "71-80",
        "81-100",
        "101+"
    ],
    include_lowest=True
)


#=========================================================
#STEP 4: SELECT MODEL FEATURES
#=========================================================


#Categorical variables used in the severity model
categorical_features = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
    "DriverAgeGroup",
    "VehicleAgeGroup",
    "BonusMalusGroup"
]


#Continuous variables used in the severity model
numeric_features = [
    "VehPower",
    "LogDensity"
]


#Combine all model variables
model_features = (
    categorical_features
    + numeric_features
)


#Create predictor dataset
X = severity_df[
    model_features
]


#Create severity response variable
y = severity_df[
    "AvgClaimAmount"
]


#Use claim count as the observation weight
#Policies with more claims provide more severity information
claim_weights = severity_df[
    "ClaimNb"
]


#=========================================================
#STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
#=========================================================


X_train, X_test, y_train, y_test, weight_train, weight_test = (
    train_test_split(
        X,
        y,
        claim_weights,
        test_size=0.20,
        random_state=42
    )
)


print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print(
    f"Training policies: "
    f"{len(X_train):,}"
)

print(
    f"Testing policies: "
    f"{len(X_test):,}"
)

print(
    f"Training claims: "
    f"{weight_train.sum():,.0f}"
)

print(
    f"Testing claims: "
    f"{weight_test.sum():,.0f}"
)


#=========================================================
#STEP 6: PREPROCESS MODEL VARIABLES
#=========================================================


#Convert categorical variables into indicator variables
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


#Standardize continuous variables
numeric_transformer = StandardScaler()


#Apply transformations to each variable type
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    ]
)


#=========================================================
#STEP 7: BUILD THE GAMMA GLM
#=========================================================


#Create Gamma regression model
#Gamma regression is appropriate for positive,
#right-skewed insurance claim amounts
gamma_model = GammaRegressor(
    alpha=0.1,
    max_iter=500
)


#Combine preprocessing and Gamma regression
severity_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            gamma_model
        )
    ]
)


#=========================================================
#STEP 8: FIT THE MODEL
#=========================================================


print("\nFitting Gamma severity model...")


#Fit using claim count as observation weight
severity_model.fit(
    X_train,
    y_train,
    model__sample_weight=weight_train
)


print("Model fitted successfully")


#=========================================================
#STEP 9: GENERATE TEST SET PREDICTIONS
#=========================================================


#Predict average claim severity
predicted_severity = (
    severity_model.predict(
        X_test
    )
)


#Calculate actual weighted average severity
actual_test_severity = np.average(
    y_test,
    weights=weight_test
)


#Calculate predicted weighted average severity
predicted_test_severity = np.average(
    predicted_severity,
    weights=weight_test
)


#=========================================================
#STEP 10: EVALUATE THE GAMMA MODEL
#=========================================================


#Calculate training-set average severity
#This becomes the baseline prediction
baseline_severity = np.average(
    y_train,
    weights=weight_train
)


#Assign the same baseline severity to every test policy
baseline_predictions = np.full(
    len(y_test),
    baseline_severity
)


#Calculate Gamma deviance for the baseline model
baseline_deviance = mean_gamma_deviance(
    y_test,
    baseline_predictions,
    sample_weight=weight_test
)


#Calculate Gamma deviance for the fitted model
model_deviance = mean_gamma_deviance(
    y_test,
    predicted_severity,
    sample_weight=weight_test
)


#Display model results
print("\n======================================")
print("GAMMA SEVERITY MODEL RESULTS")
print("======================================")

print(
    f"Actual test severity: "
    f"€{actual_test_severity:,.2f}"
)

print(
    f"Predicted test severity: "
    f"€{predicted_test_severity:,.2f}"
)

print(
    f"Baseline Gamma deviance: "
    f"{baseline_deviance:.6f}"
)

print(
    f"Model Gamma deviance: "
    f"{model_deviance:.6f}"
)