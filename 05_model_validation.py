import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_tweedie_deviance


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
#STEP 2: CREATE A CONSISTENT FREQUENCY-SEVERITY DATASET
#=========================================================


#Sum claim amounts for policies with multiple claim records
sev_by_policy = (
    sev.groupby(
        "IDpol",
        as_index=False
    )["ClaimAmount"]
    .sum()
)


#Attach observed claim dollars to every policy
validation_df = freq.merge(
    sev_by_policy,
    on="IDpol",
    how="left"
)


#Policies without recorded severity data receive zero claim dollars
validation_df["ClaimAmount"] = (
    validation_df["ClaimAmount"]
    .fillna(0)
)


#Cap exposure at one policy-year
validation_df["Exposure"] = (
    validation_df["Exposure"]
    .clip(upper=1)
)


#Cap unusually high claim counts at four
validation_df["ClaimNb"] = (
    validation_df["ClaimNb"]
    .clip(upper=4)
)


#Cap extreme claim amounts at 200,000 euros
validation_df["ClaimAmount"] = (
    validation_df["ClaimAmount"]
    .clip(upper=200000)
)


#Identify policies that report claims but have no recorded claim amount
claims_without_amount = (
    (validation_df["ClaimNb"] > 0)
    &
    (validation_df["ClaimAmount"] == 0)
)


#Set claim count to zero when no observed claim amount exists
#This makes frequency and severity experience internally consistent
validation_df.loc[
    claims_without_amount,
    "ClaimNb"
] = 0


#Calculate observed claim frequency
validation_df["Frequency"] = (
    validation_df["ClaimNb"]
    / validation_df["Exposure"]
)


#Calculate observed pure premium
validation_df["PurePremium"] = (
    validation_df["ClaimAmount"]
    / validation_df["Exposure"]
)


#Create log population density
validation_df["LogDensity"] = (
    np.log1p(
        validation_df["Density"]
    )
)


print("\n======================================")
print("CONSISTENT VALIDATION DATASET")
print("======================================")

print(
    f"Policies: "
    f"{len(validation_df):,}"
)

print(
    f"Exposure: "
    f"{validation_df['Exposure'].sum():,.2f}"
)

print(
    f"Claims: "
    f"{validation_df['ClaimNb'].sum():,.0f}"
)

print(
    f"Claim amount: "
    f"€{validation_df['ClaimAmount'].sum():,.2f}"
)

print(
    f"Policies with claims removed because no loss amount existed: "
    f"{claims_without_amount.sum():,}"
)


#=========================================================
#STEP 3: CREATE RATING VARIABLES
#=========================================================


#---------------------------------------------------------
#Create Driver Age Groups
#---------------------------------------------------------

validation_df["DriverAgeGroup"] = pd.cut(
    validation_df["DrivAge"],
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

validation_df["VehicleAgeGroup"] = pd.cut(
    validation_df["VehAge"],
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

validation_df["BonusMalusGroup"] = pd.cut(
    validation_df["BonusMalus"],
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


#Categorical rating variables
categorical_features = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
    "DriverAgeGroup",
    "VehicleAgeGroup",
    "BonusMalusGroup"
]


#Continuous rating variables
numeric_features = [
    "VehPower",
    "LogDensity"
]


#Combine all model variables
model_features = (
    categorical_features
    + numeric_features
)


#=========================================================
#STEP 5: SPLIT INTO TRAINING AND TESTING DATA
#=========================================================


#Create row indices so frequency and severity models
#use the same underlying training and testing populations
train_index, test_index = train_test_split(
    validation_df.index,
    test_size=0.20,
    random_state=42
)


#Create training and testing datasets
train_df = (
    validation_df.loc[
        train_index
    ]
    .copy()
)

test_df = (
    validation_df.loc[
        test_index
    ]
    .copy()
)


print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print(
    f"Training policies: "
    f"{len(train_df):,}"
)

print(
    f"Testing policies: "
    f"{len(test_df):,}"
)

print(
    f"Training exposure: "
    f"{train_df['Exposure'].sum():,.2f}"
)

print(
    f"Testing exposure: "
    f"{test_df['Exposure'].sum():,.2f}"
)


#=========================================================
#STEP 6: CREATE PREPROCESSING FUNCTION
#=========================================================


#Create independent preprocessing pipelines
#for the frequency and severity models
def create_preprocessor():

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )


    numeric_transformer = StandardScaler()


    return ColumnTransformer(
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
#STEP 7: FIT POISSON FREQUENCY MODEL
#=========================================================


#Create frequency modeling inputs
X_frequency_train = train_df[
    model_features
]


y_frequency_train = train_df[
    "Frequency"
]


frequency_weights_train = train_df[
    "Exposure"
]


#Create the Poisson GLM pipeline
frequency_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            PoissonRegressor(
                alpha=0.1,
                max_iter=500
            )
        )
    ]
)


print("\nFitting validation Poisson frequency model...")


#Fit the Poisson model
frequency_model.fit(
    X_frequency_train,
    y_frequency_train,
    model__sample_weight=frequency_weights_train
)


print("Validation Poisson model fitted successfully")


#=========================================================
#STEP 8: FIT GAMMA SEVERITY MODEL
#=========================================================


#Only claim-bearing policies can be used for severity modeling
severity_train_df = train_df[
    (train_df["ClaimNb"] > 0)
    &
    (train_df["ClaimAmount"] > 0)
].copy()


#Calculate average severity for each claim-bearing policy
severity_train_df["AvgClaimAmount"] = (
    severity_train_df["ClaimAmount"]
    / severity_train_df["ClaimNb"]
)


#Create severity model inputs
X_severity_train = severity_train_df[
    model_features
]


y_severity_train = severity_train_df[
    "AvgClaimAmount"
]


severity_weights_train = severity_train_df[
    "ClaimNb"
]


#Create the Gamma GLM pipeline
severity_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            GammaRegressor(
                alpha=0.1,
                max_iter=500
            )
        )
    ]
)


print("\nFitting validation Gamma severity model...")


#Fit the Gamma model
severity_model.fit(
    X_severity_train,
    y_severity_train,
    model__sample_weight=severity_weights_train
)


print("Validation Gamma model fitted successfully")


#=========================================================
#STEP 9: GENERATE COMBINED TEST PREDICTIONS
#=========================================================


#Create a copy of the test data for predictions
validation_results = test_df.copy()


#Predict claim frequency
validation_results["PredictedFrequency"] = (
    frequency_model.predict(
        validation_results[
            model_features
        ]
    )
)


#Predict claim severity
validation_results["PredictedSeverity"] = (
    severity_model.predict(
        validation_results[
            model_features
        ]
    )
)


#Calculate predicted pure premium
validation_results["PredictedPurePremium"] = (
    validation_results["PredictedFrequency"]
    * validation_results["PredictedSeverity"]
)


#Calculate predicted claim count for actual exposure
validation_results["PredictedClaims"] = (
    validation_results["PredictedFrequency"]
    * validation_results["Exposure"]
)


#Calculate predicted total loss amount
validation_results["PredictedLossAmount"] = (
    validation_results["PredictedPurePremium"]
    * validation_results["Exposure"]
)


#=========================================================
#STEP 10: CALCULATE ACTUAL TEST EXPERIENCE
#=========================================================


#Calculate total test exposure
test_exposure = (
    validation_results["Exposure"].sum()
)


#Calculate actual claims
actual_claims = (
    validation_results["ClaimNb"].sum()
)


#Calculate actual losses
actual_losses = (
    validation_results["ClaimAmount"].sum()
)


#Calculate actual claim frequency
actual_frequency = (
    actual_claims
    / test_exposure
)


#Calculate actual average severity
actual_severity = (
    actual_losses
    / actual_claims
)


#Calculate actual pure premium
actual_pure_premium = (
    actual_losses
    / test_exposure
)


#=========================================================
#STEP 11: CALCULATE PREDICTED TEST EXPERIENCE
#=========================================================


#Calculate predicted claims
predicted_claims = (
    validation_results["PredictedClaims"].sum()
)


#Calculate predicted losses
predicted_losses = (
    validation_results["PredictedLossAmount"].sum()
)


#Calculate predicted frequency
predicted_frequency = (
    predicted_claims
    / test_exposure
)


#Calculate implied predicted severity
predicted_severity = (
    predicted_losses
    / predicted_claims
)


#Calculate predicted pure premium
predicted_pure_premium = (
    predicted_losses
    / test_exposure
)


#Calculate predicted versus actual loss difference
loss_difference = (
    predicted_losses
    - actual_losses
)


#Calculate percentage error in total losses
loss_percentage_error = (
    loss_difference
    / actual_losses
    * 100
)


#=========================================================
#STEP 12: DISPLAY COMBINED VALIDATION RESULTS
#=========================================================


print("\n======================================")
print("COMBINED PRICING MODEL VALIDATION")
print("======================================")

print(
    f"Test policies: "
    f"{len(validation_results):,}"
)

print(
    f"Test exposure: "
    f"{test_exposure:,.2f}"
)

print("\n--- FREQUENCY ---")

print(
    f"Actual claims: "
    f"{actual_claims:,.0f}"
)

print(
    f"Predicted claims: "
    f"{predicted_claims:,.2f}"
)

print(
    f"Actual frequency: "
    f"{actual_frequency:.4f}"
)

print(
    f"Predicted frequency: "
    f"{predicted_frequency:.4f}"
)

print("\n--- SEVERITY ---")

print(
    f"Actual severity: "
    f"€{actual_severity:,.2f}"
)

print(
    f"Predicted severity: "
    f"€{predicted_severity:,.2f}"
)

print("\n--- PURE PREMIUM ---")

print(
    f"Actual pure premium: "
    f"€{actual_pure_premium:,.2f}"
)

print(
    f"Predicted pure premium: "
    f"€{predicted_pure_premium:,.2f}"
)

print("\n--- TOTAL LOSSES ---")

print(
    f"Actual losses: "
    f"€{actual_losses:,.2f}"
)

print(
    f"Predicted losses: "
    f"€{predicted_losses:,.2f}"
)

print(
    f"Predicted minus actual: "
    f"€{loss_difference:,.2f}"
)

print(
    f"Loss prediction error: "
    f"{loss_percentage_error:+.2f}%"
)


#=========================================================
#STEP 13: PURE PREMIUM DEVIANCE TEST
#=========================================================


#Calculate the training portfolio pure premium
training_pure_premium = (
    train_df["ClaimAmount"].sum()
    / train_df["Exposure"].sum()
)


#Create a constant baseline prediction
baseline_pure_premium = np.full(
    len(validation_results),
    training_pure_premium
)


#Calculate Tweedie deviance for the baseline model
#Power 1.5 represents a compound Poisson-Gamma structure
baseline_tweedie_deviance = mean_tweedie_deviance(
    validation_results["PurePremium"],
    baseline_pure_premium,
    sample_weight=validation_results["Exposure"],
    power=1.5
)


#Calculate Tweedie deviance for the combined pricing model
model_tweedie_deviance = mean_tweedie_deviance(
    validation_results["PurePremium"],
    validation_results["PredictedPurePremium"],
    sample_weight=validation_results["Exposure"],
    power=1.5
)


#Calculate percentage improvement versus baseline
deviance_improvement = (
    (
        baseline_tweedie_deviance
        - model_tweedie_deviance
    )
    / baseline_tweedie_deviance
    * 100
)


print("\n======================================")
print("PURE PREMIUM MODEL PERFORMANCE")
print("======================================")

print(
    f"Baseline Tweedie deviance: "
    f"{baseline_tweedie_deviance:.6f}"
)

print(
    f"Model Tweedie deviance: "
    f"{model_tweedie_deviance:.6f}"
)

print(
    f"Deviance improvement: "
    f"{deviance_improvement:.2f}%"
)


#=========================================================
#STEP 14: CREATE RISK DECILES
#=========================================================


#Divide policies into ten groups based on predicted pure premium
#Decile 1 contains lower predicted risks
#Decile 10 contains higher predicted risks
validation_results["RiskDecile"] = pd.qcut(
    validation_results["PredictedPurePremium"],
    q=10,
    labels=[
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ],
    duplicates="drop"
)


#Aggregate actual and predicted results by risk decile
risk_deciles = (
    validation_results
    .groupby(
        "RiskDecile",
        observed=True
    )
    .agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        ActualClaims=("ClaimNb", "sum"),
        ActualLosses=("ClaimAmount", "sum"),
        PredictedClaims=("PredictedClaims", "sum"),
        PredictedLosses=("PredictedLossAmount", "sum")
    )
    .reset_index()
)


#Calculate actual pure premium within each decile
risk_deciles["ActualPurePremium"] = (
    risk_deciles["ActualLosses"]
    / risk_deciles["Exposure"]
)


#Calculate predicted pure premium within each decile
risk_deciles["PredictedPurePremium"] = (
    risk_deciles["PredictedLosses"]
    / risk_deciles["Exposure"]
)


#Calculate actual versus expected loss ratio
risk_deciles["ActualToExpected"] = (
    risk_deciles["ActualLosses"]
    / risk_deciles["PredictedLosses"]
)


print("\n======================================")
print("VALIDATION BY PREDICTED RISK DECILE")
print("======================================")

print(
    risk_deciles.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "ActualClaims": "{:,.0f}".format,
            "ActualLosses": "€{:,.2f}".format,
            "PredictedClaims": "{:,.2f}".format,
            "PredictedLosses": "€{:,.2f}".format,
            "ActualPurePremium": "€{:,.2f}".format,
            "PredictedPurePremium": "€{:,.2f}".format,
            "ActualToExpected": "{:.3f}".format
        }
    )
)


#=========================================================
#STEP 15: CALCULATE ORDERED GINI COEFFICIENT
#=========================================================


#Sort policies from lowest to highest predicted pure premium
lorenz_df = (
    validation_results
    .sort_values(
        "PredictedPurePremium"
    )
    .copy()
)


#Calculate cumulative exposure
lorenz_df["CumulativeExposure"] = (
    lorenz_df["Exposure"]
    .cumsum()
)


#Convert cumulative exposure into a percentage of total exposure
lorenz_df["CumulativeExposureShare"] = (
    lorenz_df["CumulativeExposure"]
    / lorenz_df["Exposure"].sum()
)


#Calculate cumulative actual losses
lorenz_df["CumulativeLoss"] = (
    lorenz_df["ClaimAmount"]
    .cumsum()
)


#Convert cumulative losses into a percentage of total losses
lorenz_df["CumulativeLossShare"] = (
    lorenz_df["CumulativeLoss"]
    / lorenz_df["ClaimAmount"].sum()
)


#Add the origin point to the Lorenz curve
lorenz_x = np.concatenate(
    (
        [0],
        lorenz_df["CumulativeExposureShare"].to_numpy()
    )
)

lorenz_y = np.concatenate(
    (
        [0],
        lorenz_df["CumulativeLossShare"].to_numpy()
    )
)


#Calculate the area underneath the Lorenz curve
lorenz_area = np.trapezoid(
    lorenz_y,
    lorenz_x
)


#Calculate ordered Gini coefficient
ordered_gini = (
    1
    - 2 * lorenz_area
)


print("\n======================================")
print("RISK RANKING PERFORMANCE")
print("======================================")

print(
    f"Ordered Gini coefficient: "
    f"{ordered_gini:.4f}"
)


#=========================================================
#STEP 16: CREATE VALIDATION CHARTS
#=========================================================


#---------------------------------------------------------
#Actual vs Predicted Pure Premium by Risk Decile
#---------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


plt.plot(
    risk_deciles["RiskDecile"],
    risk_deciles["ActualPurePremium"],
    marker="o",
    label="Actual"
)


plt.plot(
    risk_deciles["RiskDecile"],
    risk_deciles["PredictedPurePremium"],
    marker="o",
    label="Predicted"
)


plt.title(
    "Actual vs Predicted Pure Premium by Risk Decile"
)


plt.xlabel(
    "Predicted Risk Decile"
)


plt.ylabel(
    "Pure Premium (€)"
)


plt.legend()


plt.tight_layout()


#---------------------------------------------------------
#Ordered Lorenz Curve
#---------------------------------------------------------

plt.figure(
    figsize=(7, 7)
)


plt.plot(
    lorenz_x,
    lorenz_y,
    label="Pricing Model"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Ranking"
)


plt.title(
    "Ordered Lorenz Curve"
)


plt.xlabel(
    "Cumulative Share of Exposure"
)


plt.ylabel(
    "Cumulative Share of Actual Losses"
)


plt.legend()


plt.tight_layout()


#=========================================================
#STEP 17: EXPORT VALIDATION RESULTS
#=========================================================


#Create results folder if needed
os.makedirs(
    "results",
    exist_ok=True
)


#Export risk decile validation table
risk_deciles.to_csv(
    "results/model_validation_deciles.csv",
    index=False
)


#Export policy-level validation results
validation_results[
    [
        "IDpol",
        "Exposure",
        "ClaimNb",
        "ClaimAmount",
        "PredictedFrequency",
        "PredictedSeverity",
        "PredictedPurePremium",
        "PredictedClaims",
        "PredictedLossAmount",
        "RiskDecile"
    ]
].to_csv(
    "results/model_validation_policy_results.csv",
    index=False
)


print("\n======================================")
print("VALIDATION EXPORT COMPLETE")
print("======================================")

print(
    "Validation tables saved to the results folder."
)


#=========================================================
#DISPLAY ALL CHARTS
#=========================================================


#Display all validation charts
plt.show()


#=========================================================
#EXPORT MODEL VALIDATION SUMMARY
#=========================================================

#Create a one-row summary of the main validation metrics
validation_summary = pd.DataFrame(
    [
        {
            "BaselineTweedieDeviance": baseline_tweedie_deviance,
            "ModelTweedieDeviance": model_tweedie_deviance,
            "DevianceImprovement": deviance_improvement / 100,
            "OrderedGini": ordered_gini,
            "ActualPurePremium": actual_pure_premium,
            "PredictedPurePremium": predicted_pure_premium,
            "LossPredictionError": loss_percentage_error / 100
        }
    ]
)


#Export the validation summary
validation_summary.to_csv(
    "results/model_validation_summary.csv",
    index=False
)