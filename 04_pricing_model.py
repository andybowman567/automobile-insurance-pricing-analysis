import pandas as pd
import numpy as np
import os

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, GammaRegressor


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
#STEP 2: PREPARE THE FREQUENCY MODELING DATA
#=========================================================


#Create a copy of the original frequency data
freq_model_df = freq.copy()


#Cap exposure at one policy-year
freq_model_df["Exposure"] = (
    freq_model_df["Exposure"]
    .clip(upper=1)
)


#Cap unusually high claim counts at four
freq_model_df["ClaimNb"] = (
    freq_model_df["ClaimNb"]
    .clip(upper=4)
)


#Calculate observed claim frequency
freq_model_df["Frequency"] = (
    freq_model_df["ClaimNb"]
    / freq_model_df["Exposure"]
)


#Create log population density
freq_model_df["LogDensity"] = (
    np.log1p(
        freq_model_df["Density"]
    )
)


#=========================================================
#STEP 3: PREPARE THE SEVERITY MODELING DATA
#=========================================================


#Sum claim amounts for policies with multiple claims
sev_by_policy = (
    sev.groupby(
        "IDpol",
        as_index=False
    )["ClaimAmount"]
    .sum()
)


#Merge policy characteristics with observed claim amounts
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


#Cap extreme claim amounts at 200,000 euros
severity_df["ClaimAmount"] = (
    severity_df["ClaimAmount"]
    .clip(upper=200000)
)


#Calculate average claim severity
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
#STEP 4: CREATE RATING VARIABLES
#=========================================================


#---------------------------------------------------------
#Create Driver Age Groups
#---------------------------------------------------------

freq_model_df["DriverAgeGroup"] = pd.cut(
    freq_model_df["DrivAge"],
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

freq_model_df["VehicleAgeGroup"] = pd.cut(
    freq_model_df["VehAge"],
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

freq_model_df["BonusMalusGroup"] = pd.cut(
    freq_model_df["BonusMalus"],
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
#STEP 5: SELECT MODEL FEATURES
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
#STEP 6: CREATE PREPROCESSING FUNCTION
#=========================================================


#Create a function so the frequency and severity models
#receive their own independent preprocessing pipelines
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
#STEP 7: FIT THE FINAL POISSON FREQUENCY MODEL
#=========================================================


#Create the final Poisson modeling pipeline
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


#Create frequency model inputs
X_frequency = freq_model_df[
    model_features
]

y_frequency = freq_model_df[
    "Frequency"
]

frequency_weights = freq_model_df[
    "Exposure"
]


print("\nFitting final Poisson frequency model...")


#Fit the final model using all available frequency data
frequency_model.fit(
    X_frequency,
    y_frequency,
    model__sample_weight=frequency_weights
)


print("Final Poisson model fitted successfully")


#=========================================================
#STEP 8: FIT THE FINAL GAMMA SEVERITY MODEL
#=========================================================


#Create the final Gamma modeling pipeline
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


#Create severity model inputs
X_severity = severity_df[
    model_features
]

y_severity = severity_df[
    "AvgClaimAmount"
]

severity_weights = severity_df[
    "ClaimNb"
]


print("\nFitting final Gamma severity model...")


#Fit the final model using all available severity data
severity_model.fit(
    X_severity,
    y_severity,
    model__sample_weight=severity_weights
)


print("Final Gamma model fitted successfully")


#=========================================================
#STEP 9: CREATE POLICY-LEVEL PRICING PREDICTIONS
#=========================================================


#Create a copy for final pricing results
pricing_df = freq_model_df.copy()


#Predict expected claim frequency for every policy
pricing_df["PredictedFrequency"] = (
    frequency_model.predict(
        pricing_df[
            model_features
        ]
    )
)


#Predict expected claim severity for every policy
pricing_df["PredictedSeverity"] = (
    severity_model.predict(
        pricing_df[
            model_features
        ]
    )
)


#Calculate policy-level expected claim cost
#Pure Premium = Frequency x Severity
pricing_df["PredictedPurePremium"] = (
    pricing_df["PredictedFrequency"]
    * pricing_df["PredictedSeverity"]
)


#Calculate expected claim count for each policy exposure
pricing_df["ExpectedClaims"] = (
    pricing_df["PredictedFrequency"]
    * pricing_df["Exposure"]
)


#Calculate expected total loss dollars for each policy
pricing_df["ExpectedLossAmount"] = (
    pricing_df["PredictedPurePremium"]
    * pricing_df["Exposure"]
)


#=========================================================
#STEP 10: PORTFOLIO PRICING SUMMARY
#=========================================================


#Calculate total predicted claims
predicted_total_claims = (
    pricing_df["ExpectedClaims"].sum()
)


#Calculate total predicted losses
predicted_total_losses = (
    pricing_df["ExpectedLossAmount"].sum()
)


#Calculate total portfolio exposure
pricing_total_exposure = (
    pricing_df["Exposure"].sum()
)


#Calculate portfolio predicted frequency
portfolio_predicted_frequency = (
    predicted_total_claims
    / pricing_total_exposure
)


#Calculate portfolio predicted pure premium
portfolio_predicted_pure_premium = (
    predicted_total_losses
    / pricing_total_exposure
)


#Calculate expected severity across predicted claims
portfolio_predicted_severity = (
    predicted_total_losses
    / predicted_total_claims
)


print("\n======================================")
print("PORTFOLIO PRICING MODEL RESULTS")
print("======================================")

print(
    f"Total policies: "
    f"{len(pricing_df):,}"
)

print(
    f"Total exposure: "
    f"{pricing_total_exposure:,.2f}"
)

print(
    f"Predicted total claims: "
    f"{predicted_total_claims:,.2f}"
)

print(
    f"Predicted claim frequency: "
    f"{portfolio_predicted_frequency:.4f}"
)

print(
    f"Predicted average severity: "
    f"€{portfolio_predicted_severity:,.2f}"
)

print(
    f"Predicted pure premium: "
    f"€{portfolio_predicted_pure_premium:,.2f}"
)

print(
    f"Predicted total losses: "
    f"€{predicted_total_losses:,.2f}"
)


#=========================================================
#STEP 11: CREATE PRICING RELATIVITY FUNCTION
#=========================================================


#Create pricing summaries for different rating variables
def create_pricing_summary(
        dataframe,
        grouping_variable
):

    #Aggregate expected claims and losses by risk group
    summary = (
        dataframe
        .groupby(
            grouping_variable,
            observed=True
        )
        .agg(
            Policies=("IDpol", "count"),
            Exposure=("Exposure", "sum"),
            ExpectedClaims=("ExpectedClaims", "sum"),
            ExpectedLosses=("ExpectedLossAmount", "sum")
        )
        .reset_index()
    )


    #Calculate predicted frequency
    summary["PredictedFrequency"] = (
        summary["ExpectedClaims"]
        / summary["Exposure"]
    )


    #Calculate predicted severity
    summary["PredictedSeverity"] = (
        summary["ExpectedLosses"]
        / summary["ExpectedClaims"]
    )


    #Calculate expected loss cost per policy-year
    summary["PredictedPurePremium"] = (
        summary["ExpectedLosses"]
        / summary["Exposure"]
    )


    #Calculate pricing relativity versus portfolio average
    summary["PricingRelativity"] = (
        summary["PredictedPurePremium"]
        / portfolio_predicted_pure_premium
    )


    return summary


#=========================================================
#STEP 12: DRIVER AGE PRICING RELATIVITIES
#=========================================================


driver_age_pricing = create_pricing_summary(
    pricing_df,
    "DriverAgeGroup"
)


print("\n======================================")
print("PRICING RELATIVITY BY DRIVER AGE")
print("======================================")

print(
    driver_age_pricing.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "ExpectedClaims": "{:,.2f}".format,
            "ExpectedLosses": "€{:,.2f}".format,
            "PredictedFrequency": "{:.4f}".format,
            "PredictedSeverity": "€{:,.2f}".format,
            "PredictedPurePremium": "€{:,.2f}".format,
            "PricingRelativity": "{:.3f}".format
        }
    )
)


#=========================================================
#STEP 13: VEHICLE AGE PRICING RELATIVITIES
#=========================================================


vehicle_age_pricing = create_pricing_summary(
    pricing_df,
    "VehicleAgeGroup"
)


print("\n======================================")
print("PRICING RELATIVITY BY VEHICLE AGE")
print("======================================")

print(
    vehicle_age_pricing.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "ExpectedClaims": "{:,.2f}".format,
            "ExpectedLosses": "€{:,.2f}".format,
            "PredictedFrequency": "{:.4f}".format,
            "PredictedSeverity": "€{:,.2f}".format,
            "PredictedPurePremium": "€{:,.2f}".format,
            "PricingRelativity": "{:.3f}".format
        }
    )
)


#=========================================================
#STEP 14: BONUS-MALUS PRICING RELATIVITIES
#=========================================================


bonus_malus_pricing = create_pricing_summary(
    pricing_df,
    "BonusMalusGroup"
)


print("\n======================================")
print("PRICING RELATIVITY BY BONUS-MALUS")
print("======================================")

print(
    bonus_malus_pricing.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "ExpectedClaims": "{:,.2f}".format,
            "ExpectedLosses": "€{:,.2f}".format,
            "PredictedFrequency": "{:.4f}".format,
            "PredictedSeverity": "€{:,.2f}".format,
            "PredictedPurePremium": "€{:,.2f}".format,
            "PricingRelativity": "{:.3f}".format
        }
    )
)


#=========================================================
#STEP 15: GEOGRAPHIC DENSITY PRICING RELATIVITIES
#=========================================================


#Create density quintiles for reporting purposes
pricing_df["DensityGroup"] = pd.qcut(
    pricing_df["Density"],
    q=5,
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    duplicates="drop"
)


density_pricing = create_pricing_summary(
    pricing_df,
    "DensityGroup"
)


print("\n======================================")
print("PRICING RELATIVITY BY GEOGRAPHIC DENSITY")
print("======================================")

print(
    density_pricing.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "ExpectedClaims": "{:,.2f}".format,
            "ExpectedLosses": "€{:,.2f}".format,
            "PredictedFrequency": "{:.4f}".format,
            "PredictedSeverity": "€{:,.2f}".format,
            "PredictedPurePremium": "€{:,.2f}".format,
            "PricingRelativity": "{:.3f}".format
        }
    )
)


#=========================================================
#STEP 16: EXPORT POLICY-LEVEL PRICING RESULTS
#=========================================================


#Create a results folder if it does not already exist
os.makedirs(
    "results",
    exist_ok=True
)


#Select important fields for the final pricing output
pricing_output = pricing_df[
    [
        "IDpol",
        "Exposure",
        "ClaimNb",
        "DrivAge",
        "VehAge",
        "VehPower",
        "BonusMalus",
        "Area",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
        "PredictedFrequency",
        "PredictedSeverity",
        "PredictedPurePremium",
        "ExpectedClaims",
        "ExpectedLossAmount"
    ]
]


#Export policy-level predictions to CSV
pricing_output.to_csv(
    "results/policy_pricing_results.csv",
    index=False
)


#Export pricing relativity tables
driver_age_pricing.to_csv(
    "results/driver_age_pricing.csv",
    index=False
)

vehicle_age_pricing.to_csv(
    "results/vehicle_age_pricing.csv",
    index=False
)

bonus_malus_pricing.to_csv(
    "results/bonus_malus_pricing.csv",
    index=False
)

density_pricing.to_csv(
    "results/density_pricing.csv",
    index=False
)


print("\n======================================")
print("EXPORT COMPLETE")
print("======================================")

print(
    "Policy-level pricing results and "
    "segment summaries saved to the results folder."
)