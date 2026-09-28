import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance


#=========================================================
#STEP 1: LOAD THE FREQUENCY DATA
#=========================================================


#Download the policy-level frequency dataset
print("Loading frequency data...")

freq = fetch_openml(
    data_id=41214,
    as_frame=True
).data


#Convert policy ID to an integer
freq["IDpol"] = freq["IDpol"].astype(int)


print("\nFrequency data loaded successfully")


#=========================================================
#STEP 2: CREATE THE FREQUENCY MODELING DATASET
#=========================================================


#Create a copy so the original data remains unchanged
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
#Frequency = Number of Claims / Exposure
freq_model_df["Frequency"] = (
    freq_model_df["ClaimNb"]
    / freq_model_df["Exposure"]
)


#Create log population density
#The density variable is strongly right-skewed
freq_model_df["LogDensity"] = (
    np.log1p(freq_model_df["Density"])
)


#=========================================================
#STEP 3: CREATE MODELING VARIABLES
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


#=========================================================
#STEP 4: SELECT MODEL FEATURES
#=========================================================


#Categorical variables will be converted into indicator variables
categorical_features = [
    "Area",
    "VehBrand",
    "VehGas",
    "Region",
    "DriverAgeGroup",
    "VehicleAgeGroup",
    "BonusMalusGroup"
]


#Numeric variables remain continuous
numeric_features = [
    "VehPower",
    "LogDensity"
]


#Combine the variables used by the model
model_features = (
    categorical_features
    + numeric_features
)


#Create the predictor dataset
X = freq_model_df[
    model_features
]


#Create the response variable
#The model predicts claims per policy-year
y = freq_model_df[
    "Frequency"
]


#Exposure will be used as the observation weight
exposure = freq_model_df[
    "Exposure"
]


#=========================================================
#STEP 5: SPLIT DATA INTO TRAINING AND TESTING SETS
#=========================================================


#Split the dataset so the model is evaluated on policies
#that were not used when estimating the coefficients
X_train, X_test, y_train, y_test, exposure_train, exposure_test = (
    train_test_split(
        X,
        y,
        exposure,
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
    f"Training exposure: "
    f"{exposure_train.sum():,.2f}"
)

print(
    f"Testing exposure: "
    f"{exposure_test.sum():,.2f}"
)


#=========================================================
#STEP 6: PREPROCESS MODEL VARIABLES
#=========================================================


#Convert categorical variables into indicator variables
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)


#Standardize numeric variables
numeric_transformer = StandardScaler()


#Apply the appropriate transformation to each variable type
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
#STEP 7: BUILD THE POISSON GLM
#=========================================================


#Create the Poisson regression model
#The log link is built into PoissonRegressor
poisson_model = PoissonRegressor(
    alpha=0.1,
    max_iter=500
)


#Combine preprocessing and regression into one pipeline
frequency_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            poisson_model
        )
    ]
)


#=========================================================
#STEP 8: FIT THE MODEL
#=========================================================


print("\nFitting Poisson frequency model...")


#Fit the model using exposure as the observation weight
frequency_model.fit(
    X_train,
    y_train,
    model__sample_weight=exposure_train
)


print("Model fitted successfully")


#=========================================================
#STEP 9: GENERATE TEST SET PREDICTIONS
#=========================================================


#Predict claim frequency for policies in the test dataset
predicted_frequency = frequency_model.predict(
    X_test
)


#Calculate actual test-set portfolio frequency
actual_test_frequency = (
    np.sum(
        y_test
        * exposure_test
    )
    / exposure_test.sum()
)


#Calculate predicted test-set portfolio frequency
predicted_test_frequency = (
    np.sum(
        predicted_frequency
        * exposure_test
    )
    / exposure_test.sum()
)


#=========================================================
#STEP 10: EVALUATE THE MODEL
#=========================================================


#Calculate the average training frequency
#This acts as a simple baseline prediction
baseline_frequency = np.average(
    y_train,
    weights=exposure_train
)


#Give every test policy the same baseline frequency
baseline_predictions = np.full(
    len(y_test),
    baseline_frequency
)


#Calculate Poisson deviance for the baseline model
baseline_deviance = mean_poisson_deviance(
    y_test,
    baseline_predictions,
    sample_weight=exposure_test
)


#Calculate Poisson deviance for the fitted GLM
model_deviance = mean_poisson_deviance(
    y_test,
    predicted_frequency,
    sample_weight=exposure_test
)


#Display model performance
print("\n======================================")
print("POISSON FREQUENCY MODEL RESULTS")
print("======================================")

print(
    f"Actual test frequency: "
    f"{actual_test_frequency:.4f}"
)

print(
    f"Predicted test frequency: "
    f"{predicted_test_frequency:.4f}"
)

print(
    f"Baseline Poisson deviance: "
    f"{baseline_deviance:.6f}"
)

print(
    f"Model Poisson deviance: "
    f"{model_deviance:.6f}"
)

#=========================================================
#STEP 11: EXTRACT MODEL COEFFICIENTS AND RELATIVITIES
#=========================================================


#---------------------------------------------------------
# Get Feature Names From the Preprocessing Pipeline
#---------------------------------------------------------

#Access the fitted preprocessing step
fitted_preprocessor = (
    frequency_model
    .named_steps["preprocessor"]
)


#Get the names of all transformed model variables
feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


#---------------------------------------------------------
# Get Poisson Model Coefficients
#---------------------------------------------------------

#Access the fitted Poisson regression model
fitted_poisson_model = (
    frequency_model
    .named_steps["model"]
)


#Extract the estimated model coefficients
coefficients = (
    fitted_poisson_model.coef_
)


#---------------------------------------------------------
# Create Coefficient Table
#---------------------------------------------------------

#Create a table containing each variable and its coefficient
coefficient_table = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients
    }
)


#Convert log-scale coefficients into multiplicative relativities
#Relativity = e raised to the coefficient
coefficient_table["Relativity"] = (
    np.exp(
        coefficient_table["Coefficient"]
    )
)


#Calculate the percentage change relative to 1.00
coefficient_table["PercentChange"] = (
    (
        coefficient_table["Relativity"]
        - 1
    )
    * 100
)


#---------------------------------------------------------
# Sort Rating Factors
#---------------------------------------------------------

#Sort from highest modeled relativity to lowest
coefficient_table = (
    coefficient_table
    .sort_values(
        "Relativity",
        ascending=False
    )
    .reset_index(drop=True)
)


#---------------------------------------------------------
# Display Rating Relativities
#---------------------------------------------------------

print("\n======================================")
print("POISSON MODEL RATING RELATIVITIES")
print("======================================")

print(
    coefficient_table.to_string(
        index=False,
        formatters={
            "Coefficient": "{:.4f}".format,
            "Relativity": "{:.3f}".format,
            "PercentChange": "{:+.1f}%".format
        }
    )
)

#=========================================================
#STEP 12: CALCULATE ADJUSTED FREQUENCY RELATIVITIES
#=========================================================


#---------------------------------------------------------
#Create a Function for Adjusted Relativities
#---------------------------------------------------------

#Change one rating variable at a time
#while keeping all other policy characteristics unchanged
def calculate_adjusted_relativities(
        variable,
        levels,
        base_level
):

    results = []


    #Calculate predicted frequency for each level
    for level in levels:

        #Create a copy of the test data
        scenario_data = X_test.copy()


        #Convert the selected variable to object type
        scenario_data[variable] = (
            scenario_data[variable]
            .astype(object)
        )


        #Assign every policy to the selected category
        scenario_data[variable] = level


        #Generate predicted claim frequencies
        scenario_predictions = (
            frequency_model.predict(
                scenario_data
            )
        )


        #Calculate exposure-weighted predicted frequency
        adjusted_frequency = np.average(
            scenario_predictions,
            weights=exposure_test
        )


        #Store the result
        results.append(
            {
                "Level": level,
                "AdjustedFrequency": adjusted_frequency
            }
        )


    #Convert results into a DataFrame
    relativity_table = pd.DataFrame(
        results
    )


    #Find the predicted frequency for the base category
    base_frequency = (
        relativity_table.loc[
            relativity_table["Level"] == base_level,
            "AdjustedFrequency"
        ]
        .iloc[0]
    )


    #Calculate relativity compared with the base category
    relativity_table["Relativity"] = (
        relativity_table["AdjustedFrequency"]
        / base_frequency
    )


    #Convert frequency to claims per 100 policy-years
    relativity_table["ClaimsPer100PolicyYears"] = (
        relativity_table["AdjustedFrequency"]
        * 100
    )


    #Calculate percentage difference from the base category
    relativity_table["PercentDifference"] = (
        (
            relativity_table["Relativity"]
            - 1
        )
        * 100
    )


    return relativity_table


#=========================================================
#STEP 12A: DRIVER AGE RELATIVITIES
#=========================================================

age_relativities = calculate_adjusted_relativities(
    variable="DriverAgeGroup",
    levels=[
        "18-24",
        "25-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+"
    ],
    base_level="40-49"
)


print("\n======================================")
print("ADJUSTED DRIVER AGE RELATIVITIES")
print("Base Group: 40-49")
print("======================================")

print(
    age_relativities.to_string(
        index=False,
        formatters={
            "AdjustedFrequency": "{:.4f}".format,
            "Relativity": "{:.3f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "PercentDifference": "{:+.1f}%".format
        }
    )
)


#=========================================================
#STEP 12B: VEHICLE AGE RELATIVITIES
#=========================================================

vehicle_age_relativities = calculate_adjusted_relativities(
    variable="VehicleAgeGroup",
    levels=[
        "0-1",
        "2-5",
        "6-10",
        "11-15",
        "16-20",
        "21+"
    ],
    base_level="6-10"
)


print("\n======================================")
print("ADJUSTED VEHICLE AGE RELATIVITIES")
print("Base Group: 6-10")
print("======================================")

print(
    vehicle_age_relativities.to_string(
        index=False,
        formatters={
            "AdjustedFrequency": "{:.4f}".format,
            "Relativity": "{:.3f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "PercentDifference": "{:+.1f}%".format
        }
    )
)


#=========================================================
#STEP 12C: BONUS-MALUS RELATIVITIES
#=========================================================

bonus_malus_relativities = calculate_adjusted_relativities(
    variable="BonusMalusGroup",
    levels=[
        "50",
        "51-60",
        "61-70",
        "71-80",
        "81-100",
        "101+"
    ],
    base_level="50"
)


print("\n======================================")
print("ADJUSTED BONUS-MALUS RELATIVITIES")
print("Base Group: 50")
print("======================================")

print(
    bonus_malus_relativities.to_string(
        index=False,
        formatters={
            "AdjustedFrequency": "{:.4f}".format,
            "Relativity": "{:.3f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "PercentDifference": "{:+.1f}%".format
        }
    )
)