import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


#======================================================
# Step 1: Load the Data
#======================================================


#Download the policy-level dataset from OpenML
print("Loading policy/frequency data...")
freq = fetch_openml(
    data_id=41214,
    as_frame=True,
).data


#Download the claim-level severity dataset from OpenML
print("Loading claim severity data...")
sev = fetch_openml(
    data_id=41215,
    as_frame=True,
).data


#Check that the data sets loaded successfully
print("\nDatasets loaded successfully")


#======================================================
# Step 2: Inspect the Data
#======================================================


#Display the number of rows and columns in the frequency dataset
print("\n--- Frequency Data Size ---")
print(freq.shape)
#Output format: (# policies, # variables)


#Display the number of rows and columns in the severity dataset
print("\n--- Severity Data Size ---")
print(sev.shape)
#Output format: (

#Display all the variable names in the frequency dataset
print("\n--- Frequency Columns ---")
print(freq.columns.tolist())


#Display all the variable names in the severity dataset
print("\n--- Severity Columns ---")
print(sev.columns.tolist())


#Display the first 5 policy records
print("\n--- First 5 Frequency Rows ---")
print(freq.head())


#Preview the first 5 claim records
print("\n--- First 5 Severity Rows ---")
print(sev.head())


#======================================================
# Step 3: The Basic Portfolio Stats
#======================================================

#Count total number of policie
total_policies = len(freq)

#Sum all the policy exposure
total_exposure = freq["Exposure"].sum()

#Sum the number of claims across all policies
total_claims = freq["ClaimNb"].sum()

#Calculate claims per polic-year of exposure
claim_frequency = total_claims / total_exposure

#Print the basic portfolio statistics.
print("\n======================================")
print("BASIC PORTFOLIO STATISTICS")
print("======================================")

print(f"Number of policies: {total_policies:,}")
print(f"Total exposure: {total_exposure:,.2f} policy-years")
print(f"Total claims: {total_claims:,.0f}")
print(f"Claim frequency: {claim_frequency:.4f}")
print(f"Claim frequency percentage: {claim_frequency * 100:.2f}%")


#=========================================================
# Step 4: Claim Severity and Pure Premium
#=========================================================

#Sum claim amounts for policies with multiple claims
sev_by_policy = (
    sev.groupby("IDpol",as_index=False)["ClaimAmount"]
    .sum()
)

#Merge claim dollars onto the policy-level frequency dataset
df = freq.merge(
    sev_by_policy,
    on="IDpol",
    how="left"
)

#Policies with no claims will have missing claim amounts after the merge
#Replace those missing values with 0
df["ClaimAmount"] = df["ClaimAmount"].fillna(0)


#======================================================
# Portfolio Loss Statistics
#======================================================

#Calculate the total amount paid for claims in the portfolio
total_claim_amount = df["ClaimAmount"].sum()

#Calculate average claim cost
average_severity = total_claim_amount / total_claims

#Calculate the expected cost per policy-holder
pure_premium = total_claim_amount / total_exposure

#Display the portfolio statistics.
print("\n======================================")
print("CLAIM SEVERITY AND PURE PREMIUM")
print("======================================")

print(f"Total claim amount: €{total_claim_amount:,.2f}")
print(f"Average claim severity: €{average_severity:,.2f}")
print(f"Pure premium: €{pure_premium:,.2f} per policy-year")

#======================================================
# Validate the frequency-severity relationship
#======================================================

#Pure premium is frequency x average severity
frequency_times_severity = claim_frequency * average_severity

#Display the result to compare to pure premium
print(
    f"Frequency × Severity: "
    f"€{frequency_times_severity:,.2f}"
)


#=========================================================
# Step 5: Data Validation and Cleaning Checks
#=========================================================

#---------------------------------------------------------
#Check Data Types
#---------------------------------------------------------


#Display how Python is storing each variable
#Numeric should be int or float
print("\n======================================")
print("DATA TYPES")
print("======================================")

print(freq.dtypes)

print("\nSeverity data types:")
print(sev.dtypes)


#---------------------------------------------------------
#Check Missing Values
#---------------------------------------------------------

#Count missing values in each variable to handle before modeling
print("\n======================================")
print("MISSING VALUES")
print("======================================")

print("\nFrequency data:")
print(freq.isnull().sum())

print("\nSeverity data:")
print(sev.isnull().sum())

#---------------------------------------------------------
# Check for Duplicate Policy IDs
#---------------------------------------------------------

#Each row should represent one policy
#Check IDpol should normally be unique in this dataset
duplicate_policy_ids = freq["IDpol"].duplicated().sum()

print("\n======================================")
print("DUPLICATE POLICY IDS")
print("======================================")

print(f"Duplicate policy IDs in frequency data: {duplicate_policy_ids:,}")


#---------------------------------------------------------
# Check Exposure Values
#---------------------------------------------------------

#Exposure should be positive because it represents time insured
zero_or_negative_exposure = (freq["Exposure"] <= 0).sum()

#If exposure is above 1, it means more than one policy-year is being recorded
exposure_above_one = (freq["Exposure"] > 1).sum()

print("\n======================================")
print("EXPOSURE CHECK")
print("======================================")

print(f"Exposure <= 0: {zero_or_negative_exposure:,}")
print(f"Exposure > 1: {exposure_above_one:,}")

print(f"Minimum exposure: {freq['Exposure'].min():.4f}")
print(f"Maximum exposure: {freq['Exposure'].max():.4f}")

#---------------------------------------------------------
# Check Claim Counts
#---------------------------------------------------------

#Claim counts should not be negative
negative_claim_counts = (freq["ClaimNb"] < 0).sum()

print("\n======================================")
print("CLAIM COUNT CHECK")
print("======================================")

print(f"Negative claim counts: {negative_claim_counts:,}")
print(f"Minimum claims on one policy: {freq['ClaimNb'].min():.0f}")
print(f"Maximum claims on one policy: {freq['ClaimNb'].max():.0f}")

#---------------------------------------------------------
# Check Claim Amounts
#---------------------------------------------------------

#Claim amounts should normally be positive
nonpositive_claim_amounts = (sev["ClaimAmount"] <= 0).sum()

print("\n======================================")
print("CLAIM AMOUNT CHECK")
print("======================================")

print(f"Claim amounts <= 0: {nonpositive_claim_amounts:,}")

print(f"Minimum claim amount: €{sev['ClaimAmount'].min():,.2f}")
print(f"Maximum claim amount: €{sev['ClaimAmount'].max():,.2f}")


#---------------------------------------------------------
# Exposure Important Numeric Variables
#---------------------------------------------------------


#Summarize the distribution of major pricing variables to enure there are no abnormalities
numeric_variables = [
    "Exposure",
    "ClaimNb",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density"
]

print("\n======================================")
print("NUMERIC VARIABLE SUMMARY")
print("======================================")

print(
    freq[numeric_variables]
    .describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99])
    .T
)


#---------------------------------------------------------
# Examine the Claim Amount Distribution
#---------------------------------------------------------

#Losses are typically right skewed, so large claims are no automatically outliers
print("\n======================================")
print("CLAIM AMOUNT DISTRIBUTION")
print("======================================")

print(
    sev["ClaimAmount"].describe(
        percentiles=[0.50, 0.90, 0.95, 0.99]
    )
)

#---------------------------------------------------------
# Reconcile Frequency and Severity Claim Counts
#---------------------------------------------------------

#Count the number of claim records for each policy
severity_claim_counts = (
    sev.groupby("IDpol")
    .size()
    .reset_index(name="SeverityClaimCount")
)


#Attach the severity claim count to the frequency data
claim_validation = freq[
    ["IDpol", "ClaimNb"]
].merge(
    severity_claim_counts,
    on="IDpol",
    how="left"
)


#Policies not found in the severity dataset have zero recorded severity claims
claim_validation["SeverityClaimCount"] = (
    claim_validation["SeverityClaimCount"]
    .fillna(0)
)


#Compare the claim count in the frequency dataset with the number of records in the severity dataset
claim_validation["ClaimCountDifference"] = (
    claim_validation["ClaimNb"]
    - claim_validation["SeverityClaimCount"]
)


#Count policies where the two sources agree or disagree
matching_claim_counts = (
    claim_validation["ClaimCountDifference"] == 0
).sum()

mismatching_claim_counts = (
    claim_validation["ClaimCountDifference"] != 0
).sum()


print("\n======================================")
print("CLAIM COUNT RECONCILIATION")
print("======================================")

print(
    f"Total claims from frequency data: "
    f"{total_claims:,.0f}"
)

print(
    f"Individual claim records in severity data: "
    f"{len(sev):,}"
)

print(
    f"Policies with matching claim counts: "
    f"{matching_claim_counts:,}"
)

print(
    f"Policies with mismatching claim counts: "
    f"{mismatching_claim_counts:,}"
)


#Show a few policies where the claim counts do not agree
print("\n--- Example Claim Count Mismatches ---")

print(
    claim_validation[
        claim_validation["ClaimCountDifference"] != 0
    ].head(10)
)


#=========================================================
# STEP 6A: Create Separate modeling Datasets
#=========================================================


#---------------------------------------------------------
# Frequency Modeling Dataset
#---------------------------------------------------------

#Create a copy of the original frequency data
#This keeps all policy-level claim count information
freq_model_df = freq.copy()


#Cap exposure at one policy-year
#This adjusts the small number of exposure values above 1
freq_model_df["Exposure"] = (
    freq_model_df["Exposure"]
    .clip(upper=1)
)


#Cap unusually high claim counts at four
#This prevents a few extreme observations from dominating the model
freq_model_df["ClaimNb"] = (
    freq_model_df["ClaimNb"]
    .clip(upper=4)
)


#Calculate observed claim frequency for each policy
#Frequency = Number of Claims / Exposure
freq_model_df["Frequency"] = (
    freq_model_df["ClaimNb"]
    / freq_model_df["Exposure"]
)


#---------------------------------------------------------
# Severity Modeling Dataset
#---------------------------------------------------------

#Start with the merged policy and claim amount data
#Only keep policies with a positive observed claim amount
sev_model_df = df[
    df["ClaimAmount"] > 0
].copy()


#Only keep policies that report at least one claim
sev_model_df = sev_model_df[
    sev_model_df["ClaimNb"] > 0
].copy()


#Cap extremely large claim amounts at 200,000 euros
#This limits the influence of a very small number of extreme losses
sev_model_df["ClaimAmount"] = (
    sev_model_df["ClaimAmount"]
    .clip(upper=200000)
)


#Calculate average observed severity for each policy
#Average Severity = Claim Amount / Number of Claims
sev_model_df["AvgClaimAmount"] = (
    sev_model_df["ClaimAmount"]
    / sev_model_df["ClaimNb"]
)


#=========================================================
# STEP 6B: Modeling Dataset Summary
#=========================================================


#---------------------------------------------------------
# Frequency Model Summary
#---------------------------------------------------------

#Calculate total exposure used for frequency modeling
frequency_model_exposure = (
    freq_model_df["Exposure"].sum()
)


#Calculate total claims used for frequency modeling
frequency_model_claims = (
    freq_model_df["ClaimNb"].sum()
)


#Calculate the overall modeled claim frequency
frequency_model_rate = (
    frequency_model_claims
    / frequency_model_exposure
)


#---------------------------------------------------------
# Severity Model Summary
#---------------------------------------------------------

#Calculate the number of policies used for severity modeling
severity_model_policies = len(sev_model_df)


#Calculate total claim dollars used for severity modeling
severity_model_claim_dollars = (
    sev_model_df["ClaimAmount"].sum()
)


#Calculate the mean policy-level severity
severity_model_mean = (
    sev_model_df["AvgClaimAmount"].mean()
)


#---------------------------------------------------------
# Display Modeling Dataset Summary
#---------------------------------------------------------

print("\n======================================")
print("FINAL MODELING DATASETS")
print("======================================")

print(
    f"Frequency modeling policies: "
    f"{len(freq_model_df):,}"
)

print(
    f"Frequency modeling exposure: "
    f"{frequency_model_exposure:,.2f}"
)

print(
    f"Frequency modeling claims: "
    f"{frequency_model_claims:,.0f}"
)

print(
    f"Modeled claim frequency: "
    f"{frequency_model_rate:.4f}"
)

print(
    f"Severity modeling policies: "
    f"{severity_model_policies:,}"
)

print(
    f"Severity modeling claim dollars: "
    f"€{severity_model_claim_dollars:,.2f}"
)

print(
    f"Mean policy-level severity: "
    f"€{severity_model_mean:,.2f}"
)


#=========================================================
#STEP 7: DRIVER AGE RISK SEGMENTATION
#=========================================================


#---------------------------------------------------------
# Create Driver Age Groups
#---------------------------------------------------------

#Create age boundaries for grouping drivers
age_bins = [
    17,
    24,
    29,
    39,
    49,
    59,
    69,
    100
]


#Create labels for each age group
age_labels = [
    "18-24",
    "25-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]


#Assign each driver to an age group
freq_model_df["DriverAgeGroup"] = pd.cut(
    freq_model_df["DrivAge"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)


#---------------------------------------------------------
# Calculate Claim Frequency by Driver Age
#---------------------------------------------------------

#Group policies by driver age and calculate:
#Number of policies
#Total exposure
#Total claims
age_frequency = (
    freq_model_df
    .groupby(
        "DriverAgeGroup",
        observed=True
    )
    .agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum")
    )
    .reset_index()
)


#Calculate exposure-weighted claim frequency
#Frequency = Total Claims / Total Exposure
age_frequency["ClaimFrequency"] = (
    age_frequency["Claims"]
    / age_frequency["Exposure"]
)


#Convert frequency into claims per 100 policy-years
age_frequency["ClaimsPer100PolicyYears"] = (
    age_frequency["ClaimFrequency"]
    * 100
)


#---------------------------------------------------------
# Display Driver Age Results
#---------------------------------------------------------

print("\n======================================")
print("CLAIM FREQUENCY BY DRIVER AGE")
print("======================================")

print(
    age_frequency.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "Claims": "{:,.0f}".format,
            "ClaimFrequency": "{:.4f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format
        }
    )
)


#---------------------------------------------------------
# Create Driver Age Claim Frequency Chart
#---------------------------------------------------------

#Create a bar chart showing claim frequency by driver age
plt.figure(figsize=(9, 5))

plt.bar(
    age_frequency["DriverAgeGroup"],
    age_frequency["ClaimsPer100PolicyYears"]
)


#Add a chart title
plt.title(
    "Claim Frequency by Driver Age Group"
)


#Label the horizontal axis
plt.xlabel(
    "Driver Age Group"
)


#Label the vertical axis
plt.ylabel(
    "Claims per 100 Policy-Years"
)


#Improve spacing around the chart
plt.tight_layout()


#=========================================================
# STEP 8: BONUS-MALUS RISK SEGMENTATION
#=========================================================


#---------------------------------------------------------
# Create Bonus-Malus Groups
#---------------------------------------------------------

#Create boundaries for grouping Bonus-Malus values
bonus_malus_bins = [
    49,
    50,
    60,
    70,
    80,
    100,
    230
]


#Create labels for each Bonus-Malus group
bonus_malus_labels = [
    "50",
    "51-60",
    "61-70",
    "71-80",
    "81-100",
    "101+"
]


#Assign each policy to a Bonus-Malus group
freq_model_df["BonusMalusGroup"] = pd.cut(
    freq_model_df["BonusMalus"],
    bins=bonus_malus_bins,
    labels=bonus_malus_labels,
    include_lowest=True
)


#---------------------------------------------------------
# Calculate Claim Frequency by Bonus-Malus Group
#---------------------------------------------------------

#Group policies and calculate policy count, exposure, and claims
bonus_malus_frequency = (
    freq_model_df
    .groupby(
        "BonusMalusGroup",
        observed=True
    )
    .agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum")
    )
    .reset_index()
)


#Calculate exposure-weighted claim frequency
bonus_malus_frequency["ClaimFrequency"] = (
    bonus_malus_frequency["Claims"]
    / bonus_malus_frequency["Exposure"]
)


#Convert frequency to claims per 100 policy-years
bonus_malus_frequency["ClaimsPer100PolicyYears"] = (
    bonus_malus_frequency["ClaimFrequency"]
    * 100
)


#---------------------------------------------------------
# Calculate Frequency Relativity
#---------------------------------------------------------

#Compare each Bonus-Malus group's frequency to the portfolio average
bonus_malus_frequency["FrequencyRelativity"] = (
    bonus_malus_frequency["ClaimFrequency"]
    / frequency_model_rate
)


#---------------------------------------------------------
# Display Bonus-Malus Results
#---------------------------------------------------------

print("\n======================================")
print("CLAIM FREQUENCY BY BONUS-MALUS")
print("======================================")

print(
    bonus_malus_frequency.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "Claims": "{:,.0f}".format,
            "ClaimFrequency": "{:.4f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "FrequencyRelativity": "{:.2f}".format
        }
    )
)


#---------------------------------------------------------
# Create Bonus-Malus Claim Frequency Chart
#---------------------------------------------------------

#Create a bar chart showing claim frequency by Bonus-Malus group
plt.figure(figsize=(9, 5))

plt.bar(
    bonus_malus_frequency["BonusMalusGroup"],
    bonus_malus_frequency["ClaimsPer100PolicyYears"]
)


#Add a chart title
plt.title(
    "Claim Frequency by Bonus-Malus Group"
)


#Label the horizontal axis
plt.xlabel(
    "Bonus-Malus Group"
)


#Label the vertical axis
plt.ylabel(
    "Claims per 100 Policy-Years"
)


#Improve spacing around the chart
plt.tight_layout()


#=========================================================
#STEP 9: VEHICLE AGE RISK SEGMENTATION
#=========================================================


#---------------------------------------------------------
# Create Vehicle Age Groups
#---------------------------------------------------------

#Create boundaries for grouping vehicle ages
vehicle_age_bins = [
    -1,
    1,
    5,
    10,
    15,
    20,
    100
]


#Create labels for each vehicle age group
vehicle_age_labels = [
    "0-1",
    "2-5",
    "6-10",
    "11-15",
    "16-20",
    "21+"
]


#Assign each policy to a vehicle age group
freq_model_df["VehicleAgeGroup"] = pd.cut(
    freq_model_df["VehAge"],
    bins=vehicle_age_bins,
    labels=vehicle_age_labels
)


#---------------------------------------------------------
# Calculate Claim Frequency by Vehicle Age
#---------------------------------------------------------

#Group policies by vehicle age and calculate:
#Number of policies
#Total exposure
#Total claims
vehicle_age_frequency = (
    freq_model_df
    .groupby(
        "VehicleAgeGroup",
        observed=True
    )
    .agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum")
    )
    .reset_index()
)


#Calculate exposure-weighted claim frequency
vehicle_age_frequency["ClaimFrequency"] = (
    vehicle_age_frequency["Claims"]
    / vehicle_age_frequency["Exposure"]
)


#Convert frequency to claims per 100 policy-years
vehicle_age_frequency["ClaimsPer100PolicyYears"] = (
    vehicle_age_frequency["ClaimFrequency"]
    * 100
)


#---------------------------------------------------------
# Calculate Frequency Relativity
#---------------------------------------------------------

#Compare each vehicle age group's frequency
#to the overall portfolio frequency
vehicle_age_frequency["FrequencyRelativity"] = (
    vehicle_age_frequency["ClaimFrequency"]
    / frequency_model_rate
)


#---------------------------------------------------------
# Display Vehicle Age Results
#---------------------------------------------------------

print("\n======================================")
print("CLAIM FREQUENCY BY VEHICLE AGE")
print("======================================")

print(
    vehicle_age_frequency.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "Claims": "{:,.0f}".format,
            "ClaimFrequency": "{:.4f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "FrequencyRelativity": "{:.2f}".format
        }
    )
)


#---------------------------------------------------------
# Create Vehicle Age Claim Frequency Chart
#---------------------------------------------------------

#Create a bar chart showing claim frequency by vehicle age
plt.figure(figsize=(9, 5))

plt.bar(
    vehicle_age_frequency["VehicleAgeGroup"],
    vehicle_age_frequency["ClaimsPer100PolicyYears"]
)


#Add a chart title
plt.title(
    "Claim Frequency by Vehicle Age Group"
)


#Label the horizontal axis
plt.xlabel(
    "Vehicle Age Group"
)


#Label the vertical axis
plt.ylabel(
    "Claims per 100 Policy-Years"
)


#Improve spacing around the chart
plt.tight_layout()


#=========================================================
#STEP 10: GEOGRAPHIC DENSITY RISK SEGMENTATION
#=========================================================


#---------------------------------------------------------
# Create Geographic Density Groups
#---------------------------------------------------------

#Divide policies into five groups based on population density
#Each group contains approximately the same number of policies
freq_model_df["DensityGroup"] = pd.qcut(
    freq_model_df["Density"],
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


#---------------------------------------------------------
# Calculate Claim Frequency by Density Group
#---------------------------------------------------------

#Group policies by geographic density and calculate:
#Number of policies
#Total exposure
#Total claims
density_frequency = (
    freq_model_df
    .groupby(
        "DensityGroup",
        observed=True
    )
    .agg(
        Policies=("IDpol", "count"),
        Exposure=("Exposure", "sum"),
        Claims=("ClaimNb", "sum"),
        AverageDensity=("Density", "mean")
    )
    .reset_index()
)


#Calculate exposure-weighted claim frequency
density_frequency["ClaimFrequency"] = (
    density_frequency["Claims"]
    / density_frequency["Exposure"]
)


#Convert frequency to claims per 100 policy-years
density_frequency["ClaimsPer100PolicyYears"] = (
    density_frequency["ClaimFrequency"]
    * 100
)


#---------------------------------------------------------
# Calculate Frequency Relativity
#---------------------------------------------------------

#Compare each density group's frequency
#to the overall portfolio frequency
density_frequency["FrequencyRelativity"] = (
    density_frequency["ClaimFrequency"]
    / frequency_model_rate
)


#---------------------------------------------------------
# Display Geographic Density Results
#---------------------------------------------------------

print("\n======================================")
print("CLAIM FREQUENCY BY GEOGRAPHIC DENSITY")
print("======================================")

print(
    density_frequency.to_string(
        index=False,
        formatters={
            "Policies": "{:,.0f}".format,
            "Exposure": "{:,.2f}".format,
            "Claims": "{:,.0f}".format,
            "AverageDensity": "{:,.0f}".format,
            "ClaimFrequency": "{:.4f}".format,
            "ClaimsPer100PolicyYears": "{:.2f}".format,
            "FrequencyRelativity": "{:.2f}".format
        }
    )
)


#---------------------------------------------------------
# Create Geographic Density Claim Frequency Chart
#---------------------------------------------------------

#Create a bar chart showing claim frequency by density group
plt.figure(figsize=(9, 5))

plt.bar(
    density_frequency["DensityGroup"],
    density_frequency["ClaimsPer100PolicyYears"]
)


#Add a chart title
plt.title(
    "Claim Frequency by Geographic Density"
)


#Label the horizontal axis
plt.xlabel(
    "Geographic Density Group"
)


#Label the vertical axis
plt.ylabel(
    "Claims per 100 Policy-Years"
)


#Improve spacing around the chart
plt.tight_layout()


#=========================================================
#DISPLAY ALL CHARTS
#=========================================================

#Display all charts created during the analysis
plt.show()

