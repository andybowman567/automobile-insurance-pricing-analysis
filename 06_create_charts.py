import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


#=========================================================
#STEP 1: CREATE CHARTS FOLDER
#=========================================================

#Create the charts folder if it does not already exist
os.makedirs(
    "charts",
    exist_ok=True
)


#=========================================================
#STEP 2: LOAD PRICING RESULTS
#=========================================================

#Load driver age pricing results
driver_age = pd.read_csv(
    "results/driver_age_pricing.csv"
)


#Load vehicle age pricing results
vehicle_age = pd.read_csv(
    "results/vehicle_age_pricing.csv"
)


#Load Bonus-Malus pricing results
bonus_malus = pd.read_csv(
    "results/bonus_malus_pricing.csv"
)


#Load geographic density pricing results
density = pd.read_csv(
    "results/density_pricing.csv"
)


#Load validation decile results
validation = pd.read_csv(
    "results/model_validation_deciles.csv"
)


#Load policy-level validation results
validation_policy = pd.read_csv(
    "results/model_validation_policy_results.csv"
)


print("Result files loaded successfully")


#=========================================================
#STEP 3: DRIVER AGE PRICING CHART
#=========================================================

#Create a chart showing expected loss cost by driver age
plt.figure(
    figsize=(9, 5)
)


plt.bar(
    driver_age["DriverAgeGroup"],
    driver_age["PredictedPurePremium"]
)


plt.title(
    "Predicted Pure Premium by Driver Age"
)


plt.xlabel(
    "Driver Age Group"
)


plt.ylabel(
    "Predicted Pure Premium (€)"
)


plt.tight_layout()


#Save the chart as a PNG file
plt.savefig(
    "charts/driver_age_pricing.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


#=========================================================
#STEP 4: BONUS-MALUS PRICING CHART
#=========================================================

#Create a chart showing pricing relativity by Bonus-Malus
plt.figure(
    figsize=(9, 5)
)


plt.bar(
    bonus_malus["BonusMalusGroup"],
    bonus_malus["PricingRelativity"]
)


#Add portfolio-average reference line
plt.axhline(
    y=1,
    linestyle="--"
)


plt.title(
    "Pricing Relativity by Bonus-Malus Group"
)


plt.xlabel(
    "Bonus-Malus Group"
)


plt.ylabel(
    "Pricing Relativity"
)


plt.tight_layout()


plt.savefig(
    "charts/bonus_malus_pricing_relativity.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


#=========================================================
#STEP 5: GEOGRAPHIC DENSITY PRICING CHART
#=========================================================

#Create a chart showing expected loss cost by geographic density
plt.figure(
    figsize=(9, 5)
)


plt.bar(
    density["DensityGroup"],
    density["PredictedPurePremium"]
)


plt.title(
    "Predicted Pure Premium by Geographic Density"
)


plt.xlabel(
    "Geographic Density Group"
)


plt.ylabel(
    "Predicted Pure Premium (€)"
)


plt.tight_layout()


plt.savefig(
    "charts/geographic_density_pricing.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


#=========================================================
#STEP 6: ACTUAL VS PREDICTED RISK DECILES
#=========================================================

#Create a chart comparing actual and predicted loss cost
#for policies grouped by predicted risk
plt.figure(
    figsize=(9, 5)
)


plt.plot(
    validation["RiskDecile"],
    validation["ActualPurePremium"],
    marker="o",
    label="Actual"
)


plt.plot(
    validation["RiskDecile"],
    validation["PredictedPurePremium"],
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


plt.savefig(
    "charts/actual_vs_predicted_risk_decile.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


#=========================================================
#STEP 7: ORDERED LORENZ CURVE
#=========================================================

#Sort policies from lowest to highest predicted pure premium
lorenz_df = (
    validation_policy
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


#Convert cumulative exposure to share of total exposure
lorenz_df["CumulativeExposureShare"] = (
    lorenz_df["CumulativeExposure"]
    / lorenz_df["Exposure"].sum()
)


#Calculate cumulative actual claim dollars
lorenz_df["CumulativeLoss"] = (
    lorenz_df["ClaimAmount"]
    .cumsum()
)


#Convert cumulative losses to share of total losses
lorenz_df["CumulativeLossShare"] = (
    lorenz_df["CumulativeLoss"]
    / lorenz_df["ClaimAmount"].sum()
)


#Add the origin point
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


#Calculate ordered Gini coefficient
lorenz_area = np.trapezoid(
    lorenz_y,
    lorenz_x
)


ordered_gini = (
    1
    - 2 * lorenz_area
)


#Create Lorenz curve
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
    f"Ordered Lorenz Curve | Gini = {ordered_gini:.3f}"
)


plt.xlabel(
    "Cumulative Share of Exposure"
)


plt.ylabel(
    "Cumulative Share of Actual Losses"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "charts/ordered_lorenz_curve.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


#=========================================================
#STEP 8: CONFIRM CHART EXPORT
#=========================================================

print("\n======================================")
print("CHART EXPORT COMPLETE")
print("======================================")

print("Saved:")
print("charts/driver_age_pricing.png")
print("charts/bonus_malus_pricing_relativity.png")
print("charts/geographic_density_pricing.png")
print("charts/actual_vs_predicted_risk_decile.png")
print("charts/ordered_lorenz_curve.png")

print(
    f"\nOrdered Gini coefficient: "
    f"{ordered_gini:.4f}"
)