import pandas as pd
import joblib

# Load saved model
model = joblib.load("ev_battery_model.pkl")

print("\n===== EV Battery Health Predictor =====\n")

# Take user inputs
charge_cycles = int(input("Enter Charge Cycles: "))
fast_charging = float(input("Enter Fast Charging Frequency (%): "))
temperature = float(input("Enter Average Temperature (°C): "))
aggression = float(input("Enter Driving Aggression Index (0 to 1): "))

# Create dataframe with same feature names used during training
sample = pd.DataFrame({
    "Charge_Cycles": [charge_cycles],
    "Fast_Charging_Frequency_%": [fast_charging],
    "Avg_Temperature_C": [temperature],
    "Driving_Aggression_Index": [aggression]
})

# Predict degradation
degradation = model.predict(sample)[0]

# Calculate battery health
battery_health = 100 - degradation

# Verdict Logic
if battery_health >= 90:
    verdict = "🟢 Excellent"
    recommendation = "Battery is in excellent condition."
elif battery_health >= 80:
    verdict = "🟢 Healthy"
    recommendation = "Battery is healthy. Continue normal usage."
elif battery_health >= 70:
    verdict = "🟡 Monitor"
    recommendation = "Monitor battery health regularly."
elif battery_health >= 60:
    verdict = "🟠 Needs Attention"
    recommendation = "Reduce fast charging and avoid overheating."
else:
    verdict = "🔴 Replacement Recommended"
    recommendation = "Consider battery replacement soon."

# Risk Level
risk_score = degradation

if risk_score <= 20:
    risk = "Low Risk"
elif risk_score <= 40:
    risk = "Moderate Risk"
elif risk_score <= 60:
    risk = "High Risk"
else:
    risk = "Critical Risk"

# Display results
print("\n========== RESULT ==========")
print(f"Battery Degradation : {degradation:.2f}%")
print(f"Battery Health      : {battery_health:.2f}%")
print(f"Verdict             : {verdict}")
print(f"Risk Level          : {risk}")
print(f"Recommendation      : {recommendation}")
print("============================")