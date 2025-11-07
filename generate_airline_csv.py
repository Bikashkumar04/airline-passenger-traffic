"""
Generate Airline Passenger Traffic CSV
Run this file to create airline-passenger-traffic1.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates from Jan 2012 to Oct 2024 (154 months)
start_date = datetime(2012, 1, 1)
months = 154
dates = pd.date_range(start=start_date, periods=months, freq='MS')

# Base trend: steady growth over time
base = 300
trend = np.linspace(0, 400, months)

# Seasonal pattern (peaks in summer, dips in winter)
seasonal = 100 * np.sin(np.linspace(0, months/6 * np.pi, months))

# Year-over-year growth with some variations
yearly_growth = np.array([1.0 + (i // 12) * 0.08 for i in range(months)])

# COVID-19 impact (sharp drop in 2020, gradual recovery)
covid_impact = np.ones(months)
for i, date in enumerate(dates):
    if datetime(2020, 3, 1) <= date <= datetime(2020, 12, 31):
        # Sharp drop during 2020
        covid_impact[i] = 0.3 + (i - 99) * 0.05  # Gradual recovery within 2020
    elif datetime(2021, 1, 1) <= date <= datetime(2021, 12, 31):
        # Recovery in 2021
        covid_impact[i] = 0.65 + (i - 108) * 0.02
    elif datetime(2022, 1, 1) <= date <= datetime(2022, 6, 30):
        # Continued recovery
        covid_impact[i] = 0.85 + (i - 120) * 0.02

# Random noise
noise = np.random.normal(0, 20, months)

# Combine all components
passengers = (base + trend + seasonal) * yearly_growth * covid_impact + noise

# Ensure no negative values
passengers = np.maximum(passengers, 50)

# Round to integers
passengers = passengers.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Month': dates,
    'Passengers': passengers
})

# Format the Month column as YYYY-MM-DD
df['Month'] = df['Month'].dt.strftime('%Y-%m-%d')

# Save to CSV
csv_filename = 'airline-passenger-traffic1.csv'
df.to_csv(csv_filename, index=False)

print("=" * 60)
print("✅ CSV FILE GENERATED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Dataset Summary:")
print(f"   • Total months: {len(df)}")
print(f"   • Date range: {df['Month'].iloc[0]} to {df['Month'].iloc[-1]}")
print(f"   • Min passengers: {df['Passengers'].min():,}")
print(f"   • Max passengers: {df['Passengers'].max():,}")
print(f"   • Average passengers: {df['Passengers'].mean():,.0f}")
import os
print(f"\n📁 File saved as: {csv_filename}")
print(f"   Location: {os.path.abspath(csv_filename)}")
print("\n" + "=" * 60)
print("🔍 PREVIEW - First 10 rows:")
print("=" * 60)
print(df.head(10).to_string(index=False))
print("\n" + "=" * 60)
print("🔍 PREVIEW - Last 10 rows:")
print("=" * 60)
print(df.tail(10).to_string(index=False))
print("\n" + "=" * 60)
print("🚀 NEXT STEPS:")
print("=" * 60)
print("1. Run your dashboard:")
print("   streamlit run dashboard3.py")
print("\n2. The dashboard will automatically load this CSV file!")
print("=" * 60)