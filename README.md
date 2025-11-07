# ✈️ Airline Passenger Time-Series Dashboard

A comprehensive Streamlit dashboard for analyzing and forecasting airline passenger traffic using ARIMA time series models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Understanding the Dashboard](#understanding-the-dashboard)
- [Real-World Applications](#real-world-applications)
- [Troubleshooting](#troubleshooting)
- [Technologies Used](#technologies-used)

---

## 🎯 Overview

This dashboard provides powerful time series analysis and forecasting capabilities for airline passenger data. It uses statistical models (ARIMA) to predict future passenger traffic based on historical patterns, trends, and seasonality.

**Perfect for:**
- Airlines planning flight schedules
- Airports managing resources
- Business analysts forecasting demand
- Students learning time series analysis
- Data scientists prototyping forecasting solutions

---

## ✨ Features

### 📊 Data Analysis
- **Interactive Time Series Visualization** - Plot historical passenger data with zoom and pan capabilities
- **Seasonal Decomposition** - Break down data into trend, seasonal, and residual components
- **Automatic Pattern Detection** - Identify yearly, monthly, and weekly patterns

### 🔮 Forecasting
- **Auto-ARIMA** - Automatically selects optimal model parameters
- **Manual ARIMA Configuration** - Fine-tune (p,d,q) parameters for advanced users
- **Confidence Intervals** - 80% prediction intervals for forecast uncertainty
- **Multi-horizon Forecasts** - Predict 3-36 months into the future

### 📈 Performance Metrics
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Square Error) - Penalizes larger errors
- **MAPE** (Mean Absolute Percentage Error) - Error as percentage
- **Train/Test Split** - Adjustable training data percentage (50-95%)

### 💾 Export & Download
- Download forecast results as CSV
- Save predictions for external analysis
- Share forecasts with stakeholders

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download
```bash
# Navigate to your project directory
git clone https://github.com/Bikashkumar04/airline-passenger-traffic.git
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install streamlit pandas numpy matplotlib seaborn plotly statsmodels pmdarima scikit-learn
```

### Alternative: Install from requirements.txt
If you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## 📖 Usage

### Step 1: Generate Sample Data
```bash
python generate_airline_csv.py
```
This creates `airline-passenger-traffic1.csv` with 154 months of realistic passenger data (2012-2024).

### Step 2: Run the Dashboard
```bash
streamlit run dashboard3.py
```

### Step 3: Access the Dashboard
Your default web browser will automatically open to:
```
http://localhost:8501
```

### Step 4: Configure Settings
Use the **left sidebar** to:
1. Upload your own CSV file (or use the generated demo file)
2. Set date and value column names
3. Choose forecast horizon (3-36 months)
4. Adjust train/test split ratio
5. Enable Auto-ARIMA or set manual parameters

### Step 5: Analyze Results
- View KPIs at the top (observations, last month passengers, YoY change)
- Examine the time series plot for trends
- Review seasonal decomposition charts
- Check actual vs. forecast comparison
- Download forecast CSV for further analysis

---

## 📁 Project Structure

```
practical-2/
│
├── dashboard3.py                      # Main Streamlit dashboard application
├── generate_airline_csv.py            # CSV data generator script
├── airline-passenger-traffic1.csv     # Generated sample data (auto-created)
├── README.md                          # This file
├── .venv/                             # Virtual environment (optional)
└── requirements.txt                   # Python dependencies (optional)
```

---

## 📊 Data Format

### CSV File Requirements
Your CSV file must contain at least **2 columns**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| **Month** (or custom name) | Date | Monthly timestamps | 2020-01-01 |
| **Passengers** (or custom name) | Integer | Passenger count | 450 |

### Example CSV Format
```csv
Month,Passengers
2012-01-01,312
2012-02-01,325
2012-03-01,358
2012-04-01,401
2012-05-01,456
...
```

### Best Practices
- **Minimum Data**: At least 24 months (2 years) for reliable forecasting
- **Recommended**: 36-60 months (3-5 years) for capturing seasonality
- **Frequency**: Monthly data works best
- **Consistency**: No missing months (or fill gaps appropriately)
- **Format**: Use ISO date format (YYYY-MM-DD)

---

## 🎓 Understanding the Dashboard

### Key Performance Indicators (KPIs)

#### 1. **Observations**
- Total number of data points (months)
- Date range of your dataset
- Example: `144 observations from 2012-01 → 2023-12`

#### 2. **Last Month Passengers**
- Most recent passenger count
- Current traffic level
- Example: `622 passengers`

#### 3. **Year-over-Year (YoY) Change**
- Percentage change from same month last year
- Growth or decline indicator
- Example: `+15.3%` (15.3% growth)

### Visualizations Explained

#### 📈 **Time Series Plot**
- **X-axis**: Time (months)
- **Y-axis**: Passenger count
- **Pattern Analysis**:
  - Upward slope = Growth trend
  - Repeating peaks = Seasonality
  - Flat line = Stable demand

#### 🧩 **Decomposition Chart**
Breaks down your data into components:

1. **Observed**: Original data
2. **Trend**: Long-term direction (smoothed)
3. **Seasonal**: Regular patterns (yearly cycles)
4. **Residual**: Random noise (unexplained variation)

**Use Case**: Understanding what drives your data changes

#### 🔮 **Actual vs. Forecast**
- **Solid Blue Line**: Historical actual data
- **Dashed Orange Line**: Model predictions
- **Shaded Area**: Confidence interval (80% certainty)
- **Interpretation**: 
  - Predictions inside shaded area = Good forecast
  - Predictions outside = Model needs adjustment

### Error Metrics Guide

| Metric | Formula | Interpretation | Good Value |
|--------|---------|----------------|------------|
| **MAE** | Mean Absolute Error | Average error in passengers | < 50 |
| **RMSE** | Root Mean Square Error | Penalizes large errors more | < 75 |
| **MAPE** | Mean Absolute % Error | Error as percentage | < 10% |

**Example Interpretation**:
- MAE = 25 → On average, predictions are off by ±25 passengers
- MAPE = 5% → Predictions are typically within 5% of actual values

### ARIMA Parameters (Advanced)

#### What is ARIMA?
**A**uto**R**egressive **I**ntegrated **M**oving **A**verage

#### Parameters (p, d, q):
- **p** (AR order): Number of past values used
- **d** (Differencing): Times series is differenced to make it stationary
- **q** (MA order): Number of past forecast errors used

#### Auto-ARIMA (Recommended)
- Automatically tests combinations
- Selects optimal (p,d,q) parameters
- Uses AIC/BIC criteria for selection

#### Manual Configuration
For advanced users who want control:
- Example: `(1,1,1)` = Basic ARIMA model
- Example: `(2,1,2)` = More complex patterns
- Higher values = More complex model (risk of overfitting)

---

## 🌍 Real-World Applications

### 1. **Aviation Industry** ✈️
- **Use Case**: Route planning and capacity optimization
- **Benefits**: 
  - Schedule flights based on demand forecasts
  - Allocate aircraft efficiently
  - Plan crew schedules
- **Companies**: Delta, United, Emirates

### 2. **Airport Operations** 🛫
- **Use Case**: Resource allocation and staffing
- **Benefits**:
  - Optimize security checkpoint staffing
  - Manage gate assignments
  - Plan parking and ground services
- **Companies**: Heathrow, JFK, Dubai International

### 3. **Tourism & Hospitality** 🏨
- **Use Case**: Dynamic pricing and inventory management
- **Benefits**:
  - Adjust hotel rates based on demand
  - Optimize occupancy rates
  - Plan marketing campaigns
- **Companies**: Marriott, Airbnb, Booking.com

### 4. **Retail & E-commerce** 🛒
- **Use Case**: Sales forecasting and inventory planning
- **Benefits**:
  - Predict product demand
  - Optimize stock levels
  - Reduce storage costs
- **Companies**: Amazon, Walmart, Target

### 5. **Transportation Services** 🚆
- **Use Case**: Schedule optimization
- **Benefits**:
  - Plan train/bus frequencies
  - Optimize routes
  - Reduce operational costs
- **Companies**: Uber, Lyft, Railways

### 6. **Healthcare** 🏥
- **Use Case**: Patient admission forecasting
- **Benefits**:
  - Staff hospitals appropriately
  - Manage bed capacity
  - Reduce wait times
- **Organizations**: Hospitals, Clinics, Emergency Services

### 7. **Energy & Utilities** ⚡
- **Use Case**: Demand forecasting
- **Benefits**:
  - Predict electricity consumption
  - Optimize power generation
  - Prevent blackouts
- **Companies**: Power Grid Operators, Utilities

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Module Not Found Error**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**:
```bash
pip install streamlit pandas numpy plotly statsmodels pmdarima
```

#### 2. **NumPy Binary Incompatibility**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```
**Solution**:
```bash
pip uninstall pmdarima statsmodels numpy -y
pip install numpy scipy scikit-learn statsmodels pmdarima
```

#### 3. **CSV File Not Found**
```
No file uploaded and demo file `airline-passenger-traffic1.csv` not found
```
**Solution**:
```bash
# Generate the demo file
python generate_airline_csv.py
```

#### 4. **Column Name Mismatch**
```
CSV must contain 'Month' and 'Passengers' columns
```
**Solution**:
- Check your CSV column names
- Update sidebar settings to match your column names
- Example: If your columns are `Date` and `Traffic`, enter those names

#### 5. **Date Parsing Errors**
```
Unable to parse date column
```
**Solution**:
- Ensure dates are in format: `YYYY-MM-DD`
- Example: `2020-01-01` (correct) vs `01/01/2020` (incorrect)

#### 6. **Streamlit Won't Start**
```
Port 8501 is already in use
```
**Solution**:
```bash
# Use a different port
streamlit run dashboard3.py --server.port 8502
```

#### 7. **Forecast Fails**
```
ARIMA model did not converge
```
**Solution**:
- Try Auto-ARIMA instead of manual parameters
- Ensure you have at least 24 months of data
- Check for missing values in your dataset

---

## 💻 Technologies Used

### Core Framework
- **Streamlit** (1.28+) - Web application framework

### Data Processing
- **Pandas** (2.0+) - Data manipulation and analysis
- **NumPy** (1.24+) - Numerical computing

### Visualization
- **Plotly** (5.17+) - Interactive charts
- **Matplotlib** (3.7+) - Static plots
- **Seaborn** (0.12+) - Statistical visualizations

### Time Series Modeling
- **Statsmodels** (0.14+) - Statistical models and tests
- **pmdarima** (2.0+) - Auto-ARIMA implementation
- **scikit-learn** (1.3+) - Machine learning utilities

---

## 📝 Example Workflow

### Scenario: Airline Planning for Next Year

1. **Data Collection**
   - Export last 5 years of monthly passenger data
   - Format as CSV with Month and Passengers columns

2. **Run Generator** (if using demo)
   ```bash
   python generate_airline_csv.py
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run dashboard3.py
   ```

4. **Configure Settings**
   - Upload your CSV or use demo data
   - Set forecast horizon to 12 months
   - Use Auto-ARIMA for best results
   - Set train size to 80%

5. **Analyze Results**
   - Review seasonal patterns (summer peaks?)
   - Check trend direction (growing or declining?)
   - Examine forecast accuracy (MAPE < 10%?)
   - Note confidence intervals (how certain?)

6. **Make Decisions**
   - High summer forecast → Add flights in June-August
   - Declining trend → Investigate causes
   - High uncertainty → Plan buffer capacity

7. **Export Forecasts**
   - Download CSV with predictions
   - Share with operations team
   - Update quarterly as new data arrives

---

## 🎯 Tips for Best Results

### Data Quality
✅ **Do**:
- Use consistent monthly data
- Include at least 2-3 years of history
- Fill or interpolate missing values
- Remove obvious data errors

❌ **Don't**:
- Use weekly or daily data (convert to monthly)
- Include incomplete months
- Have large data gaps
- Mix different metrics

### Model Configuration
✅ **Do**:
- Start with Auto-ARIMA
- Use 70-80% training data
- Set reasonable forecast horizons (6-12 months)
- Review error metrics

❌ **Don't**:
- Use manual ARIMA without understanding
- Forecast too far (>24 months unreliable)
- Use 95%+ training data (no testing)
- Ignore poor error metrics

### Interpretation
✅ **Do**:
- Consider confidence intervals
- Compare to domain knowledge
- Update forecasts regularly
- Document assumptions

❌ **Don't**:
- Trust forecasts blindly
- Ignore business context
- Use outdated forecasts
- Over-interpret noise

---

## 📚 Additional Resources

### Learning Time Series
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)
- [ARIMA Models Explained](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

### Streamlit Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)

### Data Science
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)
- [Python for Data Analysis](https://wesmckinney.com/book/)

---

## 📞 Support

### Getting Help
- Check the [Troubleshooting](#troubleshooting) section
- Review error messages carefully
- Ensure all dependencies are installed
- Verify CSV file format

### Common Questions

**Q: Can I use weekly or daily data?**
A: The dashboard is optimized for monthly data. For other frequencies, modify the `freq` parameter in the code.

**Q: How accurate are the forecasts?**
A: Accuracy depends on data quality and patterns. MAPE < 10% is generally good for business decisions.

**Q: Can I forecast multiple years ahead?**
A: Yes, but forecast uncertainty increases significantly beyond 12-18 months.

**Q: What if my data has missing months?**
A: The dashboard will interpolate, but it's better to fill gaps before uploading.

**Q: Can I compare multiple scenarios?**
A: Currently, the dashboard shows one forecast at a time. You can download results and compare externally.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Time series analysis powered by [Statsmodels](https://www.statsmodels.org/)
- Auto-ARIMA implementation from [pmdarima](http://alkaline-ml.com/pmdarima/)
- Visualizations by [Plotly](https://plotly.com/)

---

## 📊 Version History

- **v1.0** (2024) - Initial release with Auto-ARIMA and decomposition
- Features: Time series plotting, seasonal decomposition, ARIMA forecasting, error metrics, CSV export

---

**Made with ❤️ for data-driven decision making**

*Last Updated: November 2025*