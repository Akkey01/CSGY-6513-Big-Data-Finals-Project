# Project: Urban Transit Reliability and Demand Forecasting
by Jay Daftari(jd5829@nyu.edu), Akshat Mishra (am15111@nyu.edu) and Nikita Gupta(ng3230@nyu.edu).

## 1. Project Overview
This repository contains code and notebooks for analyzing NYC MTA subway ridership, service alerts, accessibility, and forecasting future demand using big data techniques.

**Key Components:**
- **ridership-analysis.ipynb:** Explores all processed data sources to derive insights into ridership trends, service alert impacts, and accessibility patterns. Includes time-series visualizations (rush-hour peaks, midday troughs), alert frequency histograms, and GeoPandas maps of ADA compliance.
- **big-data-project.ipynb:** Trains and evaluates an XGBoost regression model for daily ridership forecasting. Implements hyperparameter tuning (randomized search), model evaluation metrics (R², RMSE), and SHAP-based global and local feature importance analyses.

-  **Dashboard:**

---

## 2. High-Level Code Logic

## 2.1 Loading

- Leveraged PySpark for handling large datasets and efficient sampling.  
- Example: Sampled 1% of the "Hourly Ridership Data" for exploratory analysis to manage memory usage and improve computation speed.

## 2.2 Cleaning

- Removedduplicates to ensure data integrity.  
- Handledmissing values to avoid inconsistencies in analysis.  
- Standardized column names for uniformity across datasets.

## 2.3 Transformation

- Created ageosphere for mapping geographic locations using latitudes and longitudes.  
- Calculated and added a distance parameter derived from geosphere calculations to analyze ridership patterns relative to location.  
- Derived newfeatures, such as:  
  - "DayofWeek"forunderstanding weekly ridership trends.  
  - "HourofDay"for analyzing peak and off-peak usage.

## 2.4 Merging

- Integrated multiple datasets by utilizing key attributes:  
  - Station Complex ID for unifying station-level data.  
  - Transit Times for detailed ridership analysis.  
  - AgencyInformation for merging ridership datasets.

## 2.5 Exploratory Analysis

- Use Jupyter notebook to analyze hourly/daily ridership patterns.  
- Visualize service alert frequency and impact on ridership.  
- Map ADA-compliance gaps with GeoPandas.
  
## 2.6  MTA Ridership Explorer Dashboard
- **Dynamic Filtering & Metrics**: Select any station and date range to instantly update total and average daily ridership, with easy CSV export.  
- **Multi-Scale Time-Series Views**: Visualize ridership over hours, days, and weeks through interactive Plotly charts to uncover peak usage patterns.  
- **Geospatial Context**: See each station’s location on an embedded Folium map, aiding route planning and spatial analysis.  
- **Forecasting & Trend Analysis**: Generate 7-day ridership forecasts using Prophet (with uncertainty bands) and compare multiple stations’ daily trends side-by-side by borough.  
- **Revenue & Operational Insights**: Drill into revenue by payment method and fare class via a vibrant sunburst chart, and explore additional insights like ridership vs. distance to central and hour-of-day breakdowns.  

## 2.7 Model Training

- Load processed data into a Pandas DataFrame.  
- Split into training and test sets.  
- Train an XGBoost regressor with hyperparameter tuning (randomized search).  
- Evaluate R² and RMSE on test set.

## 2.8 Interpretability

- Compute SHAP values on a stratified sample.  
- Generate feature importance plots and local explanation examples.
---

## 3. Prerequisites
- **Python 3.8+**
- **Git**

## 4. Installation and Running the Project
1. Clone the repository:
2. Download the data:

- **MTA Daily Ridership Data (Post-2020)**  
  Access daily ridership data for various MTA services from March 2020 to January 2025.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Daily-Ridership-Data-2020-2025/vxuj-8kew/data)

- **MTA Hourly Ridership Data (Post-July 2020)**  
  Explore hourly ridership data for MTA services starting July 2020.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s/about_data)

- **MTA Subway Turnstile Usage Data**  
  Analyze subway turnstile usage data to understand station-level ridership patterns.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Subway-Turnstile-Usage-Data-2022/k7j9-jnct/about_data)

- **MTA Subway Stations**  
  Access information about all subway stations, including locations and service details.  
  👉 [View Dataset](https://catalog.data.gov/dataset/mta-subway-stations)

- **MTA Subway Stations and Complexes**  
  Explore data on subway stations and their interconnected complexes.  
  👉 [View Dataset](https://catalog.data.gov/dataset/mta-subway-stations-and-complexes)

- **MTA Service Alerts Data (Post-2020)**  
  Stay informed with service alerts data for MTA services from 2020 onward.  
  👉 [View Dataset](https://catalog.data.gov/dataset/mta-service-alerts-beginning-april-2020)

- **MTA Subway Major Incidents Data**  
  Review data on major subway incidents that affected service.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Subway-Major-Incidents-2020-2024/j6d2-s8m2/about_data)

- **MTA Subway Origin-Destination Data**  
  Analyze origin-destination data to understand subway travel patterns.  
  👉 [View Dataset](https://catalog.data.gov/dataset/mta-subway-origin-destination-ridership-estimate-beginning-2025)

- **MTA Subway and Bus Lost Time Data**  
  Examine data on lost time for subway and bus services.  
  👉 [View Dataset](https://catalog.data.gov/dataset/mta-subway-and-bus-lost-time-accidents-beginning-2021)

- **MTA Key Performance Indicators (2008–2021)**  
  Access key performance indicators for MTA services from 2008 to 2021.  
  👉 [View Dataset](https://catalog.data.gov/dataset/metropolitan-transportation-authority-mta-performance-indicators-per-agency-beginning-2008)

- **MTA Subway Customer Journey Metrics**  
  Explore customer journey metrics for the subway system.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-Beginning/4apg-4kt9)

- **MTA Customer Feedback Data (2014–2019)**  
  Review customer feedback data, including complaints and commendations.  
  👉 [View Dataset](https://data.ny.gov/Transportation/MTA-Customer-Feedback-Data-2014-2019/tppa-s6t6/data)

3. Run all of the ipynb file, by setting the correct path of data. (Since data is too huge to upload it on github.)
4. Download dependency using code `pip install streamlit pandas plotly streamlit-folium folium prophet` for dashboard_streamlit.py 
5. Run the dashboard_streamlit.py to see the live dashboard

---

## 5. File Structure
```
├── big-data-project.ipynb  # Analysis and Forecasting
├── ridership-analysis.ipynb # Analysis
├── dashboard_streamlit.py # Live Dashboard
└── README.md            # This file
```

---
