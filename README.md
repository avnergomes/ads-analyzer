# Ads Analyzer

## Overview
Ads Analyzer is a Streamlit-based application designed to integrate **sales data from Google Sheets** with **Meta Ads performance data** uploaded by the user.  
It provides a unified dashboard to monitor KPIs, analyze trends, and visualize funnel performance.

## Features
- Load sales data automatically from Google Sheets.
- Upload Meta Ads performance files (CSV/XLSX).
- Automatic normalization of column names across different file formats.
- KPI computation (Impressions, Clicks, Spend, Conversions, CTR, CPC, CPA, ROAS).
- Interactive visualizations using Plotly (time series and funnel charts).
- Ready for deployment on Streamlit Community Cloud.

## How to Run
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Requirements
See `requirements.txt` for Python dependencies.

---
Developed as **v_zero** release of Ads Analyzer.
