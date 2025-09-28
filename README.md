# Ads Performance Analyzer for Shows

This is a Streamlit-based dashboard for mapping Meta Ads campaigns to live shows and analyzing their performance based on ticket sales from a Google Sheet.

## Features

- Load all campaign CSVs from the `/samples` folder
- Auto-map campaigns to shows using fuzzy logic and regex
- Pull ticket sales and show metadata from a public Google Sheet
- Display health metrics (CPA, ROAS, Ticket Cost, Days to Show)
- Analyze funnel efficiency (Clicks, LPViews, AddToCart per Ticket)

## How to Use

1. Place your Meta Ads CSV files in the `samples/` folder.
2. Run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

3. The sales sheet is auto-loaded from a public Google Sheet:
   https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA

## Project Structure

```
.
├── app.py
├── data_loader.py
├── mapper.py
├── metrics.py
├── visualizer.py
├── requirements.txt
├── README.md
└── samples/
```
