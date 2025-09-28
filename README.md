# 🎯 Ads Performance Analyzer

This Streamlit app helps visualize and analyze ads performance from Facebook/Meta campaign exports.

---

## 🚀 Features

- 🧠 **Smart Show ID Parsing** – handles formats like `WDC_0927`, `S3`, `CA-Calgary-Traffic`
- 🧼 **Funnel Normalization** – auto-detects: clicks, LP views, add-to-cart, conversions
- 📊 **Visual Dashboard** – includes:
  - Total Spend / Conversions / CPA
  - ROAS by Show
  - CPA by Show
  - Funnel Breakdown Chart

---

## 📁 Folder Structure

```
rewrite_ads_analyzer/
├── app.py               # Streamlit frontend
├── data_loader.py       # Data loading and cleaning logic
├── parser.py            # Show & funnel normalization logic
├── visualizer.py        # Summary metrics and plots
├── requirements.txt     # Dependencies
```

---

## 🛠️ How to Run

1. 🔽 Download the folder
2. 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

3. ▶️ Start the app:

```bash
streamlit run app.py
```

4. 📂 Provide the path to your ads `.csv` folder when prompted (e.g., `/mnt/data/samples/samples`)

---

## 📦 Input Format

Supports campaign CSVs from Meta Ads Manager with columns like:
- `Campaign name`
- `Amount spent`
- `Link clicks`
- `Landing page views`
- `Adds to cart`
- `Results`

---

## 🧩 Client Requests Implemented
- Health-of-Show indicators
- Flexible show parsing
- Legacy funnel mapping
- Integrated with Google Sheet metrics

---

Built with ❤️ by CodeNinja 🥷