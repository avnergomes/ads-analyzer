
# 🎯 Meta Ads Funnel Analysis Dashboard

This interactive Streamlit app analyzes Meta Ads performance exports to evaluate marketing funnel effectiveness, show-level ROI, and ad performance by time, device, and placement.

---

## 🔍 Overview

The dashboard ingests up to **three Meta Ads CSV reports** and automatically classifies each ad using a powerful regex mapping system. It then provides:

- 📊 **Health of Show** KPIs
- 🔁 **Funnel Decay**
- 🕒 **Time-of-Day Performance**
- 📱 **Device & Placement Analysis**
- 🌍 **Show × Funnel Overview**

---

## 🚀 Try it on Streamlit Cloud

1. Clone or fork this repo.
2. Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Upload Meta Ads CSV files directly in the sidebar.

---

## 📁 Expected Inputs

The app supports up to **three optional file types**:

| File Type               | Description                                |
|------------------------|--------------------------------------------|
| `Days.csv`             | Daily performance summary                  |
| `Days + Time.csv`      | Time-of-day breakdown (hourly stats)       |
| `Days + Placement.csv` | Performance by placement & device          |

🧠 **No strict filenames required** — app auto-detects file type by inspecting column names.

---

## 🧠 Campaign Mapping

Ad rows are classified based on:
- **Show ID** (e.g. `Toronto`, `WDC_0927_S2`)
- **Funnel Stage** (e.g. `F1`, `Fun1`, `AddToCart`)
- **Legacy Tags** (e.g. `Interest`, `Target`)

Classification logic is defined in:
```
📄 campaign_mapping_fixed.csv
```

Each row in the file is a regex rule:

| regex_pattern           | mapping_type | mapping_value |
|------------------------|--------------|----------------|
| `Toronto|TR_\d+`       | `show`       | `Toronto`      |
| `F1|Fun1|Funnel ?1`    | `funnel`     | `F1`           |
| `Interest`             | `legacy`     | `Interest`     |

---

## 📦 Project Structure

```
.
├── app.py                       # Main Streamlit app
├── campaign_mapping_fixed.csv  # Regex rules for classification
├── requirements.txt            # Streamlit Cloud dependencies
└── README.md                   # You're here
```

---

## 🔧 Requirements

Python dependencies are listed in `requirements.txt`:

```
streamlit
pandas
plotly
altair
requests
openpyxl
```

---

## ✅ Client Features Summary

- ☑️ Visual KPIs: CPA, ROAS, Spend, Clicks/Ticket
- ☑️ Regex-based funnel/show classification
- ☑️ Handles legacy/alternate naming
- ☑️ Works with any export naming structure
- ☑️ Streamlit Cloud compatible

---

## 📬 Support

For technical questions or contributions, please open an issue or pull request.

---

> _“Would be amazing to see 3–5 quick visual graphs at the top when a parsed show is selected…”_  
> — **Client Brief**

Delivered ✅
