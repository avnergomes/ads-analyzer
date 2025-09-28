# DiA v2.0 - Enhanced Show Analytics Dashboard

## 🎯 System Corrected According to Client Requirements

### ✅ Implemented Solutions

#### 1. **Health Dashboard with 5 Quick Visual Indicators**
- ✅ **Show Health**: Overall score based on sales %, ROAS, days remaining
- ✅ **Sales Velocity**: Sales speed vs daily target  
- ✅ **Ad Efficiency**: ROAS with color coding
- ✅ **Funnel Health**: Conversion efficiency between funnels
- ✅ **Revenue Pace**: Revenue progress vs total potential

#### 2. **Enhanced Show ID Mapping**
- ✅ Recognizes variations: `WDC_0927`, `WDC_0927_S2`, `WDC_0927_S3`, `S4`
- ✅ City name fallback: "Washington DC #2" → `WDC_0927_S2`
- ✅ Legacy patterns: "US-DC-Sales-2024 - Interest - 2nd - Support2"
- ✅ Priority system to avoid conflicts
- ✅ Dynamic token scoring using Google Sheet show catalog (city, date, time, sequence)
- ✅ Confidence scoring with full mapping and unmatched campaign audit tables

#### 3. **Improved Funnel Detection**
- ✅ **F1**: `F1`, `Fun1`, `LPViews`, `LPViews_F1`, `LPViews_Fun1`
- ✅ **F2**: `F2`, `Fun2`, `AddToCart`, `AddToCart_F2`
- ✅ **F3**: `F3`, `Fun3`, `CONV_AddtoCart`, `CONV_F3`
- ✅ **Legacy**: Campaigns with `Interest` or `Target`

#### 4. **Metrics from Online Ticket Sale Sheet (Column N)**
- ✅ **Ticket Cost** = Show Budget ÷ Total Tickets Sold
- ✅ **Core Stats**: Days to show, Capacity, Holds, Sold
- ✅ **Daily Sales Target** = Remaining ÷ Days left (vs 7-day average)
- ✅ **Funnel Efficiency**: Clicks/LP Views/Add to Cart/Conversions per ticket
- ✅ **Daily Sales CPA**: Total spend ÷ Total tickets sold
- ✅ **ROAS**: (Revenue from sales / Spend)

#### 5. **Google Sheets Integration**
- ✅ Direct loading from sheet: `https://docs.google.com/spreadsheets/d/1hVm1OALKQ244zuJBQV0SsQT08A2_JTDlPytUNULRofA/edit?gid=0#gid=0`
- ✅ Automatic sales data processing
- ✅ Caching for performance

#### 6. **Meta Ads Multi-file Support & Mapping Overview**
- ✅ Upload and consolidate up to **3 Meta Ads CSV exports** simultaneously
- ✅ Automatic source tracking and aggregation of spend/results per show
- ✅ Visual overview of mapped vs unmatched campaigns with confidence scores
- ✅ Sample unmatched listings to speed up manual review

## 🚀 Installation and Setup

### Prerequisites
```bash
pip install streamlit pandas plotly numpy requests
```

### File Structure
```
DiA/
├── app.py                   # Main dashboard (Streamlit Cloud compatible)
├── data_processor.py        # Data processor
├── campaign_mapping.csv     # Mapping rules
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Streamlit Community Cloud Setup

#### requirements.txt
```txt
streamlit==1.28.1
pandas==2.0.3
plotly==5.17.0
numpy==1.24.3
requests==2.31.0
```

#### .streamlit/config.toml (Optional)
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 200
```

### How to Deploy on Streamlit Community Cloud

1. **Fork/Clone this repository**
2. **Push to your GitHub account**
3. **Visit**: https://share.streamlit.io/
4. **Connect your GitHub repository**
5. **Set main file path**: `app.py`
6. **Deploy!**

### Local Development
```bash
# 1. Clone repository
git clone <your-repo-url>
cd DiA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run dashboard
streamlit run app.py
```

## 📊 How to Use the System

### 1. **Data Loading**

#### Option A: Google Sheets (Recommended)
1. Paste sheet URL in sidebar
2. Click "Load from Google Sheets"
3. Upload up to **three** Meta Ads CSV exports (e.g., main, device, time breakdown)

#### Option B: File Upload
1. Upload sales data CSV
2. Upload Meta Ads data CSV files (maximum of three per batch)

### 2. **Health Dashboard Interpretation**

#### 🏥 Health Indicators (Traffic Light System)
- 🟢 **Green**: Excellent performance, continue current strategy
- 🟡 **Yellow**: Attention needed, monitor closely
- 🔴 **Red**: Immediate action required

#### 📊 Indicator Metrics

**Show Health (0-100%)**
- 70-100%: 🟢 Healthy
- 40-70%: 🟡 Caution
- 0-40%: 🔴 Critical

**Sales Velocity (%)**
- ≥100%: 🟢 Target being met
- 70-99%: 🟡 Below target
- <70%: 🔴 Well below target

**ROAS (Return on Ad Spend)**
- ≥3.0x: 🟢 Excellent
- 1.5-3.0x: 🟡 Good
- <1.5x: 🔴 Inefficient

## 🗺️ Mapping System

### Show ID Patterns (Priority Order)

#### Priority 1: Standard Format
```regex
WDC_0927, WDC_0927_S2, WDC_0927_S3, etc.
TR_0410, TR_0410_S2, OTW_1006, etc.
```

#### Priority 2: City Name Variations
```regex
"Washington DC #2" → WDC_0927_S2
"Toronto 4 shows" → TR_0410_S4
"Bengaluru#3" → BLR_1126_S3
```

#### Priority 3: Legacy Patterns
```regex
"US-DC-Sales-2024 - Interest - 2nd" → WDC_0927_S2
"CA-Toronto-Sales-2024 - Target" → TR_0410
```

### Funnel Mapping

#### F1 (Landing Page Views)
- `F1`, `Fun1`, `Funnel1`
- `LPView`, `LP_View`, `LPV`
- `Landing Page View`, `Landing_Page_View`

#### F2 (Add to Cart)
- `F2`, `Fun2`, `Funnel2`
- `AddToCart`, `Add_To_Cart`, `ATC`
- `Cart`

#### F3 (Purchase/Conversion)
- `F3`, `Fun3`, `Funnel3`
- `Purchase`, `Sales`, `Checkout`
- `Conversion`, `Conv`, `Conv_AddtoCart`

## 🔧 Customization and Maintenance

### Adding New Shows
1. Edit mapping patterns in `app.py`
2. Add regex pattern for new show
3. Set priority (1=high, 3=low)

### Adjusting Health Metrics
In `app.py`, function `create_health_dashboard()`:
```python
# Adjust thresholds
if metrics['sold_percentage'] > 85: health_score += 30  # Adjust to 90
if metrics['roas'] > 3: health_score += 25              # Adjust to 2.5
```

### Customizing Colors and Layout
```python
# In create_health_dashboard()
health_color = "green" if health_score > 70 else "orange" if health_score > 40 else "red"

# Funnel colors
colors = {'F1': '#3498db', 'F2': '#e74c3c', 'F3': '#2ecc71'}
```

## 🐛 Troubleshooting

### Issue: Campaigns not being mapped
**Solution**: 
1. Check "Sample Campaign Names" section in dashboard
2. Adjust regex patterns in `parse_show_id_enhanced()`
3. Add new patterns with appropriate priority

### Issue: Google Sheets not loading
**Solution**:
1. Verify sheet is public or shared
2. Test manual CSV export URL
3. Check internet connection

### Issue: Health Dashboard not appearing
**Solution**:
1. Verify valid sales data exists
2. Confirm metrics calculated correctly
3. Check Streamlit logs for errors

### Issue: Streamlit Cloud deployment fails
**Solution**:
1. Verify `requirements.txt` has correct versions
2. Check file paths are relative
3. Ensure no local-only dependencies
4. Verify repository is public or properly connected

## 📈 Performance Monitoring

### Daily KPIs to Track
1. **Show Health Score**: Keep >70%
2. **Sales Velocity**: Keep ≥100% of target
3. **ROAS**: Keep ≥2.0x minimum
4. **Cost per Ticket**: Keep <30% of ATP
5. **Daily Sales vs Target**: Monitor daily gap

### Recommended Alerts
- 🔴 Health Score <40%: Immediate action
- 🔴 ROAS <1.5x: Optimize campaigns
- 🔴 Sales Velocity <70%: Review strategy
- 🟡 Cost per Ticket >30% ATP: Monitor CPMs

## 🌐 Streamlit Community Cloud Compatibility

### Features Optimized for Cloud Deployment

#### Memory Optimization
```python
@st.cache_data(ttl=3600)  # 1-hour cache
def load_google_sheet_data(sheet_url):
    # Optimized loading with caching
    
@st.cache_data
def process_data(sales_df, meta_df):
    # Cached data processing
```

#### Error Handling
```python
try:
    # Data loading with fallback
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    return None
```

#### Resource Management
- Efficient data processing with chunking
- Memory cleanup after operations
- Optimized chart rendering
- Minimal external dependencies

### Cloud Deployment Checklist
- ✅ All file paths are relative
- ✅ Dependencies in requirements.txt
- ✅ No local file system dependencies
- ✅ Error handling for network issues
- ✅ Caching for performance
- ✅ Memory-efficient operations
- ✅ Mobile-responsive design

## 🤝 Support and Maintenance

### Technical Support Contact
- Campaign mapping issues
- Adding new shows or patterns
- Metrics and visualization customizations
- Integration with other data sources

### Version Control
- **v2.0**: Current system with health dashboard and enhanced mapping
- **v1.0**: Original basic system

### Contributing
1. Fork the repository
2. Create feature branch
3. Submit pull request
4. Include tests for new mapping patterns

## 📝 Data Format Requirements

### Sales Data CSV Format
```csv
Show ID,Show Date,Show Name,Capacity,Total Sold,Remaining,ATP,Report Date
WDC_0927,2025-09-27,27.Wash DC,1406,1379,4,76,2025-09-27
TR_0410_S2,2025-10-04,4.Toronto#2,4060,3760,0,100,2025-09-27
```

### Meta Ads Data CSV Format
```csv
ad_set_name,amount_spent_(usd),results,impressions,reporting_starts,result_indicator
WDC_0927_S2_F1_LPView,150.25,1250,25000,2025-09-26,landing_page_view
TR_0410_F2_AddToCart,89.50,85,12000,2025-09-26,add_to_cart
```

## 🔒 Security and Privacy

### Data Handling
- No data stored permanently on server
- Session-based data processing
- Google Sheets accessed read-only
- No sensitive data cached

### Privacy Compliance
- Data processed in memory only
- No personal information required
- Aggregated analytics only
- Compliant with standard privacy practices

---

**DiA v2.0** - Analytics system designed to provide actionable insights and fast decisions through intuitive visual indicators and comprehensive show performance metrics. Fully compatible with Streamlit Community Cloud for easy deployment and scaling.

## 🚀 Quick Start Links

- **🌐 Live Demo**: [Deploy to Streamlit Cloud](https://share.streamlit.io/)
- **📊 Sample Data**: Use the provided Google Sheets URL
- **🔧 GitHub Repository**: Fork and customize
- **📖 Documentation**: This README

*Built for modern show analytics with enterprise-grade reliability and cloud-native architecture.*
