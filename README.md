# Excel Insights

Automated data analysis platform that transforms raw datasets into actionable insights. Upload CSV/Excel → Get cleaned data, feature rankings, dimensionality analysis, and a PDF report.

**[Live Demo](https://excel-insights.onrender.com)** · **[Repository](https://github.com/yasaswiyendluri/Excel_Insights)**

---

## What It Does

```
Upload Data → Clean & Analyze → Get Insights + Report
```

- **Data Cleaning:** Handles missing values, duplicates, categorical encoding
- **Feature Selection (ANOVA):** Ranks features by statistical significance (p < 0.05)
- **Dimensionality Reduction (PCA):** Projects to 2D while reporting variance retained
- **Auto-Insights:** Detects correlations, skewness, data quality issues
- **Visualizations:** Histograms, scatter plots, correlation heatmaps, PCA projection
- **PDF Export:** Single-click report with statistics + charts

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | Django 6.0+ | File handling, session management, routing |
| Data Processing | Pandas 2.x + NumPy | Efficient cleaning, type-aware operations |
| ML Analysis | scikit-learn | ANOVA (f_classif), PCA, StandardScaler |
| Visualization | Matplotlib + Seaborn | Fast rendering, ReportLab-compatible |
| PDF Generation | ReportLab | Pure Python, embeds images cleanly |

---

## Architecture

```
User Upload
    ↓
File Validation & Read (CSV/XLSX)
    ↓
Data Cleaning
├─ Fill missing (mean/median/mode)
├─ Drop duplicates
├─ LabelEncode categorical features
└─ Heuristic-based scaling (skips encoded categorical columns)
    ↓
ANOVA Feature Selection
├─ One-hot encode remaining categorical features (pd.get_dummies)
├─ StandardScale numeric features
├─ SelectKBest(f_classif) → Rank by F-score
└─ Filter: p-value < 0.05 (statistical significance threshold)
    ↓
PCA Analysis
├─ Project to 2D
└─ Report variance explained
    ↓
Visualization Engine
├─ Histogram (distribution)
├─ Scatter (feature relationship)
├─ Heatmap (correlations)
└─ PCA scatter (2D projection)
    ↓
Insights + PDF Report Export
```

---

## Quick Start

```bash
# Clone & setup
git clone https://github.com/yasaswiyendluri/Excel_Insights.git
cd Excel_Insights
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run
python manage.py migrate
python manage.py runserver
# → http://127.0.0.1:8000
```

---

## Performance

| Metric | Spec |
|--------|------|
| File Size | 5–50 MB (typical) |
| Rows | 100–100k (tested; performance tested on ~50k rows) |
| Processing Time | 2–30 seconds (varies by row count and numeric column count) |
| Memory Usage | ~4x file size (in-memory DataFrame storage) |

**Constraints:** Session-based storage using SQLite backend; limited to single-user analysis per session. For production scale, migrate to temporary file persistence + async background job queue.

---

## Key Technical Decisions

### 1. Smart Categorical Detection
Prevents scaling of integer-like categorical columns (e.g., gender: 0/1). Detects via cardinality heuristic (`nunique < 20`).

### 2. ANOVA over Correlation
Handles both numeric and categorical predictors via one-hot encoding (`pd.get_dummies`) before feature selection. F-statistic from `f_classif` captures feature-target association strength; p-value < 0.05 determines statistical significance threshold.

### 3. PCA with Variance Reporting
Projects to 2D for visualization while reporting % variance retained (e.g., "2 PCs capture 87% of variance"). Honest about information loss.

### 4. Mixed Data Handling
- Handles categorical variables via LabelEncoder (cleaning) and pd.get_dummies (ANOVA)
- Fills numeric missing values via mean/median/mode selection
- Smart scaling: detects and skips integer-like categorical columns to preserve semantics
- Single unified pipeline; no manual type configuration needed

---

## How to Use

1. **Upload** — Select `.csv`, `.xlsx`, or `.xls`
2. **Review Summary** — Data types, missing %, unique values per column
3. **Clean Data** — Select missing value strategy (mean/median/mode); LabelEncode categorical features; remove duplicates; download cleaned CSV
4. **Feature Selection** — Pick target column; system one-hot encodes categorical variables, applies ANOVA (f_classif), ranks by F-score, filters significant features (p < 0.05)
5. **PCA Analysis** — View 2D principal component projection + variance explained ratio
6. **View Insights** — Auto-detected correlations, skewness anomalies, missing data flags
7. **Export PDF** — Download formatted report with statistics + 4 visualizations (histogram, scatter, heatmap, PCA)

---

## Project Structure

```
Excel_Insights/
├── core/
│   ├── views.py              # Pipeline orchestration (upload → analysis)
│   ├── services/
│   │   └── summary.py        # Insight generation logic
│   ├── templates/
│   │   └── home.html         # UI & upload form
│   └── static/
│       └── members/
│           └── style.css     # Frontend styling
│
├── portal/
│   ├── settings.py           # Django config (DEBUG=False in prod)
│   ├── wsgi.py              # Gunicorn entry point
│   └── urls.py
│
├── requirements.txt          # Python dependencies
├── render.yaml              # Render deployment config
└── manage.py                # Django CLI
```

---

## Deployment

Deployed on **Render** (free tier).

**Current Setup:**
- Backend: SQLite + session-based storage
- Static files: WhiteNoise middleware
- Server: Gunicorn WSGI application
- Environment config: `DEBUG=False`, `SECRET_KEY` from env variables
- HTTPS: `SECURE_PROXY_SSL_HEADER` enabled for SSL termination

**To Scale Beyond Free Tier:**
- Upgrade Render plan (4 GB RAM, 2x concurrency)
- Replace session storage with temporary file persistence
- Migrate database to PostgreSQL (for concurrent user sessions)
- Add background task queue (Celery + Redis) for long-running analyses
- Implement result caching layer for repeated dataset uploads

---

## Features Used

- Descriptive statistics (mean, median, std, min, max, quantiles)
- Correlation matrix analysis
- ANOVA for feature importance ranking
- PCA for dimensionality reduction & variance analysis
- Skewness detection (non-normal distributions)
- Data cleaning (missing values, duplicates, encoding, scaling)
- Mixed data type handling (numeric, categorical, dates)
- Auto-insight generation (anomalies, relationships, quality flags)

---

## Common Issues

| Problem | Solution |
|---------|----------|
| **"Module not found"** | `pip install -r requirements.txt` |
| **Port 8000 in use** | `python manage.py runserver 8001` |
| **Upload fails** | Check file size (<100 MB) and format (.csv/.xlsx) |
| **PCA error** | Need ≥2 numeric columns after cleaning |
| **Memory error on large file** | Split dataset or upgrade RAM |

---

## Dependencies

```
Django==6.0.3
pandas==2.0.x
numpy==1.24.x
scikit-learn==1.3.x
matplotlib==3.7.x
seaborn==0.12.x
openpyxl==3.10.x
reportlab==4.0.x
gunicorn==21.x
whitenoise==6.5.x
```

---

## Contributing

Contributions are welcome via GitHub issues and pull requests.
- Report bugs or suggest enhancements in the repository issues.
- Send PRs for documentation improvements, bug fixes, or new analysis features.


Issues and PRs welcome. Some ideas for improvement:

- **Dask/Polars Integration** — Out-of-core processing for >100 MB datasets
- **User Authentication** — Save and share analyses; user history tracking
- **Background Jobs** — Celery + Redis for async processing with progress tracking
- **Advanced Statistics** — Kruskal-Wallis test, chi-square test, effect size reporting
- **Outlier Detection** — Isolation Forest, Z-score flagging, DBSCAN clustering
- **Interactive Dashboards** — Plotly/Streamlit-based exploratory UI
- **API Endpoint** — REST API for programmatic dataset analysis

---

**Questions?** Open an issue on [GitHub](https://github.com/yasaswiyendluri/Excel_Insights/issues) or check the [live demo](https://excel-insights.onrender.com).