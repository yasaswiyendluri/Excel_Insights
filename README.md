
---

# Excel Insights

**Automated Data Analysis & Visualization Tool**

## Overview

Excel Insights is a Django web app that lets you upload Excel/CSV files and automatically:

* Clean data
* Analyze features (ANOVA, PCA)
* Generate insights
* Create visualizations and reports

No coding required.

---

## Requirements

**Python:** 3.10+

**Libraries:**

* Django
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* openpyxl
* reportlab

**System:**

* 4GB RAM minimum
* 500MB storage
* Windows / Linux / macOS

---

## Setup

```bash
# 1. Extract project
cd Excel_Insights

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup database
python manage.py migrate

# 6. Run server
python manage.py runserver
```

Open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## Project Structure (Simple)

```
core/
 ├── views.py          # Main logic (upload, analysis)
 ├── services/
 │    └── summary.py   # Data summaries
 ├── templates/        # UI (HTML)
 ├── static/           # CSS files
```

---

## How to Use

### 1. Upload Data

* Upload `.xlsx`, `.xls`, or `.csv`

### 2. Basic Summary

* View rows, columns, missing values
* See distributions and data quality

### 3. Data Cleaning

* Handle missing values (mean/median/mode)
* Remove duplicates
* Encode and scale data

### 4. Feature Selection (ANOVA)

* Choose target column
* Finds important features (p < 0.05)

### 5. PCA Analysis

* Reduce dimensions
* View variance and scatter plot

### 6. Auto Insights

* Detect:

  * correlations
  * skewness
  * missing data issues

### 7. Visualization

* Histograms, Scatter plots
* Heatmap
* Download PDF report

---

## Supported Formats

**Input:**

* Excel (.xlsx, .xls)
* CSV

**Output:**

* CSV (cleaned data)
* PDF report
* Charts (in browser)

---

## Common Issues

* **Module not found** → install requirements
* **Port in use** → `python manage.py runserver 8001`
* **Upload fails** → check format and size (<100MB)
* **PCA error** → need at least 2 numeric columns

---

## Features Used

* Descriptive statistics
* Correlation analysis
* ANOVA (feature importance)
* PCA (dimensionality reduction)
* Skewness detection
* Data cleaning (missing values, encoding, scaling)

---

## Authentication

No login required.

---

## Version

* v1.0 (2026)
* Django 4.x
* Python 3.10+

---

