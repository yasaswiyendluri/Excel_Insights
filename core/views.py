from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io, base64
from io import BytesIO

# IMPORT NEW SUMMARY SERVICE
from core.services.summary import DataSummaryService, generate_quick_summary

# ============================================
# FILE READING FUNCTION
# ============================================
def read_file(file):
    filename = file.name.lower()
    if filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file, engine='openpyxl')
    elif filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format")
    
    df.columns = df.columns.astype(str).str.strip()
    return df

# ============================================
# MISSING VALUES INFO
# ============================================
def missing_vals(df):
    missing_info = []

    for col in df.columns:
        missing_count = df[col].isnull().sum()

        if missing_count > 0:
            col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

            missing_info.append({
                "name": col,
                "missing_count": int(missing_count),
                "type": col_type
            })

    return missing_info

# ============================================
# BASIC SUMMARY WITH NEW SUMMARY SERVICE
# ============================================
def basic_summary(df):
    """
    Generate basic summary using new DataSummaryService
    This replaces the old basic_summary function logic
    """
    summary = {}
    summary['rows'] = df.shape[0]
    summary['cols'] = df.shape[1]
    summary['columns'] = list(df.columns)
    summary['dtypes'] = df.dtypes.astype(str).to_frame("Data Type").to_html(classes="table")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        summary['missing'] = missing.to_frame("No of missing values").to_html(classes="table")
    else:
        summary['missing'] = "<p>No missing values!</p>"

    summary['stats'] = df.describe(include='all').to_html(classes="table")
    summary['tables'] = df.head().to_html(classes="table")

    # ✅ NEW: Use DataSummaryService for comprehensive summary
    try:
        summary_html = generate_quick_summary(df)
        summary['data_summary'] = summary_html
    except Exception as e:
        summary['data_summary'] = f"<p>Could not generate summary: {str(e)}</p>"

    return summary

# ============================================
# DATA CLEANING FUNCTION
# ============================================
def data_cleaning(request, df):
    df_processed = df.copy()
    steps = []
    
    for col in df_processed.columns[df_processed.isnull().any()]:
        action = request.POST.get(f"missing_{col}")
        if not action:
            continue
        
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if action == "mean":
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif action == "median":
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            elif action == "mode":
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            elif action == "unknown":
                df_processed[col].fillna("Unknown", inplace=True)
            elif action == "drop_row":
                df_processed.dropna(subset=[col], inplace=True)
            elif action == "drop_col":
                df_processed.drop(columns=[col], inplace=True)
        else:
            if action == "mode":
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            elif action == "unknown":
                df_processed[col].fillna("Unknown", inplace=True)
            elif action == "drop_row":
                df_processed.dropna(subset=[col], inplace=True)
            elif action == "drop_col":
                df_processed.drop(columns=[col], inplace=True)
        
        steps.append(f"{col}: {action}")
    
    # Remove duplicates
    if request.POST.get("remove_duplicates"):
        before = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        after = len(df_processed)
        steps.append(f"Removed {before - after} duplicate rows")
    
    # Encoding
    if request.POST.get("encode"):
        for col in df_processed.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        steps.append("Categorical encoding applied")
    
    # Scaling
    if request.POST.get("scale"):
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        steps.append("Scaling applied")
    
    return df_processed, steps

# ============================================
# ANOVA FEATURE SELECTION
# ============================================
def run_anova(df, target_col):
    df = df.copy()
    df = df.dropna(axis=0)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    F, p = f_classif(X, y)
    result_df = pd.DataFrame({
        "Feature": X.columns,
        "F-Score": F,
        "P-value": p
    })
    result_df = result_df.sort_values(by="F-Score", ascending=False)
    top_features = result_df[result_df["P-value"] < 0.05]["Feature"].tolist()
    
    return result_df, top_features

# ============================================
# PCA ANALYSIS
# ============================================
def run_pca(df, n_components=2):
    df = df.copy()
    df = df.select_dtypes(include=['number'])
    df = df.dropna()

    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for PCA")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=cols)

    variance = pca.explained_variance_ratio_
    variance_text = ", ".join([f"PC{i+1}: {round(v*100, 2)}%" for i, v in enumerate(variance)])

    plot_url = None
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
        plt.xlabel(f"PC1 ({round(variance[0]*100, 2)}%)")
        plt.ylabel(f"PC2 ({round(variance[1]*100, 2)}%)")
        plt.title("PCA Scatter Plot")
        plt.grid(True, alpha=0.3)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()

    return pca_df, variance_text, plot_url

# ============================================
# AUTO INSIGHTS FUNCTION
# ============================================
def generate_insights(df):
    insights = []

    rows, cols = df.shape
    insights.append(f"Dataset has {rows} rows and {cols} columns")

    num_cols = df.select_dtypes(include=['number']).shape[1]
    cat_cols = df.select_dtypes(include=['object']).shape[1]
    insights.append(f"{num_cols} numeric columns and {cat_cols} categorical columns")

    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    missing_percent = (total_missing / total_cells) * 100

    if total_missing > 0:
        insights.append(f"Dataset has {round(missing_percent, 2)}% missing values")
        for col in df.columns:
            miss = df[col].isnull().sum()
            if miss > 0:
                percent = (miss / len(df)) * 100
                if percent > 30:
                    insights.append(f"⚠️ Column '{col}' has high missing values ({round(percent, 2)}%)")
    else:
        insights.append("No missing values detected")

    corr = df.corr(numeric_only=True)
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.7:
                insights.append(f"'{corr.columns[i]}' and '{corr.columns[j]}' are highly correlated ({round(val, 2)})")

    for col in df.select_dtypes(include=['number']).columns:
        skew = df[col].skew()
        if skew > 1:
            insights.append(f"'{col}' is highly right-skewed")
        elif skew < -1:
            insights.append(f"'{col}' is highly left-skewed")

    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = df[col].nunique()
        if unique_vals == len(df):
            insights.append(f"'{col}' has all unique values (likely an ID column)")
        elif unique_vals < 5:
            insights.append(f"'{col}' has low variety ({unique_vals} unique values)")

    return insights

# ============================================
# OPTIMIZED VISUALIZATIONS
# ============================================
def generate_visualizations(df):
    plots = {}

    # Get top 5 numeric columns by variance
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 5:
        num_cols = df[num_cols].var().nlargest(5).index.tolist()
    num_cols = num_cols[:5]

    # HISTOGRAMS
    histograms = []
    for col in num_cols:
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(df[col].dropna(), bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            plt.title(f"{col} - Distribution", fontsize=12, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3, axis='y')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            histograms.append(base64.b64encode(buffer.getvalue()).decode())
            buffer.close()
            plt.close()
        except Exception as e:
            print(f"Error creating histogram for {col}: {e}")
    
    plots['histograms'] = histograms

    # SCATTER PLOTS
    scatterplots = []
    if len(num_cols) >= 2:
        for i in range(min(3, len(num_cols)-1)):
            try:
                plt.figure(figsize=(8, 5))
                plt.scatter(df[num_cols[i]], df[num_cols[i+1]], alpha=0.6, color='darkgreen', s=50)
                plt.xlabel(num_cols[i], fontsize=11)
                plt.ylabel(num_cols[i+1], fontsize=11)
                plt.title(f"{num_cols[i]} vs {num_cols[i+1]}", fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                scatterplots.append(base64.b64encode(buffer.getvalue()).decode())
                buffer.close()
                plt.close()
            except Exception as e:
                print(f"Error creating scatter plot: {e}")
    
    plots['scatterplots'] = scatterplots

    # HEATMAP
    try:
        corr = df[num_cols].corr(numeric_only=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1,
            vmax=1
        )
        plt.title("Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        heatmap = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()

        plots['heatmap'] = heatmap
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        plots['heatmap'] = None

    return plots

# ============================================
# PDF REPORT GENERATION
# ============================================
def create_pdf_report(df, insights, plots):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Excel Insights Report", styles['Title']))
    elements.append(Spacer(1, 10))

    # Dataset info
    elements.append(Paragraph(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # Insights
    elements.append(Paragraph("Key Insights:", styles['Heading2']))
    for ins in insights[:10]:
        elements.append(Paragraph(f"• {ins}", styles['Normal']))
    elements.append(Spacer(1, 20))

    # Add images
    if 'histograms' in plots and plots['histograms']:
        elements.append(Paragraph("Histograms", styles['Heading2']))
        for img in plots['histograms'][:3]:
            try:
                img_data = base64.b64decode(img)
                img_buffer = BytesIO(img_data)
                elements.append(Image(img_buffer, width=500, height=300))
                elements.append(Spacer(1, 15))
            except:
                pass

    if 'heatmap' in plots and plots['heatmap']:
        elements.append(Paragraph("Correlation Heatmap", styles['Heading2']))
        try:
            img_data = base64.b64decode(plots['heatmap'])
            img_buffer = BytesIO(img_data)
            elements.append(Image(img_buffer, width=500, height=400))
        except:
            pass

    doc.build(elements)

    buffer.seek(0)
    return buffer

# ============================================
# HOME VIEW (MAIN)
# ============================================
def home(request):
    context = {'uploaded': False}

    data = request.session.get('data')
    if data:
        df = pd.DataFrame(data)
        context['columns'] = list(df.columns)
        context['uploaded'] = True
        context['missing_info'] = missing_vals(df)
        context['insights'] = generate_insights(df)
        plots = generate_visualizations(df)
        context.update(plots)

    if request.method == 'POST':
        action = request.POST.get("action")
        context['action'] = action

        # UPLOAD ACTION
        if action == "upload":
            file = request.FILES.get('file')

            if not file:
                context['error'] = "No file uploaded"
                return render(request, 'home.html', context)
            
            try:
                df = read_file(file)
                request.session['data'] = df.to_dict()
                
                # ✅ Store filename for display
                request.session['filename'] = file.name

                summary = basic_summary(df)
                context.update(summary)
                plots = generate_visualizations(df)
                context.update(plots)
                context['missing_info'] = missing_vals(df)
                context['insights'] = generate_insights(df)
                context['uploaded'] = True
                context['filename'] = file.name  # Pass filename to template
            except Exception as e:
                context['error'] = str(e)

        # CLEANING ACTION
        elif action == "clean":
            data = request.session.get('data')

            if not data:
                context['error'] = "No dataset found. Upload first."
                return render(request, 'home.html', context)
            
            df = pd.DataFrame(data)
            df_processed, steps = data_cleaning(request, df)
            request.session['data'] = df_processed.to_dict()

            context['processed'] = df_processed.head().to_html(classes="table")
            context['steps'] = steps
            
            summary = basic_summary(df_processed)
            context.update(summary)
            plots = generate_visualizations(df_processed)
            context.update(plots)
            context['missing_info'] = missing_vals(df_processed)
            context['uploaded'] = True

        # ANOVA ACTION
        elif action == "anova":
            data = request.session.get('data')

            if not data:
                context['error'] = "Upload dataset first"
                return render(request, 'home.html', context)
            
            df = pd.DataFrame(data)
            target_col = request.POST.get("target")
            
            if not target_col or target_col not in df.columns:
                context['error'] = "Invalid target column"
                return render(request, 'home.html', context)
            
            try:
                result_df, top_features = run_anova(df, target_col)
                context['anova'] = result_df.to_html(classes="table", index=False)
                context['top_features'] = top_features
                context['columns'] = list(df.columns)
                context['uploaded'] = True
                context['action'] = "anova"
            except Exception as e:
                context['error'] = str(e)

        # PCA ACTION
        elif action == "pca":
            data = request.session.get('data')

            if not data:
                context['error'] = "Upload dataset first"
                return render(request, 'home.html', context)

            df = pd.DataFrame(data)

            try:
                n = request.POST.get("components")
                n = int(n) if n else 2

                pca_df, variance, plot = run_pca(df, n)

                context['pca'] = pca_df.head().to_html(classes="table", index=False)
                context['variance'] = variance
                context['pca_plot'] = plot

                context['uploaded'] = True
                context['action'] = "pca"

            except Exception as e:
                context['error'] = str(e)

        # INSIGHTS ACTION
        elif action == "insights":
            data = request.session.get('data')

            if not data:
                context['error'] = "Upload dataset first"
                return render(request, 'home.html', context)

            df = pd.DataFrame(data)

            try:
                insights = generate_insights(df)
                context['insights'] = insights
                context['uploaded'] = True
                context['action'] = "insights"
            except Exception as e:
                context['error'] = str(e)

    return render(request, 'home.html', context)

# ============================================
# DOWNLOAD PDF REPORT
# ============================================
def download_report(request):
    data = request.session.get('data')

    if not data:
        return HttpResponse("No dataset available")

    df = pd.DataFrame(data)
    insights = generate_insights(df)
    plots = generate_visualizations(df)

    pdf_buffer = create_pdf_report(df, insights, plots)

    response = HttpResponse(pdf_buffer, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'

    return response

# ============================================
# DOWNLOAD PROCESSED DATA
# ============================================
def download_file(request):
    data = request.session.get('data')

    if data is None:
        return HttpResponse("No data available")

    df = pd.DataFrame(data)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="processed.csv"'

    df.to_csv(response, index=False)

    return response