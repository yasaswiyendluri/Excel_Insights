from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import io, base64
from io import BytesIO

# import summary from services file
from core.services.summary import generate_quick_summary

#get numeric columns

def numeric_columns(df, ratio=0.1, unique=10):
   
    continuous_cols = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)

        if unique_count <= 2:
            continue
        if unique_count <= unique and unique_ratio < ratio:
            continue

        continuous_cols.append(col)
    return continuous_cols

# to get numeric columns that are moslty already encoded
def cat_like_num_columns(df, unique=15):

    encoded_like_cols = []
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        unique_count = series.nunique()
        is_integer_like = np.allclose(series, np.round(series))
        if is_integer_like and unique_count <= unique:
            encoded_like_cols.append(col)

    return encoded_like_cols

# read my uploaded file
def read_file(file):
    filename = file.name.lower()
    if filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file, engine='openpyxl')
    elif filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format")
    
    df.columns = df.columns.astype(str).str.strip() #cleaning column names
    return df

# function for checking the missing vals
def missing_vals(df):
    missing_info = []

    for col in df.columns:
        miss_cnt = df[col].isnull().sum()

        if miss_cnt > 0:
            col_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

            missing_info.append({
                "name": col,
                "missing_count": int(miss_cnt),
                "type": col_type
            })

    return missing_info

#function for basic summary
def basic_summary(df):

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
        summary['missing'] = "<p>No missing values</p>"

    summary['stats'] = df.describe(include='all').to_html(classes="table")
    summary['tables'] = df.head().to_html(classes="table")

    try:
        summary_html = generate_quick_summary(df)
        summary['data_summary'] = summary_html
    except Exception as e:
        summary['data_summary'] = f"<p>Could not generate summary: {str(e)}</p>"

    return summary

#function for data cleaning and processing
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
    
    if request.POST.get("remove_duplicates"):
        before = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        after = len(df_processed)
        steps.append(f"Removed {before - after} duplicate rows")
    
    encoded_columns = []

    if request.POST.get("encode"):
        for col in df_processed.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoded_columns.append(col)
        steps.append("Categorical encoding applied")
    
    if request.POST.get("scale"):
        continuous_cols = numeric_columns(df_processed)
        encoded_like_cols = set(cat_like_num_columns(df_processed))
        explicit_encoded_cols = set(encoded_columns)
        scale_cols = [
            col for col in continuous_cols
            if col not in encoded_like_cols and col not in explicit_encoded_cols
        ]

        if len(scale_cols) > 0:
            scaler = StandardScaler()
            df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])
            steps.append(f"Scaling applied to continuous columns: {', '.join(scale_cols)}")
            skipped_cols = sorted(list((encoded_like_cols | explicit_encoded_cols) & set(continuous_cols)))
            if skipped_cols:
                steps.append(
                    f"Skipped scaling for encoded categorical columns: {', '.join(skipped_cols)}"
                )
        else:
            steps.append("Scaling skipped (no suitable continuous numeric columns found)")
    
    return df_processed, steps

#function for ANOVA feature selection
def run_anova(df, target_col, feature_count=None):
    if target_col not in df.columns:
        raise ValueError("Selected target column not found in dataset")

    model_df = df.copy().dropna().reset_index(drop=True)
    if model_df.empty:
        raise ValueError("Dataset is empty after dropping missing values")

    X_raw = model_df.drop(columns=[target_col])
    y_raw = model_df[target_col]

    if X_raw.shape[1] == 0:
        raise ValueError("No feature columns available after removing target column")

    feature_names = list(X_raw.columns)
    X_encoded = pd.get_dummies(X_raw, drop_first=False)

    if X_encoded.shape[1] == 0:
        raise ValueError("No usable features available for selection")

    if pd.api.types.is_numeric_dtype(y_raw):
        score_func = f_regression
        y_model = y_raw.astype(float)
    else:
        score_func = f_classif
        y_model = LabelEncoder().fit_transform(y_raw.astype(str))

    max_features = min(len(feature_names), X_encoded.shape[1])
    if feature_count is None:
        k = max_features
    else:
        k = max(1, min(int(feature_count), max_features))

    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_encoded, y_model)

    encoded_scores = pd.Series(selector.scores_, index=X_encoded.columns).fillna(0.0)

    feature_scores = {}
    for col in feature_names:
        prefix = f"{col}_"
        matching = [enc_col for enc_col in encoded_scores.index if enc_col == col or enc_col.startswith(prefix)]
        feature_scores[col] = float(encoded_scores[matching].max()) if matching else 0.0

    ranked_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    selected_cols = [col for col, _ in ranked_features[:k]]

    return selected_cols, feature_scores

#function for PCA
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
    plot_url=None

    return pca_df, variance_text, plot_url

#function for insights
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
                    insights.append(f"Column '{col}' has high missing values ({round(percent, 2)}%)")
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

#function for data visualization
def generate_visualizations(df):
    plots = {}

    num_cols = numeric_columns(df)
    if not num_cols:
        num_cols = df.select_dtypes(include='number').columns.tolist()
    if len(num_cols) > 5:
        num_cols = df[num_cols].var().nlargest(5).index.tolist()
    num_cols = num_cols[:5]

    histograms = []
    for col in num_cols[:3]:
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

    # Pick top correlated pairs for scatter plots
    scatterplots = []
    if len(num_cols) >= 2:
        pair_scores = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col_x = num_cols[i]
                col_y = num_cols[j]
                pair_df = df[[col_x, col_y]].dropna()
                if len(pair_df) < 3:
                    continue
                corr_val = pair_df[col_x].corr(pair_df[col_y])
                score = abs(corr_val) if pd.notna(corr_val) else 0
                pair_scores.append((score, col_x, col_y))

        pair_scores.sort(reverse=True, key=lambda x: x[0])
        selected_pairs = [(x, y) for _, x, y in pair_scores[:3]]
        if not selected_pairs:
            selected_pairs = [(num_cols[i], num_cols[i + 1]) for i in range(min(3, len(num_cols) - 1))]

        for col_x, col_y in selected_pairs:
            try:
                pair_df = df[[col_x, col_y]].dropna()
                if pair_df.empty:
                    continue

                plt.figure(figsize=(8, 5))
                plt.scatter(pair_df[col_x], pair_df[col_y], alpha=0.65, color='#2E7D32', s=28, edgecolors='none')
                plt.xlabel(col_x, fontsize=11)
                plt.ylabel(col_y, fontsize=11)
                plt.title(f"{col_x} vs {col_y}", fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.25, linewidth=0.6)
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                scatterplots.append(base64.b64encode(buffer.getvalue()).decode())
                buffer.close()
                plt.close()
            except Exception as e:
                print(f"Error creating scatter plot: {e}")
    
    plots['scatterplots'] = scatterplots

    if len(num_cols) >= 2:
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
    else:
        plots['heatmap'] = None

    return plots

def create_pdf_report(df, insights, plots):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Excel Insights Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}", styles['Normal']))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Key Insights:", styles['Heading2']))
    for ins in insights[:10]:
        elements.append(Paragraph(f"• {ins}", styles['Normal']))
    elements.append(Spacer(1, 20))

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

    if 'scatterplots' in plots and plots['scatterplots']:
        elements.append(Paragraph("Scatter Plots", styles['Heading2']))
        
        for img in plots['scatterplots'][:3]:
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


def home(request):
    context = {'uploaded': False}

    data = request.session.get('original_data') or request.session.get('data')
    if data:
        df = pd.DataFrame(data)
        summary = basic_summary(df)
        context.update(summary)
        context['columns'] = list(df.columns)
        context['uploaded'] = True
        context['missing_info'] = missing_vals(df)
        context['insights'] = generate_insights(df)
        plots = generate_visualizations(df)
        context.update(plots)

    if request.method == 'POST':
        action = request.POST.get("action")
        context['action'] = action

        if action == "upload":
            file = request.FILES.get('file')

            if not file:
                context['error'] = "No file uploaded"
                return render(request, 'home.html', context)
            
            try:
                df = read_file(file)
                request.session['original_data'] = df.to_dict()
                request.session['data'] = df.to_dict() 

                request.session['filename'] = file.name

                summary = basic_summary(df)
                context.update(summary)
                plots = generate_visualizations(df)
                context.update(plots)
                context['missing_info'] = missing_vals(df)
                context['insights'] = generate_insights(df)
                context['uploaded'] = True
                context['filename'] = file.name
            except Exception as e:
                context['error'] = str(e)
        # data cleaning and processing
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
            
            original_data = request.session.get('original_data')
            original_df = pd.DataFrame(original_data)

            summary = basic_summary(original_df)
            context.update(summary)
            plots = generate_visualizations(df_processed)
            context.update(plots)
            context['missing_info'] = missing_vals(df_processed)
            context['uploaded'] = True
        # ANOVA feature selection
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
                requested_count = request.POST.get("feature_count")
                feature_count = int(requested_count) if requested_count else None

                selected_cols, feature_scores = run_anova(df, target_col, feature_count)
                ranked_rows = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
                result_df = pd.DataFrame(ranked_rows, columns=["Feature", "Score"])

                selected_count = len(selected_cols)
                selected_df = df[selected_cols] if selected_cols else pd.DataFrame()
                selected_head = selected_df.head().to_html(classes="table", index=False) if selected_cols else None
                request.session['selected_features_data'] = selected_df.to_dict()

                context['anova'] = result_df.to_html(classes="table", index=False)
                context['selected_head'] = selected_head
                context['selected_count'] = selected_count
                context['target_col'] = target_col
                context['columns'] = list(df.columns)
                context['uploaded'] = True
                context['action'] = "anova"
            except Exception as e:
                context['error'] = str(e)
        # PCA
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
                request.session['pca_data'] = pca_df.to_dict()

                context['uploaded'] = True
                context['action'] = "pca"

            except Exception as e:
                context['error'] = str(e)

        # insights
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

def download_file(request):
    data = request.session.get('data')

    if data is None:
        return HttpResponse("No data available")

    df = pd.DataFrame(data)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="processed.csv"'

    df.to_csv(response, index=False)

    return response


def download_basic_stats(request):
    data = request.session.get('original_data')
    if data is None:
        return HttpResponse("No data available")

    df = pd.DataFrame(data)
    stats_df = df.describe(include='all').transpose().reset_index().rename(columns={"index": "column"})

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="basic_statistics.csv"'
    stats_df.to_csv(response, index=False)
    return response


def download_selected_features(request):
    selected_data = request.session.get('selected_features_data')
    if not selected_data:
        return HttpResponse("No selected feature data available")

    selected_df = pd.DataFrame(selected_data)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="selected_features.csv"'
    selected_df.to_csv(response, index=False)
    return response


def download_pca(request):
    pca_data = request.session.get('pca_data')
    if not pca_data:
        return HttpResponse("No PCA data available")

    pca_df = pd.DataFrame(pca_data)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="pca_transformed.csv"'
    pca_df.to_csv(response, index=False)
    return response