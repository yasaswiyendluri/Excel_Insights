from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io
import base64


def home(request):
    context = {
        'uploaded':False
    }

    if request.method == 'POST':
        context['uploaded'] = True
        file = request.FILES.get('file')

        if file is None:
            context['error'] = "Please select a file"
            return render(request, "home.html", context)
        filename = file.name.lower()
        try:
            # -------- FILE HANDLING --------
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, engine='openpyxl',header=None)
                for i in range(5):
                    if df.iloc[i].notnull().sum()>len(df.columns)*0.5:
                        df.columns=df.iloc[i]
                        df=df[i+1:]
                        break
                df = df.reset_index(drop=True)
                df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
                df = df.loc[:, ~df.columns.duplicated()]
                
                df.reset_index(drop=True, inplace=True)

            elif filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                context['error'] = "Unsupported file format"
                return render(request, "home.html", context)
            # Ensure column names are strings
            df.columns = df.columns.astype(str).str.strip()

            # ---------- ORIGINAL DATA ----------
            context['tables'] = df.head().to_html(classes="table table-bordered")

            # ---------- FEATURE ENGINEERING ----------
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            df_processed = df.copy()
            for col in numeric_cols:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            for col in categorical_cols:
                df_processed[col] = df_processed[col].fillna("Unknown")

            # Datetime handling
            for col in df_processed.select_dtypes(include=['datetime64[ns]']).columns:
                df_processed[col + "_year"] = df_processed[col].dt.year
                df_processed[col + "_month"] = df_processed[col].dt.month
                df_processed[col + "_day"] = df_processed[col].dt.day
                df_processed.drop(col, axis=1, inplace=True)

            # Encoding
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))

            # Scaling
            numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
            numeric_df = df_processed[numeric_cols]

            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

            # ---------- CLEAN MISSING VALUES (FIXED) ----------
            missing = df.isnull().sum()

            ignore_cols = ['roll', 'id', 'name']
            missing = missing[~missing.index.str.lower().str.contains('|'.join(ignore_cols))]
            missing = missing[missing > 0]
            missing = missing.sort_values(ascending=False).head(10)

            if not missing.empty:
                context['missing'] = missing.to_frame("Missing Values").to_html(classes="table")
            else:
                context['missing'] = "<p>No missing values 🎉</p>"

            # ---------- OTHER ANALYSIS ----------
            context['dtypes'] = df.dtypes.astype(str).to_frame("Data Types").to_html(classes="table")
            context['stats'] = df.describe(include='all').to_html(classes="table")

            context['processed'] = df_processed.head().to_html(classes="table table-bordered")

            # Save for download
            request.session['processed_data'] = df_processed.to_dict()

            # ---------- AUTO INSIGHTS ----------
            insights = []

            insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")

            total_missing = df.isnull().sum().sum()
            if total_missing > 0:
                insights.append(f"Dataset contains {total_missing} missing values")
            else:
                insights.append("No missing values found")

            if not numeric_df.empty and numeric_df.dropna().shape[0]>1:
                try:
                    corr_matrix = numeric_df.corr()
                    if not corr_matrix.isnull().all().all():
                        corr_pairs = corr_matrix.unstack()
                        corr_pairs = corr_pairs[corr_pairs != 1]

                        if not corr_pairs.empty:
                            top_pair = corr_pairs.abs().idxmax()
                            insights.append(f"Strongest relationship: {top_pair[0]} ↔ {top_pair[1]}")

                    variances = numeric_df.var()
                    if not variances.isnull().all():
                        top_feature = variances.idxmax()
                        insights.append(f"Most influential feature: {top_feature}")
                except Exception as e:
                    print("Insight error:",e)
            else:
                insights.append("Not enough numeric data for correlation analysis")
            context['insights'] = insights
            # ---------- HISTOGRAM ----------
            if not numeric_df.empty:
                plt.figure(figsize=(6, 4))
                numeric_df.hist(figsize=(8,6))
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                context['histogram'] = base64.b64encode(buffer.getvalue()).decode()

                buffer.close()
                plt.close()

            # ---------- HEATMAP ----------
            if not numeric_df.empty:
                plt.figure(figsize=(6, 4))
                corr = numeric_df.corr()

                plt.imshow(corr, cmap='coolwarm')
                plt.colorbar()
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                plt.yticks(range(len(corr.columns)), corr.columns)

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                context['heatmap'] = base64.b64encode(buffer.getvalue()).decode()

                buffer.close()
                plt.close()

        except Exception as e:
            context['error'] = str(e)
            print("ERROR:", e)
            return render(request, "home.html", context)
    return render(request,"home.html",context)

def download_file(request):
    data = request.session.get('processed_data')

    if data is None:
        return HttpResponse("No data available")

    df = pd.DataFrame(data)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="processed_data.csv"'

    df.to_csv(response, index=False)

    return response