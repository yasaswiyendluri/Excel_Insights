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
        'uploaded': False
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
                df = pd.read_excel(file, engine='openpyxl', header=None)

                for i in range(5):
                    if df.iloc[i].notnull().sum() > len(df.columns) * 0.5:
                        df.columns = df.iloc[i]
                        df = df[i + 1:]
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

            # ---------- FILTER USEFUL NUMERIC COLUMNS ----------
            filtered_cols = []

            for col in numeric_df.columns:
                if numeric_df[col].nunique() > 10:
                    if not any(x in col.lower() for x in ['id', 'no', 'mobile']):
                        filtered_cols.append(col)

            clean_numeric_df = numeric_df[filtered_cols]

            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

            # ---------- CLEAN MISSING VALUES ----------
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

            if not numeric_df.empty and numeric_df.dropna().shape[0] > 1:
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
                    print("Insight error:", e)
            else:
                insights.append("Not enough numeric data for correlation analysis")

            context['insights'] = insights

        
            # ---------- HISTOGRAM ----------
            if not clean_numeric_df.empty:
                plt.figure(figsize=(8, 6))
                clean_numeric_df.hist(figsize=(10, 8))
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)

                context['histogram'] = base64.b64encode(buffer.getvalue()).decode()

                buffer.close()
                plt.close()
            else:
                context['histogram'] = None

            # ---------- HEATMAP ----------
            if not numeric_df.empty:
                plt.figure(figsize=(14, 10))

                corr = numeric_df.corr()

                plt.imshow(corr, cmap='coolwarm', aspect='auto')
                plt.colorbar()

                plt.xticks(
                    ticks=range(len(corr.columns)),
                    labels=corr.columns,
                    rotation=90,
                    fontsize=8
                )
                plt.yticks(
                    ticks=range(len(corr.columns)),
                    labels=corr.columns,
                    fontsize=8
                )

                plt.title("Correlation Heatmap")

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.3, left=0.3)

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)

                context['heatmap'] = base64.b64encode(buffer.getvalue()).decode()

                buffer.close()
                plt.close()

        except Exception as e:
            context['error'] = str(e)
            print("ERROR:", e)
            return render(request, "home.html", context)

    return render(request, "home.html", context)


def download_file(request):
    data = request.session.get('processed_data')

    if data is None:
        return HttpResponse("No data available")

    df = pd.DataFrame(data)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=\"processed_data.csv\"'

    df.to_csv(response, index=False)

    return response