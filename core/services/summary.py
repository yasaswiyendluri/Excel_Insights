"""
Summary Service Module
Handles all data summary, statistics, and count operations
Works with any domain: health, finance, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class DataSummaryService:
    """Service for generating comprehensive data summaries"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

   #basic statistics

    def get_basic_stats(self) -> Dict[str, Any]:
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing = int(self.df.isnull().sum().sum())
        return {
            'rows': self.df.shape[0],
            'columns': self.df.shape[1],
            'numeric_cols': len(self.numeric_cols),
            'categorical_cols': len(self.categorical_cols),
            'missing_count': missing,
            'missing_percent': round((missing / total_cells) * 100, 2) if total_cells else 0,
            'duplicates': int(self.df.duplicated().sum()),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        }

    #column wise

    def _value_counts_table(self, column: str, top_n: int = 10) -> str:
        """Return an HTML table of value counts matching the app's table style."""
        counts = self.df[column].value_counts().head(top_n)
        pcts = (counts / len(self.df) * 100).round(2)

        rows = ""
        for val, cnt, pct in zip(counts.index, counts.values, pcts.values):
            rows += f"""
            <tr>
                <td>{val}</td>
                <td>{cnt}</td>
                <td>{pct}%</td>
            </tr>"""

        return f"""
        <table>
            <tr><th>Value</th><th>Count</th><th>Percentage</th></tr>
            {rows}
        </table>"""

    # Domain wise

    def _find_column(self, candidates: list, pool: list) -> str | None:
        for col in pool:
            if col.lower() in candidates:
                return col
        return None

    def get_gender_distribution(self):
        col = self._find_column(['gender', 'sex', 'gênero'], self.categorical_cols)
        if not col:
            return None
        return {'column_name': col, 'html_table': self._value_counts_table(col)}

    def get_age_statistics(self):
        col = self._find_column(['age', 'years', 'edad', 'âge'], self.numeric_cols)
        if not col:
            return None
        s = self.df[col].dropna()
        return {
            'column_name': col,
            'min': float(s.min()),
            'max': float(s.max()),
            'mean': float(s.mean()),
            'median': float(s.median()),
            'std': float(s.std()),
        }

    def get_health_status_distribution(self):
        keywords = ['status', 'disease', 'condition', 'diagnosis', 'outcome',
                    'target', 'label', 'class', 'category']
        col = next(
            (c for c in self.categorical_cols
             if any(k in c.lower() for k in keywords)),
            None
        )
        if not col:
            return None
        return {'column_name': col, 'html_table': self._value_counts_table(col)}

    def get_categorical_summaries(self, limit: int = 50) -> list:
        result = []
        for col in self.categorical_cols:
            unique = self.df[col].nunique()
            if unique == len(self.df) or unique > limit:
                continue
            result.append({
                'name': col,
                'unique_count': unique,
                'html_table': self._value_counts_table(col),
            })
        return result

    def get_numeric_summaries(self) -> list:
        result = []
        for col in self.numeric_cols:
            s = self.df[col].dropna()
            result.append({
                'name': col,
                'dtype': str(self.df[col].dtype),
                'count': int(s.count()),
                'mean': float(s.mean()),
                'median': float(s.median()),
                'std': float(s.std()),
                'min': float(s.min()),
                'max': float(s.max()),
                'q25': float(s.quantile(0.25)),
                'q75': float(s.quantile(0.75)),
                'missing': int(self.df[col].isnull().sum()),
            })
        return result

    # HTML part

    def generate_summary_html(self) -> str:
        basic   = self.get_basic_stats()
        gender  = self.get_gender_distribution()
        age     = self.get_age_statistics()
        health  = self.get_health_status_distribution()
        cats    = self.get_categorical_summaries()
        nums    = self.get_numeric_summaries()

        mp = basic['missing_percent']
        quality_color = '#4CAF50' if mp < 5 else ('#FF9800' if mp < 20 else '#F44336')
        quality_label = 'Excellent' if mp < 5 else ('Good' if mp < 20 else 'Fair')

        html = ""

        # Dataset overview section
        html += f"""
        <h3 style="font-size:14px; font-weight:600; color:#4CAF50;
                   margin:0 0 14px 0; padding-bottom:8px; border-bottom:1px solid #2a2a3d;">
            Dataset Overview
        </h3>
        <div class="overview-grid">
            <div class="stat-item">
                <span class="stat-label">Total Records</span>
                <span class="stat-value">{basic['rows']:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Features</span>
                <span class="stat-value">{basic['columns']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Numeric Cols</span>
                <span class="stat-value">{basic['numeric_cols']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Categorical Cols</span>
                <span class="stat-value">{basic['categorical_cols']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Missing Values</span>
                <span class="stat-value">{basic['missing_count']} ({mp}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Duplicate Rows</span>
                <span class="stat-value">{basic['duplicates']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Memory Usage</span>
                <span class="stat-value">{basic['memory_usage']}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Data Quality</span>
                <span class="stat-value">
                    <span class="quality-badge" style="background:{quality_color};">
                        {quality_label}
                    </span>
                </span>
            </div>
        </div>
        """

        # age stats
        if age:
            html += f"""
            <h3 class="sum-section-title">Age Statistics</h3>
            <div class="age-row">
                <div class="age-cell">
                    <div class="age-cell-label">Range</div>
                    <div class="age-cell-value">{age['min']:.0f} – {age['max']:.0f} yrs</div>
                </div>
                <div class="age-cell">
                    <div class="age-cell-label">Mean</div>
                    <div class="age-cell-value">{age['mean']:.1f} yrs</div>
                </div>
                <div class="age-cell">
                    <div class="age-cell-label">Median</div>
                    <div class="age-cell-value">{age['median']:.1f} yrs</div>
                </div>
                <div class="age-cell">
                    <div class="age-cell-label">Std Dev</div>
                    <div class="age-cell-value">{age['std']:.2f}</div>
                </div>
            </div>
            """

        # gender distn
        if gender:
            html += f"""
            <h3 class="sum-section-title">Gender Distribution</h3>
            {gender['html_table']}
            """

        # Health stats
        if health:
            html += f"""
            <h3 class="sum-section-title">{health['column_name'].replace('_', ' ').title()} Distribution</h3>
            {health['html_table']}
            """

        # categorical features
        if cats:
            html += '<h3 class="sum-section-title">Categorical Features</h3>'
            for item in cats[:10]:
                html += f"""
                <div class="feature-collapse" onclick="this.classList.toggle('open')">
                    <div class="feature-row">
                        <span class="feature-col-name">{item['name']}</span>
                        <span class="feature-col-meta">{item['unique_count']} unique values ▾</span>
                    </div>
                    <div class="feature-collapse-body">
                        {item['html_table']}
                    </div>
                </div>
                """

        # numeric features  
        if nums:
            html += '<h3 class="sum-section-title">Numeric Features</h3>'
            for item in nums[:10]:
                miss_color = '#4CAF50' if item['missing'] == 0 else '#FF9800'
                html += f"""
                <div class="feature-collapse" onclick="this.classList.toggle('open')">
                    <div class="feature-row">
                        <span class="feature-col-name">{item['name']}</span>
                        <span class="feature-col-meta">{item['dtype']} · missing: {item['missing']} ▾</span>
                    </div>
                    <div class="feature-collapse-body">
                        <div class="num-mini-grid">
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Count</div>
                                <div class="num-mini-value">{item['count']}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Mean</div>
                                <div class="num-mini-value">{item['mean']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Median</div>
                                <div class="num-mini-value">{item['median']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Std Dev</div>
                                <div class="num-mini-value">{item['std']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Min</div>
                                <div class="num-mini-value">{item['min']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Max</div>
                                <div class="num-mini-value">{item['max']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Q1 (25%)</div>
                                <div class="num-mini-value">{item['q25']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Q3 (75%)</div>
                                <div class="num-mini-value">{item['q75']:.2f}</div>
                            </div>
                            <div class="num-mini-cell">
                                <div class="num-mini-label">Missing</div>
                                <div class="num-mini-value" style="color:{miss_color};">{item['missing']}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """

        return html


def get_summary_service(df: pd.DataFrame) -> DataSummaryService:
    return DataSummaryService(df)


def generate_quick_summary(df: pd.DataFrame) -> str:
    return DataSummaryService(df).generate_summary_html()