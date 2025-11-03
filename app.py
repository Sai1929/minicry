
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy import stats
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pdf_utils import styles

import os

# PDF utils (local)
from pdf_utils import (
    generate_section_pdf,
    merge_section_pdfs,
    df_to_table_flowable,
    append_image_flowable
)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Process Performance Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f9f9fb;
            padding: 1.5rem;
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #1a1a1a;
        }
        .stDataFrame, .stTable {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            background: white;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .stMetric {
            background: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        hr {
            border: 0;
            border-top: 1px solid #e0e0e0;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .baseline-cell {
            background-color: #e6fff0;
            padding: 6px;
            border-radius: 4px;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 10px;
        }
        .section-header {
            color: #34495E;
            font-size: 16px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 5px;
            border-bottom: 1px solid #e0e0e0;
        }
        .section-option {
            margin-left: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helper Function: Minitab-style Anderson–Darling Test ---
def anderson_darling_minitab(data):
    ad_result = stats.anderson(data, dist='norm')
    ad_stat = ad_result.statistic
    n = len(data)
    adj_ad = ad_stat * (1 + 0.75/n + 2.25/(n**2))
    if adj_ad >= 0.6:
        p = np.exp(1.2937 - 5.709 * adj_ad - 0.0186 * adj_ad**2)
    elif adj_ad >= 0.34:
        p = np.exp(0.9177 - 4.279 * adj_ad - 1.38 * adj_ad**2)
    elif adj_ad >= 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * adj_ad - 59.938 * adj_ad**2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * adj_ad - 223.73 * adj_ad**2)
    p = np.clip(p, 0, 1)
    return adj_ad, p

# image saving helper
def save_fig_as_png(fig, name):
    path = f"{name}.png"
    fig.savefig(os.path.join("reports", "Regression_plot.png"))
    plt.close(fig)
    return path

# --- Page Title ---
st.title("Process Performance Model")

# --- Sidebar Navigation (Collapsible like Minitab / Crystal Ball) ---
st.sidebar.markdown("<h2 style='color:#1f2937;'>Dashboard Sections</h2>", unsafe_allow_html=True)

# --- Main Navigation Group ---
nav_group = st.sidebar.radio("Select Mode", ["Minitab", "Crystal Ball"], horizontal=True)

# --- Initialize variable ---
section = None
if nav_group == "Minitab":
    section = st.sidebar.radio(
        "Minitab Modules",
        (
            "Upload & Normality",  # now appears first
            "Regression",             # renamed from "Upload & Regression"
            "Correlation & p-values",
            "IMR Chart",
            "Prediction"
        ),
        label_visibility="collapsed"
    )
elif nav_group == "Crystal Ball":
    section = st.sidebar.radio(
        "Crystal Ball Modules",
        (
            "Define Assumptions",
            "What-If Analysis",
            "Forecasting & Sensitivity"
        ),
        label_visibility="collapsed"
    )

# --- File Upload Section ---
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded:
    with st.spinner("Loading dataset..."):
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.session_state["df"] = df  # Always store latest data

    # ==========================================================
    # Upload & Normality
    # ==========================================================
    if section == "Upload & Normality":
        st.header("Distribution and Normality Analysis")
        df = st.session_state.get("df", df)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if "Residuals" in df.columns:
            st.success("Regression residuals detected — you can include them in analysis.")
            numeric_cols = ["Residuals"] + [c for c in numeric_cols if c != "Residuals"]

        selected_vars = st.multiselect(
            "Select variables for analysis",
            numeric_cols,
            default=["Residuals"] if "Residuals" in numeric_cols else []
        )

        if selected_vars:
            # Build flowables for the Upload & Normality section
            flowables = []
            flowables.append(Paragraph(f"Dataset: {uploaded.name}", style=None if False else None))
            flowables.append(Spacer(1, 6))

            for selected_var in selected_vars:
                st.markdown(f"### {selected_var}")
                with st.spinner(f"Analyzing {selected_var}..."):
                    data = df[selected_var].dropna()
                    n = len(data)

                    # --- Descriptive Stats ---
                    mean_val = np.mean(data)
                    std_val = np.std(data, ddof=1)
                    variance = np.var(data, ddof=1)
                    skew = stats.skew(data)
                    kurt = stats.kurtosis(data)

                    # --- Anderson–Darling Test ---
                    ad_stat, approx_p = anderson_darling_minitab(data)
                    norm_result = ("Data appears NORMAL (p > 0.05)" if approx_p > 0.05 else "Data is NOT normal (p < 0.05)")

                    # --- Confidence Intervals ---
                    ci_mean = stats.t.interval(0.95, n-1, loc=mean_val, scale=std_val/np.sqrt(n))
                    ci_median = np.percentile(data, [2.5, 97.5])
                    ci_std = (
                        std_val * np.sqrt((n - 1) / stats.chi2.ppf(0.975, n - 1)),
                        std_val * np.sqrt((n - 1) / stats.chi2.ppf(0.025, n - 1))
                    )

                    q1, median, q3 = np.percentile(data, [25, 50, 75])
                    data_min, data_max = np.min(data), np.max(data)

                    # --- Plot ---
                    fig_main, (ax1, ax_box) = plt.subplots(
                        2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]}
                    )

                    sns.histplot(data, kde=False, bins=10, color="skyblue", ax=ax1)
                    xmin, xmax = ax1.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, mean_val, std_val)
                    ax1.plot(x, p * len(data) * (xmax - xmin) / 10, 'r-', lw=2)
                    ax1.set_title(f"{selected_var} Distribution with Normal Curve")

                    sns.boxplot(x=data, color="lightblue", ax=ax_box, orient="h")
                    ax_box.set_xlabel(selected_var)

                    # --- Summary ---
                    summary_text = (
                        f"Anderson–Darling A-Squared: {ad_stat:.3f}\n"
                        f"Approx. P-Value: {approx_p:.3f}\n"
                        f"{norm_result}\n"
                        f"Mean: {mean_val:.5f}  StDev: {std_val:.5f}  Variance: {variance:.5f}\n"
                        f"Skewness: {skew:.5f}  Kurtosis: {kurt:.5f}  N: {n}\n"
                        f"95% CI Mean: {ci_mean[0]:.5f} to {ci_mean[1]:.5f}"
                    )

                    st.pyplot(fig_main)
                    st.text(summary_text)
                    st.table(pd.DataFrame({
                        "Statistic": ["Minimum", "1st Quartile (Q1)", "Median", "3rd Quartile (Q3)", "Maximum"],
                        "Value": [data_min, q1, median, q3, data_max]
                    }))

                    # Save plot and append to flowables for PDF
                    img_path = save_fig_as_png(fig_main, f"upload_normality_{selected_var}")
                    append_image_flowable(flowables, f"{selected_var} Distribution & Boxplot", img_path)
                    # add summary table
                    stats_df = pd.DataFrame({
                        "Metric": ["A-Squared", "Approx. P", "Mean", "Std", "Variance", "Skew", "Kurtosis", "N"],
                        "Value": [round(ad_stat,3), round(approx_p,3), round(mean_val,5), round(std_val,5),
                                  round(variance,5), round(skew,5), round(kurt,5), int(n)]
                    })
                    flowables.append(df_to_table_flowable(stats_df))
                    flowables.append(Spacer(1,12))

            # Generate per-section PDF (fixed name - overwrites on re-run)
            generate_section_pdf("Upload & Normality", flowables)
            st.success(" Upload & Normality PDF generated (Upload_&_Normality_report.pdf).")
        else:
            st.info("Select one or more numeric variables to begin analysis.")
    # ==========================================================
    # Regression (updated PDF format with ANOVA & Equation)
    # ==========================================================
    elif section == "Regression":
        st.header("Regression Analysis")

        prev_dep = st.session_state.get("dep_var", None)
        prev_indep = st.session_state.get("indep_vars", None)

        dep_var = st.selectbox(
            "Select dependent variable (Y)",
            num_cols,
            index=num_cols.index(prev_dep) if prev_dep in num_cols else 0,
            key="dep_var_select"
        )

        available_indep_vars = [c for c in num_cols if c != dep_var]
        indep_vars = st.multiselect(
            "Select independent variables (X)",
            available_indep_vars,
            default=(
                prev_indep if prev_indep and all(v in available_indep_vars for v in prev_indep)
                else available_indep_vars
            ),
            key="indep_var_select"
        )

        st.session_state["dep_var"] = dep_var
        st.session_state["indep_vars"] = indep_vars

        if indep_vars:
            with st.spinner("Running regression analysis..."):
                X = df[indep_vars]
                X = sm.add_constant(X)
                y = df[dep_var]
                model = sm.OLS(y, X).fit()
                # Save residuals
                df["Residuals"] = model.resid

            st.session_state["model"] = model
            st.session_state["df"] = df

            # Preview
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            # correlation / pairplot
            with st.spinner("Generating correlation charts..."):
                st.subheader("Correlation Outliers Chart")
                selected_corr_vars = [dep_var] + indep_vars
                if len(selected_corr_vars) >= 2:
                    sample_df = df[selected_corr_vars].sample(min(len(df), 1000), random_state=42)
                    fig_corr = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
                    st.pyplot(fig_corr.fig)
                else:
                    st.info("Select at least two numeric variables to generate the matrix plot.")

            # Regression summary UI (unchanged)
            st.subheader("Regression Summary")
            summary_html = model.summary().as_html()
            st.markdown("""
            <style>
            .summary-box {
                background-color: #fdfdfd;
                border: 1px solid #d3d3d3;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #2f2f2f;
                overflow-x: auto;
                white-space: pre-wrap;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown(f"<div class='summary-box'>{summary_html}</div>", unsafe_allow_html=True)

            # =========================
            # ANOVA: compute Type I-like contributions using ols drop-one approach
            # =========================
            anova_display = None
            try:
                from statsmodels.formula.api import ols
                from scipy import stats as spstats

                # Build formula safely using Q() to handle special chars
                # e.g. f"Q('{dep_var}') ~ Q('x1') + Q('x2')"
                formula_full = f"Q('{dep_var}') ~ " + " + ".join([f"Q('{v}')" for v in indep_vars])
                anova_model = ols(formula_full, data=df).fit()

                fitted = anova_model.fittedvalues
                yvals = anova_model.model.endog
                n = len(yvals)

                SSR_full = np.sum((fitted - yvals.mean()) ** 2)
                SSE = np.sum(anova_model.resid ** 2)
                SST = SSR_full + SSE

                df_reg = len(indep_vars)
                df_resid = int(anova_model.df_resid)
                df_total = df_reg + df_resid

                MSE = SSE / df_resid if df_resid > 0 else np.nan

                term_rows = []
                for term in indep_vars:
                    other_terms = [v for v in indep_vars if v != term]
                    if other_terms:
                        formula_reduced = f"Q('{dep_var}') ~ " + " + ".join([f"Q('{v}')" for v in other_terms])
                    else:
                        formula_reduced = f"Q('{dep_var}') ~ 1"

                    reduced_model = ols(formula_reduced, data=df).fit()
                    fitted_reduced = reduced_model.fittedvalues
                    SSR_reduced = np.sum((fitted_reduced - yvals.mean()) ** 2)

                    SS_term = SSR_full - SSR_reduced
                    DF_term = 1
                    MS_term = SS_term / DF_term if DF_term != 0 else np.nan
                    F_term = (MS_term / MSE) if (MSE != 0 and not np.isnan(MS_term)) else np.nan
                    p_term = spstats.f.sf(F_term, DF_term, df_resid) if not np.isnan(F_term) else np.nan

                    term_rows.append({
                        "Source": term,
                        "DF": DF_term,
                        "Adj SS": SS_term,
                        "Adj MS": MS_term,
                        "F-Value": F_term,
                        "P-Value": p_term
                    })

                terms_df = pd.DataFrame(term_rows)

                regression_row = pd.DataFrame([{
                    "Source": "Regression",
                    "DF": df_reg,
                    "Adj SS": SSR_full,
                    "Adj MS": SSR_full / df_reg if df_reg > 0 else np.nan,
                    "F-Value": (SSR_full / df_reg) / MSE if (MSE != 0) else np.nan,
                    "P-Value": spstats.f.sf((SSR_full / df_reg) / MSE, df_reg, df_resid) if (MSE != 0) else np.nan
                }])

                error_row = pd.DataFrame([{
                    "Source": "Error",
                    "DF": df_resid,
                    "Adj SS": SSE,
                    "Adj MS": MSE,
                    "F-Value": np.nan,
                    "P-Value": np.nan
                }])

                total_row = pd.DataFrame([{
                    "Source": "Total",
                    "DF": df_total,
                    "Adj SS": SST,
                    "Adj MS": np.nan,
                    "F-Value": np.nan,
                    "P-Value": np.nan
                }])

                # indent term names for readability
                terms_df_display = terms_df.copy()
                terms_df_display["Source"] = "   " + terms_df_display["Source"]

                anova_display = pd.concat([regression_row, terms_df_display, error_row, total_row], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not compute ANOVA: {e}")
                anova_display = None

            # Show ANOVA in Streamlit UI (if computed)
            if anova_display is not None:
                st.subheader("Analysis of Variance (ANOVA)")
                def highlight_rows(row):
                    if row['Source'].strip() in ['Regression', 'Error', 'Total']:
                        return ['font-weight: bold; background-color: #f5f5f5'] * len(row)
                    elif isinstance(row['Source'], str) and row['Source'].startswith("   "):
                        return ['background-color: #fafafa'] * len(row)
                    else:
                        return [''] * len(row)

                st.dataframe(
                    anova_display[['Source','DF','Adj SS','Adj MS','F-Value','P-Value']]
                    .style.apply(highlight_rows, axis=1)
                    .format({
                        'DF': lambda x: f"{int(x)}" if pd.notna(x) and isinstance(x, (int, np.integer, float, np.floating)) else "—",
                        'Adj SS': lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else "—",
                        'Adj MS': lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float)) else "—",
                        'F-Value': lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "—",
                        'P-Value': lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
                    })
                )

            # Regression Equation (build pretty text)
            params = model.params
            # pick intercept name if present
            intercept = None
            for k in params.index:
                if k.lower() in ['intercept', 'const']:
                    intercept = params[k]
                    break

            equation_parts = []
            if intercept is not None:
                equation_parts.append(f"{intercept:.4f}")
            for v in indep_vars:
                # try raw name, else Q('name') variants
                coef = params.get(v, np.nan)
                if pd.isna(coef):
                    # find any param that contains var name (safe fallback)
                    for pk in params.index:
                        if v in str(pk):
                            coef = params[pk]
                            break
                if pd.notna(coef):
                    equation_parts.append(f"({coef:.4f} × {v})")
                else:
                    equation_parts.append(f"(NA × {v})")

            equation_text = f"{dep_var} = " + " + ".join(equation_parts)
            st.subheader("Regression Equation")
            st.markdown(f"**{equation_text}**")

            # R² metrics in UI
            colA, colB = st.columns(2)
            colA.metric("R² (Goodness of Fit)", f"{model.rsquared:.3f}")
            colB.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")

            # Regression plot: single X or fitted vs actual
            fig_reg, ax_reg = plt.subplots(figsize=(8,5))
            ax_reg.scatter(df[indep_vars[0]], df[dep_var], alpha=0.6)
            if len(indep_vars) == 1:
                X_plot = sm.add_constant(df[indep_vars[0]])
                ax_reg.plot(df[indep_vars[0]], model.predict(X_plot), color='red')
                ax_reg.set_xlabel(indep_vars[0])
                ax_reg.set_ylabel(dep_var)
            else:
                ax_reg.scatter(model.fittedvalues, df[dep_var], alpha=0.6)
                ax_reg.set_xlabel("Fitted values")
                ax_reg.set_ylabel(dep_var)
                ax_reg.set_title("Fitted vs Actual / Regression scatter")
            st.pyplot(fig_reg)

            # -----------------------
            # Build PDF flowables (ANOVA, equation, metrics, summary text, plot)
            # -----------------------
            flowables = []

            # Model summary text
            flowables.append(Paragraph("Model Summary (text)", styles["Heading3"]))
            summary_text = str(model.summary())
            flowables.append(Paragraph(summary_text.replace("\n", "<br/>"), styles["Code"]))
            flowables.append(Spacer(1,12))

            # ANOVA table for PDF (if available)
            if anova_display is not None:
                # format numeric columns nicely for PDF table
                pdf_anova = anova_display.copy()
                # Round numeric columns
                for col in ["Adj SS","Adj MS","F-Value","P-Value"]:
                    if col in pdf_anova.columns:
                        pdf_anova[col] = pdf_anova[col].apply(lambda x: "" if pd.isna(x) else (f"{x:.6f}" if col!="F-Value" and col!="P-Value" else (f"{x:.2f}" if col=="F-Value" else f"{x:.3f}")))
                flowables.append(Paragraph("Analysis of Variance (ANOVA)", styles["Heading3"]))
                flowables.append(df_to_table_flowable(pdf_anova[['Source','DF','Adj SS','Adj MS','F-Value','P-Value']], table_width=380, font_size=7))
                flowables.append(Spacer(1,12))

            # Regression equation in PDF
            flowables.append(Paragraph("Regression Equation", styles["Heading3"]))
            flowables.append(Paragraph(equation_text, styles["Normal"]))
            flowables.append(Spacer(1,12))

            # Key metrics table for PDF
            metrics_df = pd.DataFrame({
                "Metric": ["R-squared", "Adj. R-squared", "No. Observations", "DF Residual"],
                "Value": [round(model.rsquared,4), round(model.rsquared_adj,4), int(model.nobs), int(model.df_resid)]
            })
            flowables.append(Paragraph("Key Metrics", styles["Heading3"]))
            flowables.append(df_to_table_flowable(metrics_df))
            flowables.append(Spacer(1,12))

            # Regression plot saved & appended
            reg_img = save_fig_as_png(fig_reg, "regression_plot")
            append_image_flowable(flowables, "Regression Plot", reg_img)

            # generate per-section PDF (overwrites)
            generate_section_pdf("Regression", flowables)
            st.success("Regression PDF generated (Regression_report.pdf).")
        else:
            st.info("Select one or more independent variables to run regression.")

    # ==========================================================
    # Correlation & p-values
    # ==========================================================
    elif section == "Correlation & p-values":
        if "model" in st.session_state:
            df = st.session_state["df"]
            model = st.session_state["model"]
            indep_vars = st.session_state["indep_vars"]

            st.header("Correlation Heatmap and Variable Significance")
            with st.spinner("Generating correlation heatmap..."):
                fig3, ax3 = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
                st.pyplot(fig3)

            with st.spinner("Computing variable significance..."):
                pvals = model.pvalues.drop("const", errors="ignore")
                sig = pd.DataFrame({"Variable": pvals.index, "p-Value": pvals.values})
                sig["Significant?"] = sig["p-Value"].apply(lambda x: "Yes" if x < 0.05 else "No")
                st.subheader("Regression Variable Significance (p-Values)")
                st.dataframe(sig, use_container_width=True)

                fig, ax = plt.subplots()
                sns.barplot(x="Variable", y="p-Value", data=sig, ax=ax)
                ax.axhline(0.05, color="red", linestyle="--")
                ax.set_title("p-Value Significance Threshold (0.05)")
                st.pyplot(fig)

            with st.spinner("Calculating Variance Inflation Factor (VIF)..."):
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X = df[indep_vars].dropna()
                X = sm.add_constant(X)
                vif_data = pd.DataFrame()
                vif_data["Variable"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                vif_data = vif_data[vif_data["Variable"] != "const"]

                st.subheader("Variance Inflation Factor (VIF) — Multicollinearity Check")
                st.markdown("""
                - **VIF < 5** → Low multicollinearity 
                - **VIF 5–10** → Moderate multicollinearity  
                - **VIF > 10** → High multicollinearity 
                """)
                st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True)

                fig_vif, ax_vif = plt.subplots()
                sns.barplot(x="Variable", y="VIF", data=vif_data, ax=ax_vif, color="skyblue")
                ax_vif.axhline(5, color="orange", linestyle="--", label="VIF = 5 Threshold")
                ax_vif.axhline(10, color="red", linestyle="--", label="VIF = 10 Threshold")
                ax_vif.set_title("Variance Inflation Factor (VIF) Chart")
                ax_vif.legend()
                st.pyplot(fig_vif)

            # Build flowables for correlation PDF
            flowables = []
            corr_df = df.corr().round(3)
            flowables.append(df_to_table_flowable(corr_df))
            flowables.append(Spacer(1,12))
            flowables.append(Paragraph("Variable Significance (p-values)", style=None if False else None))
            flowables.append(df_to_table_flowable(sig.reset_index(drop=True)))
            flowables.append(Spacer(1,12))

            corr_img = save_fig_as_png(fig3, "correlation_heatmap")
            append_image_flowable(flowables, "Correlation Heatmap", corr_img)

            # VIF chart
            vif_img = save_fig_as_png(fig_vif, "vif_chart")
            append_image_flowable(flowables, "VIF Chart", vif_img)

            generate_section_pdf("Correlation & p-values", flowables)
            st.success("Correlation PDF generated (Correlation_&_p-values_report.pdf).")
        else:
            st.warning("Run regression first to view significance and multicollinearity charts.")

    # ==========================================================
    # IMR Chart
    # ==========================================================
    elif section == "IMR Chart":
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.header("IMR Chart (Individuals and Moving Range)")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(numeric_cols) == 0:
                st.warning("No numeric columns available for IMR Chart.")
            else:
                selected_imr_var = st.selectbox("Select variable for IMR Chart", numeric_cols)

                if selected_imr_var:
                    data_imr = df[selected_imr_var].dropna().reset_index(drop=True)

                    # --- Calculate Moving Range ---
                    mr = data_imr.diff().abs().dropna()

                    # --- Individuals Chart Limits ---
                    mean_i = data_imr.mean()
                    mean_mr = mr.mean()
                    UCL_I = mean_i + 2.66 * mean_mr
                    LCL_I = mean_i - 2.66 * mean_mr

                    # --- Moving Range Chart Limits ---
                    UCL_MR = 3.267 * mean_mr
                    LCL_MR = 0

                    # --- Plot Individuals Chart ---
                    fig_i, ax_i = plt.subplots(figsize=(10, 4))
                    ax_i.plot(data_imr.index + 1, data_imr, marker='o', color='blue', label='Individuals')
                    ax_i.axhline(mean_i, color='green', linestyle='-', label='Mean')
                    ax_i.axhline(UCL_I, color='red', linestyle='--', label=f'UCL = {UCL_I:.2f}')
                    ax_i.axhline(LCL_I, color='red', linestyle='--', label=f'LCL = {LCL_I:.2f}')
                    ax_i.set_title(f'Individuals Chart for {selected_imr_var}')
                    ax_i.set_xlabel('Observation')
                    ax_i.set_ylabel(selected_imr_var)
                    ax_i.legend()
                    st.pyplot(fig_i)

                    # --- Plot Moving Range Chart ---
                    fig_mr, ax_mr = plt.subplots(figsize=(10, 4))
                    ax_mr.plot(mr.index + 1, mr, marker='o', color='orange', label='Moving Range')
                    ax_mr.axhline(mean_mr, color='green', linestyle='-', label='Mean MR')
                    ax_mr.axhline(UCL_MR, color='red', linestyle='--', label=f'UCL = {UCL_MR:.2f}')
                    ax_mr.set_title(f'Moving Range Chart for {selected_imr_var}')
                    ax_mr.set_xlabel('Observation')
                    ax_mr.set_ylabel('Moving Range')
                    ax_mr.legend()
                    st.pyplot(fig_mr)

                    # --- IMR Summary ---
                    st.markdown("### IMR Summary Statistics")
                    imr_summary = {
                        "Individuals Mean": mean_i,
                        "Moving Range Mean": mean_mr,
                        "UCL (Individuals)": UCL_I,
                        "LCL (Individuals)": LCL_I,
                        "UCL (Moving Range)": UCL_MR,
                        "LCL (Moving Range)": LCL_MR,
                        "Number of Observations": len(data_imr)
                    }
                    st.table(pd.DataFrame(imr_summary, index=[0]).T.rename(columns={0: "Value"}))

                    # build pdf for IMR
                    flowables = []
                    img1 = save_fig_as_png(fig_i, f"imr_individuals_{selected_imr_var}")
                    append_image_flowable(flowables, "Individuals Chart", img1)
                    img2 = save_fig_as_png(fig_mr, f"imr_mr_{selected_imr_var}")
                    append_image_flowable(flowables, "Moving Range Chart", img2)
                    imr_df = pd.DataFrame(imr_summary, index=[0]).T.reset_index().rename(columns={"index":"Statistic", 0:"Value"})
                    flowables.append(Paragraph("IMR Summary", style=None if False else None))
                    flowables.append(df_to_table_flowable(imr_df))
                    flowables.append(Spacer(1,12))

                    generate_section_pdf("IMR Chart", flowables)
                    st.success("IMR PDF generated (IMR_Chart_report.pdf).")
        else:
            st.warning("Please run regression first to load data.")

    # ==========================================================
    # Prediction
    # ==========================================================
    elif section == "Prediction":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header("Predict New Outcome")
            inputs = {}
            for var in indep_vars:
                val = st.number_input(f"Enter {var}", value=float(df[var].mean()))
                inputs[var] = val

            inputs_df = pd.DataFrame([inputs])
            inputs_df = sm.add_constant(inputs_df, has_constant='add')
            with st.spinner("Generating prediction..."):
                prediction = model.predict(inputs_df)[0]
            st.success(f"Predicted {dep_var}: {prediction:.4f}")

            # build pdf for Prediction
            flowables = []
            flowables.append(Paragraph(f"Predicted Variable: {dep_var}", style=None if False else None))
            flowables.append(Spacer(1,6))
            vals_text = ", ".join([f"{k}={v}" for k,v in inputs.items()])
            flowables.append(Paragraph(f"Input Values: {vals_text}", style=None if False else None))
            flowables.append(Spacer(1,6))
            flowables.append(Paragraph(f"Predicted {dep_var}: {prediction:.4f}", style=None if False else None))
            flowables.append(Spacer(1,12))

            generate_section_pdf("Prediction", flowables)
            st.success("Prediction PDF generated (Prediction_report.pdf).")
        else:
            st.warning("Run regression first to make predictions.")

    # ==========================================================
    # Crystal Ball: Define Assumptions
    # ==========================================================
    elif section == "Define Assumptions":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header(" Define Assumptions (Normal Distribution)")
            st.markdown("""
            Enter the **Baseline X**, **Mean**, and **Standard Deviation** for each Independent Variable (X).  
            The Normal Distribution curves will be displayed beside the table.
            """)

            baseline_inputs = {}
            means = {}
            std_devs = {}
            remarks = {}

            st.markdown("### Independent Variables & Assumptions")
            table_col1, table_col2 = st.columns([2, 3])

            with table_col1:
                st.markdown("#### Input Table")
                data_rows = []
                i = 1
                for var in indep_vars:
                    st.markdown(f"**{i}. {var}**")
                    baseline_inputs[var] = st.number_input(f"Baseline X ({var})", key=f"base_{var}", value=float(df[var].mean()))
                    means[var] = st.number_input(f"Mean ({var})", key=f"mean_{var}", value=float(df[var].mean()))
                    std_devs[var] = st.number_input(f"Std Dev ({var})", key=f"std_{var}", value=1.0, min_value=0.0001)
                    remarks[var] = st.text_input(f"Remarks ({var})", key=f"remark_{var}", value="")
                    data_rows.append({
                        "SI No": i,
                        "Independent Variable": var,
                        "Baseline X": baseline_inputs[var],
                        "Mean": means[var],
                        "Std Dev": std_devs[var],
                        "Remarks": remarks[var]
                    })
                    st.markdown("---")
                    i += 1

                baseline_df = pd.DataFrame(data_rows)
                st.dataframe(baseline_df, use_container_width=True)

            with table_col2:
                st.markdown("#### Normal Distribution Graphs")
                fig, axs = plt.subplots(len(indep_vars), 1, figsize=(6, len(indep_vars)*2.5))
                if len(indep_vars) == 1:
                    axs = [axs]  # Ensure iterable

                for idx, var in enumerate(indep_vars):
                    mean_val = means[var]
                    std_val = std_devs[var]
                    x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 200)
                    y = stats.norm.pdf(x, mean_val, std_val)
                    axs[idx].plot(x, y, color='green', lw=2)
                    axs[idx].fill_between(x, y, color='lightgreen', alpha=0.6)
                    axs[idx].set_title(f"{var} (μ={mean_val:.2f}, σ={std_val:.2f})")
                    axs[idx].set_xlabel("Value")
                    axs[idx].set_ylabel("Probability")
                plt.tight_layout()
                st.pyplot(fig)

            const_val = float(model.params.get("const", 0.0))
            st.markdown(f"**Constant (Intercept):** {const_val:.4f}")

            input_df = pd.DataFrame([baseline_inputs])
            input_df = sm.add_constant(input_df, has_constant="add")
            for col in model.params.index:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[model.params.index]
            predicted_y = float(np.dot(input_df.values, model.params.values))
            st.success(f"**Predicted {dep_var} based on Baseline X values:** {predicted_y:.4f}")

            # build pdf for assumptions
            flowables = []
            flowables.append(df_to_table_flowable(baseline_df))
            flowables.append(Spacer(1,12))
            img = save_fig_as_png(fig, "assumptions_dist")
            append_image_flowable(flowables, "Normal Distribution Graphs", img)
            flowables.append(Paragraph(f"Predicted {dep_var} based on Baseline: {predicted_y:.4f}", style=None if False else None))

            generate_section_pdf("Define Assumptions", flowables)
            st.success(" Define Assumptions PDF generated (Define_Assumptions_report.pdf).")
        else:
            st.warning("Run regression first to define assumptions.")

    # ==========================================================
    # Crystal Ball: What-If Analysis
    # ==========================================================
    elif section == "What-If Analysis":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header(" What-If Analysis (Proposed Values)")
            cols = st.columns([0.1, 0.5, 0.4])
            cols[0].markdown("**SI No**")
            cols[1].markdown("**Independent Variable (X)**")
            cols[2].markdown("**Proposed Value for What-If Analysis**")

            proposed_values = {}
            for i, var in enumerate(indep_vars, start=1):
                c0, c1, c2 = st.columns([0.1, 0.5, 0.4])
                c0.write(i)
                c1.write(var)
                proposed_values[var] = c2.number_input(f"proposed_{var}", key=f"proposed_{var}", value=float(df[var].mean()))

            st.markdown("---")
            target_mean = st.number_input("Target Mean", value=float(df[dep_var].mean()))
            lsl = st.number_input("Lower Spec Limit (LSL)", value=float(df[dep_var].quantile(0.05)))
            usl = st.number_input("Upper Spec Limit (USL)", value=float(df[dep_var].quantile(0.95)))
            st.markdown("---")

            input_df = pd.DataFrame([proposed_values])
            input_df = sm.add_constant(input_df, has_constant='add')
            for col in model.params.index:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[model.params.index]
            predicted_mean = float(np.dot(input_df.values, model.params.values))

            residuals = model.resid
            sd_resid = np.std(residuals, ddof=1)
            n = len(df)
            se = sd_resid / np.sqrt(n)
            lpi = predicted_mean - 1.96 * se
            upi = predicted_mean + 1.96 * se

            result_df = pd.DataFrame({
                "Metric": [
                    "Prediction Mean (Ŷ)", "Target Mean",
                    "Lower Prediction Interval (LPI)", "Upper Prediction Interval (UPI)",
                    "Lower Spec Limit (LSL)", "Upper Spec Limit (USL)",
                    "SD of Residuals", "No. of Data Points", "Standard Error (SE)"
                ],
                "Value": [
                    round(predicted_mean, 5), round(target_mean, 5),
                    round(lpi, 5), round(upi, 5),
                    round(lsl, 5), round(usl, 5),
                    round(sd_resid, 8), int(n), round(se, 8)
                ]
            })

            st.table(result_df)
            st.success(f"Predicted {dep_var}: **{predicted_mean:.4f}** (95 % PI: {lpi:.4f} – {upi:.4f})")

            # Build PDF for What-If
            flowables = []
            prop_df = pd.DataFrame([proposed_values]).T.reset_index().rename(columns={"index":"Variable", 0:"Proposed Value"})
            flowables.append(Paragraph("Proposed Values", style=None if False else None))
            flowables.append(df_to_table_flowable(prop_df))
            flowables.append(Spacer(1,12))
            flowables.append(Paragraph("Prediction Results", style=None if False else None))
            flowables.append(df_to_table_flowable(result_df))
            generate_section_pdf("What-If Analysis", flowables)
            st.success("What-If PDF generated (What-If_Analysis_report.pdf).")
        else:
            st.warning("Run regression first to perform What-If Analysis.")

    # ==========================================================
    # Crystal Ball: Forecasting & Sensitivity
    # ==========================================================
    elif section == "Forecasting & Sensitivity":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header(" Forecasting & Sensitivity Analysis ")

            lsl = st.number_input("Lower Specification Limit (LSL)", value=float(df[dep_var].quantile(0.05)))
            usl = st.number_input("Upper Specification Limit (USL)", value=float(df[dep_var].quantile(0.95)))
            target = st.number_input("Target Value", value=float(df[dep_var].mean()))
            num_sim = st.slider("Number of Forecast Simulations", 500, 50000, 1000, step=500)

            st.markdown("---")

            np.random.seed(42)
            X_means = df[indep_vars].mean()
            X_stds = df[indep_vars].std()

            simulated_data = {}
            for var in indep_vars:
                simulated_data[var] = np.random.normal(
                    loc=X_means[var], scale=X_stds[var], size=num_sim
                )

            sim_df = pd.DataFrame(simulated_data)
            sim_df = sm.add_constant(sim_df, has_constant='add')

            coeffs = model.params
            sim_df["Forecast_Y"] = np.dot(sim_df[coeffs.index], coeffs)
            y_forecast = sim_df["Forecast_Y"]

            mean_y = np.mean(y_forecast)
            std_y = np.std(y_forecast, ddof=1)

            cp = (usl - lsl) / (6 * std_y) if std_y > 0 else np.nan
            cpk_lower = (mean_y - lsl) / (3 * std_y)
            cpk_upper = (usl - mean_y) / (3 * std_y)
            cpk = min(cpk_lower, cpk_upper)
            zlsl = (mean_y - lsl) / std_y
            zusl = (usl - mean_y) / std_y
            ppm_below = stats.norm.cdf((lsl - mean_y) / std_y) * 1e6
            ppm_above = (1 - stats.norm.cdf((usl - mean_y) / std_y)) * 1e6
            ppm_total = ppm_below + ppm_above
            certainty = np.mean((y_forecast >= lsl) & (y_forecast <= usl)) * 100

            metrics = {
                "Mean": mean_y,
                "Std Dev": std_y,
                "Cp": cp,
                "Cpk-lower": cpk_lower,
                "Cpk-upper": cpk_upper,
                "Cpk": cpk,
                "Z-LSL": zlsl,
                "Z-USL": zusl,
                "Z-total": min(zlsl, zusl),
                "PPM-below": ppm_below,
                "PPM-above": ppm_above,
                "PPM-total": ppm_total,
                "Certainty %": certainty,
                "LSL": lsl,
                "USL": usl,
                "Target": target
            }

            st.table(pd.DataFrame(metrics, index=["Value"]).T.rename(columns={"Value":"Value"}))

            # Forecast distribution chart
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(y_forecast, bins=40, color="skyblue", kde=True, ax=ax)
            ax.axvline(lsl, color='red', linestyle='--', label=f"LSL = {lsl:.3f}")
            ax.axvline(usl, color='red', linestyle='--', label=f"USL = {usl:.3f}")
            ax.axvline(target, color='green', linestyle='--', label=f"Target = {target:.3f}")
            ax.set_title(f"Forecast Distribution for {dep_var}")
            ax.set_xlabel(dep_var)
            ax.legend()
            st.pyplot(fig)

            # Sensitivity
            Y = y_forecast.values
            var_Y = np.var(Y, ddof=1)

            if var_Y <= 0 or np.isnan(var_Y):
                st.warning("Cannot compute sensitivity (no forecast variance).")
            else:
                spearman_rhos = []
                for var in indep_vars:
                    X = sim_df[var].values
                    if np.allclose(X, X[0]):
                        spearman_rhos.append(0.0)
                        continue
                    rho, _ = stats.spearmanr(X, Y)
                    if np.isnan(rho):
                        rho = 0.0
                    spearman_rhos.append(rho)

                spearman_rhos = np.array(spearman_rhos, dtype=float)
                rho_sq = spearman_rhos ** 2
                total_rho_sq = rho_sq.sum()
                if total_rho_sq == 0:
                    contrib_percent = np.sign(spearman_rhos) * (np.abs(spearman_rhos) / (np.abs(spearman_rhos).sum() if np.abs(spearman_rhos).sum() > 0 else 1.0)) * 100
                else:
                    contrib_percent = (rho_sq / total_rho_sq) * 100
                    contrib_percent = contrib_percent * np.sign(spearman_rhos)

                sens_df = pd.DataFrame({
                    "Variable": indep_vars,
                    "Contribution (%)": contrib_percent
                })

                sens_df = sens_df.sort_values("Contribution (%)", ascending=True).reset_index(drop=True)

                fig_sens, ax_sens = plt.subplots(figsize=(10, max(3, 0.6 * len(indep_vars))))
                colors_list = ['#2E86C1' if v > 0 else '#E74C3C' for v in sens_df["Contribution (%)"]]
                ax_sens.barh(sens_df["Variable"], sens_df["Contribution (%)"], color=colors_list, edgecolor='black', height=0.6)
                ax_sens.axvline(0, color='black', linewidth=1)
                ax_sens.set_xlabel("Contribution to Variance (%)")
                ax_sens.set_title("Sensitivity: Contribution to Variance (Crystal Ball style)")

                for i, v in enumerate(sens_df["Contribution (%)"]):
                    if v >= 0:
                        ax_sens.text(v + 0.5, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold')
                    else:
                        ax_sens.text(v - 0.5, i, f"{v:.1f}%", va='center', ha='right', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig_sens)

                display_df = sens_df.copy()
                display_df["Abs (%)"] = display_df["Contribution (%)"].abs()
                display_df = display_df.sort_values("Abs (%)", ascending=False)[["Variable", "Contribution (%)"]]
                st.table(display_df.set_index("Variable"))

            # Build forecasting PDF
            flowables = []
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric","Value"])
            flowables.append(df_to_table_flowable(metrics_df))
            flowables.append(Spacer(1,12))
            hist_img = save_fig_as_png(fig, "forecast_dist")
            append_image_flowable(flowables, "Forecast Distribution", hist_img)
            sens_img = save_fig_as_png(fig_sens, "sensitivity")
            append_image_flowable(flowables, "Sensitivity (Tornado)", sens_img)

            generate_section_pdf("Forecasting & Sensitivity", flowables)
            st.success(" Forecasting PDF generated (Forecasting_&_Sensitivity_report.pdf).")
        else:
            st.warning("Run regression first to perform forecasting and sensitivity analysis.")

    # === After each section: list existing per-section reports and merge option ===
    st.markdown("---")
    st.markdown("### Generated Section Reports (this folder)")
    # show fixed list order (only those that exist)
    fixed_names = [
        "Upload_&_Normality_report.pdf",
        "Regression_report.pdf",
        "Correlation_&_p-values_report.pdf",
        "IMR_Chart_report.pdf",
        "Prediction_report.pdf",
        "Define_Assumptions_report.pdf",
        "What-If_Analysis_report.pdf",
        "Forecasting_&_Sensitivity_report.pdf"
    ]
    import os

any_exists = False
for fname in fixed_names:
    if os.path.exists(fname):
        any_exists = True
        display_name = os.path.basename(fname)  #  only the filename
        st.write(display_name)
        with open(fname, "rb") as f:
            st.download_button(
                label=f" Download {display_name}",
                data=f,
                file_name=display_name,
                mime="application/pdf"
            )

if not any_exists:
    st.write("No section reports generated yet. Run a section to create its PDF.")

st.markdown("---")
st.header(" Final Report Download")

if st.button(" Merge All Section Reports and Download (timestamped)"):
    merged_file = merge_section_pdfs()
    if merged_file:
        display_name = os.path.basename(merged_file)  #  only the short file name
        clean_name = display_name.replace("Minitab_crystalball_Report", "minitab_crystal_ball_report")
        st.success(f" Merged file created: {clean_name}")
        with open(merged_file, "rb") as f:
            st.download_button(
                label=f" Download Combined Report ({clean_name})",
                data=f,
                file_name=clean_name,  #  short + clean filename
                mime="application/pdf"
            )
    else:
        st.warning("No individual section reports found to merge. Generate section PDFs first.")


# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'> Performance Dashboard | Built by Web Synergies</p>",
    unsafe_allow_html=True
)
