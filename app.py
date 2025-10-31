import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy import stats

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

        /* --- Sidebar Custom Styling --- */
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

# --- Page Title ---
st.title("Process Performance Model")

# --- Sidebar Navigation (Collapsible like Minitab / Crystal Ball) ---
st.sidebar.markdown("<h2 style='color:#1f2937;'>Dashboard Sections</h2>", unsafe_allow_html=True)

# --- Main Navigation Group ---
nav_group = st.sidebar.radio("Select Mode", ["Minitab", "Crystal Ball"], horizontal=True)

# --- Initialize variable ---
section = None

# --- Minitab Section ---
if nav_group == "Minitab":
    section = st.sidebar.radio(
        "Minitab Modules",
        (
            "Upload & Regression",
            "Residuals & Normality",
            "Correlation & p-values",
            "IMR Chart",
            "Prediction"
        ),
        label_visibility="collapsed"
    )

# --- Crystal Ball Section ---
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

#section = minitab_section if minitab_section else crystalball_section


# --- File Upload Section ---
uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded:
    with st.spinner("Loading dataset..."):
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # --- Regression Analysis Section ---
    if section == "Upload & Regression":
        st.header("Regression Analysis")

        # Restore previously selected variables if available
        prev_dep = st.session_state.get("dep_var", None)
        prev_indep = st.session_state.get("indep_vars", None)

        # Dependent variable selectbox
        dep_var = st.selectbox(
            "Select dependent variable (Y)",
            num_cols,
            index=num_cols.index(prev_dep) if prev_dep in num_cols else 0,
            key="dep_var_select"
        )

        # Independent variables multiselect (excluding selected dependent)
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

        # Store current selections
        st.session_state["dep_var"] = dep_var
        st.session_state["indep_vars"] = indep_vars

        if indep_vars:
            with st.spinner("Running regression analysis..."):
                # Fit full model (matrix API is fine for coefficients/residuals)
                X = df[indep_vars]
                X = sm.add_constant(X)
                y = df[dep_var]
                model = sm.OLS(y, X).fit()

                # Save residuals
                df["Residuals"] = np.nan
                df.loc[y.index, "Residuals"] = model.resid

            # Preview
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            #correlation charts
            with st.spinner("Generating correlation charts..."):
                st.subheader("Correlation Outliers Chart")
                selected_corr_vars = [dep_var] + indep_vars
                if len(selected_corr_vars) >= 2:
                    #  Use sampling for speed
                    sample_df = df[selected_corr_vars].sample(min(len(df), 1000), random_state=42)
                    fig_corr = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
                    st.pyplot(fig_corr.fig)
                else:
                    st.info("Select at least two numeric variables to generate the matrix plot.")

            # Regression Summary
            st.subheader("Regression Summary")

            # Convert the summary to a clean HTML-style box
            summary_html = model.summary().as_html()

            # Apply a cleaner and more compact style
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


            # ============================
            # ANOVA (Minitab-style, matching reference)
            # ============================
            try:
                # imports required
                from statsmodels.formula.api import ols
                from scipy import stats as spstats

                # Full model built with formula API for convenience (but we will compute SS manually)
                # Use Q() to safely reference problematic column names in formula
                formula_full = f"Q('{dep_var}') ~ " + " + ".join([f"Q('{v}')" for v in indep_vars])
                anova_model = ols(formula_full, data=df).fit()

                # Compute SSR (Regression sum of squares), SSE, SST
                fitted = anova_model.fittedvalues
                yvals = anova_model.model.endog
                n = len(yvals)

                SSR_full = np.sum((fitted - yvals.mean()) ** 2)    # Regression SS (explained)
                SSE = np.sum(anova_model.resid ** 2)              # Error SS (residual)
                SST = SSR_full + SSE                              # Total SS

                # Degrees of freedom
                df_reg = len(indep_vars)       # regression df (number of predictors)
                df_resid = int(anova_model.df_resid)
                df_total = df_reg + df_resid

                # MSE (error mean square)
                MSE = SSE / df_resid if df_resid > 0 else np.nan

                # For each predictor: compute contribution as SSR_full - SSR_reduced
                term_rows = []
                for term in indep_vars:
                    # Build reduced formula without current term
                    other_terms = [v for v in indep_vars if v != term]
                    if other_terms:
                        formula_reduced = f"Q('{dep_var}') ~ " + " + ".join([f"Q('{v}')" for v in other_terms])
                    else:
                        # reduced model is intercept-only
                        formula_reduced = f"Q('{dep_var}') ~ 1"

                    reduced_model = ols(formula_reduced, data=df).fit()
                    fitted_reduced = reduced_model.fittedvalues
                    SSR_reduced = np.sum((fitted_reduced - yvals.mean()) ** 2)

                    # SS contributed by this term (Type: drop-in-SSR => same as Type III contribution)
                    SS_term = SSR_full - SSR_reduced
                    DF_term = 1  # each predictor contributes 1 df in standard regression
                    MS_term = SS_term / DF_term if DF_term != 0 else np.nan
                    F_term = (MS_term / MSE) if MSE != 0 and not np.isnan(MS_term) else np.nan
                    p_term = spstats.f.sf(F_term, DF_term, df_resid) if not np.isnan(F_term) else np.nan

                    term_rows.append({
                        "Source": term,
                        "DF": DF_term,
                        "Adj SS": SS_term,
                        "Adj MS": MS_term,
                        "F-Value": F_term,
                        "P-Value": p_term
                    })

                # Create DataFrame for terms
                terms_df = pd.DataFrame(term_rows)

                # Regression main row (aggregate)
                regression_row = pd.DataFrame([{
                    "Source": "Regression",
                    "DF": df_reg,
                    "Adj SS": SSR_full,
                    "Adj MS": SSR_full / df_reg if df_reg > 0 else np.nan,
                    "F-Value": (SSR_full / df_reg) / MSE if MSE != 0 else np.nan,
                    "P-Value": spstats.f.sf((SSR_full / df_reg) / MSE, df_reg, df_resid) if MSE != 0 else np.nan
                }])

                # Error row
                error_row = pd.DataFrame([{
                    "Source": "Error",
                    "DF": df_resid,
                    "Adj SS": SSE,
                    "Adj MS": MSE,
                    "F-Value": np.nan,
                    "P-Value": np.nan
                }])

                # Total row
                total_row = pd.DataFrame([{
                    "Source": "Total",
                    "DF": df_total,
                    "Adj SS": SST,
                    "Adj MS": np.nan,
                    "F-Value": np.nan,
                    "P-Value": np.nan
                }])

                # Indent term names for subrows
                terms_df_display = terms_df.copy()
                terms_df_display["Source"] = "   " + terms_df_display["Source"]

                # Combine into final display table: Regression, sub-terms, Error, Total
                anova_display = pd.concat([regression_row, terms_df_display, error_row, total_row], ignore_index=True)

                # Format numeric columns but keep NaNs so formatting doesn't break
                # We'll display em-dash for NaN values using format lambdas.
                def highlight_rows(row):
                    if row['Source'].strip() in ['Regression', 'Error', 'Total']:
                        return ['font-weight: bold; background-color: #f5f5f5'] * len(row)
                    elif isinstance(row['Source'], str) and row['Source'].startswith("   "):
                        return ['background-color: #fafafa'] * len(row)
                    else:
                        return [''] * len(row)

                # Display the ANOVA table
                st.subheader("Analysis of Variance (ANOVA)")
                st.dataframe(
                    anova_display[['Source', 'DF', 'Adj SS', 'Adj MS', 'F-Value', 'P-Value']]
                    .style.apply(highlight_rows, axis=1)
                    .format({
                        'DF': lambda x: f"{int(x)}" if pd.notna(x) and isinstance(x, (int, np.integer, float, np.floating)) else "—",
                        'Adj SS': lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else "—",
                        'Adj MS': lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else "—",
                        'F-Value': lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else "—",
                        'P-Value': lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float, np.integer, np.floating)) else "—"
                    })
                )

            except Exception as e:
                st.warning(f" Could not compute ANOVA table: {e}")

            # Regression Equation
            st.subheader("Regression Equation")
            # Build equation from model.params (handles special names since we used formula internally)
            params = model.params
            # If 'Intercept' or 'const' present, find the intercept name:
            intercept_name = None
            for k in params.index:
                if k.lower() in ['intercept', 'const']:
                    intercept_name = k
                    break

            eq_parts = []
            if intercept_name:
                eq_parts.append(f"{params[intercept_name]:.4f}")
            else:
                # no intercept name found: use first numeric param as intercept if necessary
                pass

            for v in indep_vars:
                coef = params.get(v, params.get(f"Q('{v}')", np.nan))
                eq_parts.append(f"({coef:.4f} × {v})")

            equation = f"{dep_var} = " + " + ".join(eq_parts)
            st.markdown(f"**{equation}**")

            # R² Metrics
            colA, colB = st.columns(2)
            colA.metric("R² (Goodness of Fit)", f"{model.rsquared:.3f}")
            colB.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")

            # Save for later
            st.session_state["df"] = df
            st.session_state["model"] = model

    # --- Residuals & Normality Section ---
    elif section == "Residuals & Normality":
        if "df" in st.session_state:
            df = st.session_state["df"]

            st.header("Distribution and Normality Analysis")

            selected_vars = st.multiselect(
                "Select variables for analysis",
                df.select_dtypes(include=np.number).columns.tolist(),
                default=[]
            )

            if selected_vars:
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

                        # --- Anderson–Darling Normality Test (Minitab style) ---
                        ad_stat, approx_p = anderson_darling_minitab(data)
                        norm_result = "Data appears NORMAL (p > 0.05)" if approx_p > 0.05 else "Data is NOT normal (p < 0.05)"

                        # --- Confidence Intervals ---
                        ci_mean = stats.t.interval(0.95, n-1, loc=mean_val, scale=std_val/np.sqrt(n))
                        ci_median = np.percentile(data, [2.5, 97.5])
                        ci_std = (
                            std_val * np.sqrt((n - 1) / stats.chi2.ppf(0.975, n - 1)),
                            std_val * np.sqrt((n - 1) / stats.chi2.ppf(0.025, n - 1))
                        )

                        q1, median, q3 = np.percentile(data, [25, 50, 75])
                        data_min, data_max = np.min(data), np.max(data)

                        # --- Visualization (Histogram + Boxplot) ---
                        fig_main, (ax1, ax_box) = plt.subplots(
                            2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]}
                        )

                        sns.histplot(data, kde=False, bins=10, color="skyblue", ax=ax1)
                        xmin, xmax = ax1.get_xlim()
                        x = np.linspace(xmin, xmax, 100)
                        p = stats.norm.pdf(x, mean_val, std_val)
                        ax1.plot(x, p * len(data) * (xmax - xmin) / 10, 'r-', lw=2)
                        ax1.set_title(f"{selected_var} Distribution with Normal Curve")
                        ax1.set_xlabel(selected_var)
                        ax1.set_ylabel("Frequency")

                        sns.boxplot(x=data, color="lightblue", ax=ax_box, orient="h")
                        ax_box.set_xlabel(selected_var)

                        # --- Summary Text ---
                        summary_text = (
                            f"**Anderson–Darling Normality Test (Minitab-style)**\n"
                            f"A-Squared*: {ad_stat:.3f}\n"
                            f"Approx. P-Value: {approx_p:.3f}\n"
                            f"→ {norm_result}\n\n"
                            f"**Descriptive Statistics**\n"
                            f"Mean: {mean_val:.5f}\n"
                            f"StDev: {std_val:.5f}\n"
                            f"Variance: {variance:.5f}\n"
                            f"Skewness: {skew:.5f}\n"
                            f"Kurtosis: {kurt:.5f}\n"
                            f"N: {n}\n\n"
                            f"**95% Confidence Intervals**\n"
                            f"Mean: {ci_mean[0]:.5f} to {ci_mean[1]:.5f}\n"
                            f"Median: {ci_median[0]:.5f} to {ci_median[1]:.5f}\n"
                            f"StDev: {ci_std[0]:.5f} to {ci_std[1]:.5f}"
                        )

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.pyplot(fig_main)
                        with col2:
                            st.markdown("### Summary Statistics")
                            st.text(summary_text)
                            st.markdown("### Outlier Summary")
                            st.table(pd.DataFrame({
                                "Statistic": ["Minimum", "1st Quartile (Q1)", "Median", "3rd Quartile (Q3)", "Maximum"],
                                "Value": [data_min, q1, median, q3, data_max]
                            }))
            else:
                st.info("Select one or more numeric variables to begin analysis.")
        else:
            st.warning("Upload a dataset to begin analysis.")

    # --- Correlation & p-values Section ---
    elif section == "Correlation & p-values":
        if "model" in st.session_state:
            df = st.session_state["df"]
            model = st.session_state["model"]
            indep_vars = st.session_state["indep_vars"]

            st.header("Correlation Heatmap and Variable Significance")

            # --- Correlation Heatmap ---
            with st.spinner("Generating correlation heatmap..."):
                fig3, ax3 = plt.subplots()
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
                st.pyplot(fig3)

            # --- p-Value Significance Table ---
            with st.spinner("Computing variable significance..."):
                pvals = model.pvalues.drop("const", errors="ignore")
                sig = pd.DataFrame({"Variable": pvals.index, "p-Value": pvals.values})
                sig["Significant?"] = sig["p-Value"].apply(lambda x: "Yes" if x < 0.05 else "No")
                st.subheader("Regression Variable Significance (p-Values)")
                st.dataframe(sig, use_container_width=True)

                # --- Bar Chart for p-values ---
                fig, ax = plt.subplots()
                sns.barplot(x="Variable", y="p-Value", data=sig, ax=ax)
                ax.axhline(0.05, color="red", linestyle="--")
                ax.set_title("p-Value Significance Threshold (0.05)")
                st.pyplot(fig)

            # --- Variance Inflation Factor (VIF) Calculation ---
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
        else:
            st.warning("Run regression first to view significance and multicollinearity charts.")

    # --- IMR Chart Section ---
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
                    ax_mr.axhline(LCL_MR, color='red', linestyle='--', label=f'LCL = {LCL_MR:.2f}')
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
        else:
            st.warning("Please run regression first to load data.")
    # --- Prediction Section ---
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
        else:
            st.warning("Run regression first to make predictions.")
# --- Forecasting / Define Assumptions (Crystal Ball Style) ---
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

            # Create storage dictionaries
            baseline_inputs = {}
            means = {}
            std_devs = {}
            remarks = {}

            # --- Table headers ---
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

            # --- Right side: Normal Distribution Graphs ---
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

            # --- Constant term (intercept) ---
            const_val = float(model.params.get("const", 0.0))
            st.markdown(f"**Constant (Intercept):** {const_val:.4f}")

            # --- Compute Predicted Y using Baseline X values ---
            input_df = pd.DataFrame([baseline_inputs])
            input_df = sm.add_constant(input_df, has_constant="add")

            # Align model parameters with input columns
            for col in model.params.index:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[model.params.index]

            predicted_y = float(np.dot(input_df.values, model.params.values))
            st.success(f"**Predicted {dep_var} based on Baseline X values:** {predicted_y:.4f}")

        else:
            st.warning("Run regression first to define assumptions.")

 # --- Step 2: What-If Analysis ---
    elif section == "What-If Analysis":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header(" What-If Analysis (Proposed Values)")
            st.markdown("Enter proposed values for each independent variable (X) to predict the output (Y).")

            # Table layout for entering proposed Xs
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

            # Target + Spec limits
            st.subheader("Target & Specification Limits")
            target_mean = st.number_input("Target Mean", value=float(df[dep_var].mean()))
            lsl = st.number_input("Lower Spec Limit (LSL)", value=float(df[dep_var].quantile(0.05)))
            usl = st.number_input("Upper Spec Limit (USL)", value=float(df[dep_var].quantile(0.95)))

            st.markdown("---")

            # Prediction computation
            st.subheader("Prediction Analysis Results")

            # Prepare input for prediction
            input_df = pd.DataFrame([proposed_values])
            input_df = sm.add_constant(input_df, has_constant='add')
            for col in model.params.index:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[model.params.index]

            predicted_mean = float(np.dot(input_df.values, model.params.values))

            # Residual & error stats
            residuals = model.resid
            sd_resid = np.std(residuals, ddof=1)
            n = len(df)
            se = sd_resid / np.sqrt(n)

            # 95 % prediction interval
            lpi = predicted_mean - 1.96 * se
            upi = predicted_mean + 1.96 * se

            # Summary table
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

        else:
            st.warning("Run regression first to perform What-If Analysis.")
# --- Step 3: Forecasting & Sensitivity (Crystal Ball Style) ---
    elif section == "Forecasting & Sensitivity":
        if "model" in st.session_state:
            model = st.session_state["model"]
            dep_var = st.session_state["dep_var"]
            indep_vars = st.session_state["indep_vars"]
            df = st.session_state["df"]

            st.header(" Forecasting & Sensitivity Analysis ")

            # --- Step 3.1: Define Forecast Parameters ---
            st.subheader("Define Forecast Parameters")
            st.markdown("Enter LSL, USL and Target for your forecast variable just like in Crystal Ball.")

            lsl = st.number_input("Lower Specification Limit (LSL)", value=float(df[dep_var].quantile(0.05)))
            usl = st.number_input("Upper Specification Limit (USL)", value=float(df[dep_var].quantile(0.95)))
            target = st.number_input("Target Value", value=float(df[dep_var].mean()))
            num_sim = st.slider("Number of Forecast Simulations", 500, 50000, 1000, step=500)

            st.markdown("---")

            # --- Step 3.2: Run Forecast Simulation ---
            st.subheader("Run Forecast Simulation")

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

            # --- Step 3.3: Compute Metrics (Crystal Ball Style) ---
            st.subheader("Forecast Summary Statistics")

            mean_y = np.mean(y_forecast)
            std_y = np.std(y_forecast, ddof=1)

            cp = (usl - lsl) / (6 * std_y) if std_y > 0 else np.nan
            cpk_lower = (mean_y - lsl) / (3 * std_y)
            cpk_upper = (usl - mean_y) / (3 * std_y)
            cpk = min(cpk_lower, cpk_upper)
            zlsl = (mean_y - lsl) / std_y
            zusl = (usl - mean_y) / std_y
            ztotal = min(zlsl, zusl)
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
                "Z-total": ztotal,
                "PPM-below": ppm_below,
                "PPM-above": ppm_above,
                "PPM-total": ppm_total,
                "Certainty %": certainty,
                "LSL": lsl,
                "USL": usl,
                "Target": target
            }

            st.table(pd.DataFrame(metrics, index=["Forecast Value"]).T.rename(columns={"Forecast Value": "Value"}))

            # --- Step 3.4: Forecast Distribution Chart ---
            st.subheader("Forecast Distribution Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(y_forecast, bins=40, color="skyblue", kde=True, ax=ax)
            ax.axvline(lsl, color='red', linestyle='--', label=f"LSL = {lsl:.3f}")
            ax.axvline(usl, color='red', linestyle='--', label=f"USL = {usl:.3f}")
            ax.axvline(target, color='green', linestyle='--', label=f"Target = {target:.3f}")
            ax.set_title(f"Forecast Distribution for {dep_var}")
            ax.set_xlabel(dep_var)
            ax.legend()
            st.pyplot(fig)

            st.markdown(f" **Certainty (LSL ≤ Y ≤ USL): {certainty:.2f}%**")

            st.markdown("---")

            # ============================================================
            # Sensitivity Analysis — Crystal Ball method (Spearman^2 normalization)
            # ============================================================
            st.subheader("Sensitivity Analysis — Contribution to Variance (Crystal Ball-style)")

            Y = y_forecast.values
            var_Y = np.var(Y, ddof=1)

            if var_Y <= 0 or np.isnan(var_Y):
                st.warning("Cannot compute sensitivity (no forecast variance).")
            else:
                # compute Spearman rho for each input against forecast Y
                spearman_rhos = []
                for var in indep_vars:
                    X = sim_df[var].values
                    # if no variance in X, rho is zero
                    if np.allclose(X, X[0]):
                        spearman_rhos.append(0.0)
                        continue
                    rho, _ = stats.spearmanr(X, Y)
                    if np.isnan(rho):
                        rho = 0.0
                    spearman_rhos.append(rho)

                spearman_rhos = np.array(spearman_rhos, dtype=float)

                # squared rhos (proportion of rank-variance explained)
                rho_sq = spearman_rhos ** 2

                total_rho_sq = rho_sq.sum()
                if total_rho_sq == 0:
                    # fallback: use absolute rho normalization
                    contrib_percent = np.sign(spearman_rhos) * (np.abs(spearman_rhos) / (np.abs(spearman_rhos).sum() if np.abs(spearman_rhos).sum() > 0 else 1.0)) * 100
                else:
                    # normalize squared rho to sum 100 and preserve sign from original rho
                    contrib_percent = (rho_sq / total_rho_sq) * 100
                    contrib_percent = contrib_percent * np.sign(spearman_rhos)

                sens_df = pd.DataFrame({
                    "Variable": indep_vars,
                    "Contribution (%)": contrib_percent
                })

                # sort for plotting (so largest magnitude at top)
                sens_df = sens_df.sort_values("Contribution (%)", ascending=True).reset_index(drop=True)

                # Tornado-style horizontal bar plot that matches Crystal Ball visuals
                fig_sens, ax_sens = plt.subplots(figsize=(10, max(3, 0.6 * len(indep_vars))))
                colors = ['#2E86C1' if v > 0 else '#E74C3C' for v in sens_df["Contribution (%)"]]
                ax_sens.barh(sens_df["Variable"], sens_df["Contribution (%)"], color=colors, edgecolor='black', height=0.6)
                ax_sens.axvline(0, color='black', linewidth=1)
                ax_sens.set_xlabel("Contribution to Variance (%)")
                ax_sens.set_title("Sensitivity: Contribution to Variance (Crystal Ball style)")

                # annotate percentages near bars
                for i, v in enumerate(sens_df["Contribution (%)"]):
                    if v >= 0:
                        ax_sens.text(v + 0.5, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold')
                    else:
                        ax_sens.text(v - 0.5, i, f"{v:.1f}%", va='center', ha='right', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig_sens)

                # also show numeric table sorted descending by absolute contribution (like CB)
                display_df = sens_df.copy()
                display_df["Abs (%)"] = display_df["Contribution (%)"].abs()
                display_df = display_df.sort_values("Abs (%)", ascending=False)[["Variable", "Contribution (%)"]]
                st.table(display_df.set_index("Variable"))

        else:
            st.warning("Run regression first to perform forecasting and sensitivity analysis.")


# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Process Performance Dashboard | Built by Web Synergies</p>",
    unsafe_allow_html=True

)
