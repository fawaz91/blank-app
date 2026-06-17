# app.py
# Enhanced Survival Distribution Fitting Agent
# Inputs:
#   1) IPD-style data: time, event
#   2) Digitised KM data + number-at-risk table
#
# Outputs:
#   - Reconstructed pseudo-IPD
#   - Parametric survival fits
#   - AIC/BIC ranking
#   - Estimated distribution parameters
#   - Excel / TreeAge / R survival-function formulas
#   - User-selected survival curves
#   - Weighted survival curve between two selected distributions
#   - Cycle transition probabilities
#   - Monthly-to-annual transition probability conversion
#   - Excel export

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from lifelines import (
    KaplanMeierFitter,
    ExponentialFitter,
    WeibullFitter,
    LogNormalFitter,
    LogLogisticFitter,
    GeneralizedGammaFitter,
)

warnings.filterwarnings("ignore")


# -----------------------------
# Utility functions
# -----------------------------

def normalise_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def read_uploaded_table(uploaded_file):
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV or Excel file.")

    if df is None or len(df) == 0:
        raise ValueError("Uploaded file is empty or could not be parsed.")

    return df


def clean_survival_values(km_data):
    km_data = km_data.copy()
    if km_data["survival"].max() > 1.5:
        km_data["survival"] = km_data["survival"] / 100.0

    km_data["survival"] = km_data["survival"].clip(lower=0.000001, upper=1.0)
    return km_data


def validate_ipd(ipd):
    ipd = normalise_columns(ipd)

    required = {"time", "event"}
    missing = required - set(ipd.columns)

    if missing:
        raise ValueError(f"IPD file must contain columns: {required}. Missing: {missing}")

    ipd = ipd[["time", "event"]].copy()
    ipd["time"] = pd.to_numeric(ipd["time"], errors="coerce")
    ipd["event"] = pd.to_numeric(ipd["event"], errors="coerce")

    ipd = ipd.dropna()
    ipd["event"] = ipd["event"].astype(int)
    ipd = ipd[ipd["event"].isin([0, 1])]
    ipd = ipd[ipd["time"] >= 0]

    if len(ipd) == 0:
        raise ValueError("No valid IPD rows found.")

    return ipd.sort_values("time").reset_index(drop=True)


def validate_km_and_risk(km_data, risk_table):
    km_data = normalise_columns(km_data)
    risk_table = normalise_columns(risk_table)

    required_km = {"time", "survival"}
    required_risk = {"time", "n_risk"}

    missing_km = required_km - set(km_data.columns)
    missing_risk = required_risk - set(risk_table.columns)

    if missing_km:
        raise ValueError(f"KM file must contain columns: {required_km}. Missing: {missing_km}")

    if missing_risk:
        raise ValueError(f"Risk table must contain columns: {required_risk}. Missing: {missing_risk}")

    km_data = km_data[["time", "survival"]].copy()
    risk_table = risk_table[["time", "n_risk"]].copy()

    km_data["time"] = pd.to_numeric(km_data["time"], errors="coerce")
    km_data["survival"] = pd.to_numeric(km_data["survival"], errors="coerce")
    risk_table["time"] = pd.to_numeric(risk_table["time"], errors="coerce")
    risk_table["n_risk"] = pd.to_numeric(risk_table["n_risk"], errors="coerce")

    km_data = km_data.dropna()
    risk_table = risk_table.dropna()

    if len(km_data) == 0:
        raise ValueError("KM data file contains no valid rows after cleaning.")

    if len(risk_table) == 0:
        raise ValueError("Risk table contains no valid rows after cleaning.")

    risk_table["n_risk"] = risk_table["n_risk"].round().astype(int)
    km_data = clean_survival_values(km_data)

    km_data = km_data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    risk_table = risk_table.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    if km_data["time"].min() > 0:
        km_data = pd.concat(
            [pd.DataFrame({"time": [0.0], "survival": [1.0]}), km_data],
            ignore_index=True,
        )

    if risk_table["time"].min() > 0:
        raise ValueError("Risk table should start at time 0.")

    if risk_table.iloc[0]["n_risk"] <= 0:
        raise ValueError("Initial number at risk must be positive.")

    return km_data, risk_table


def survival_at_time(km_data, t):
    eligible = km_data[km_data["time"] <= t]
    if len(eligible) == 0:
        return 1.0
    return float(eligible["survival"].iloc[-1])


def safe_model_survival(model, t):
    """Return S(t) from a fitted lifelines model, clipped to [0, 1]."""
    t = max(float(t), 0.0)
    try:
        s = float(model.survival_function_at_times([t]).values[0])
    except Exception:
        s = np.nan

    if pd.isna(s):
        return np.nan

    return float(np.clip(s, 0.0, 1.0))


# -----------------------------
# Guyot-style pseudo-IPD reconstruction
# -----------------------------

def reconstruct_pseudo_ipd_from_km(km_data, risk_table, add_final_zero_risk=True):
    """
    First-pass Guyot-style reconstruction.

    This is approximate and auditable, not a replacement for a fully validated
    Guyot/IPDfromKM implementation.
    """

    km_data, risk_table = validate_km_and_risk(km_data, risk_table)
    diagnostics = []

    max_km_time = float(km_data["time"].max())
    last_risk_time = float(risk_table["time"].max())

    if add_final_zero_risk and last_risk_time < max_km_time:
        diagnostics.append(
            "Risk table ended before final KM time. Added an artificial final risk-table row "
            "with n_risk = 0 at the last KM time. Check whether this is clinically reasonable."
        )
        risk_table = pd.concat(
            [risk_table, pd.DataFrame({"time": [max_km_time], "n_risk": [0]})],
            ignore_index=True,
        ).sort_values("time").reset_index(drop=True)

    ipd_rows = []

    for i in range(len(risk_table) - 1):
        interval_start = float(risk_table.loc[i, "time"])
        interval_end = float(risk_table.loc[i + 1, "time"])
        n_risk_start = int(risk_table.loc[i, "n_risk"])
        n_risk_next = int(risk_table.loc[i + 1, "n_risk"])

        total_allowed_losses = max(0, n_risk_start - n_risk_next)

        interval_km = km_data[
            (km_data["time"] > interval_start) &
            (km_data["time"] <= interval_end)
        ].copy()

        previous_survival = survival_at_time(km_data, interval_start)
        n_current = n_risk_start
        total_events = 0

        for _, row in interval_km.iterrows():
            t = float(row["time"])
            current_survival = float(row["survival"])

            if current_survival < previous_survival and previous_survival > 0:
                raw_events = int(round(n_current * (1.0 - current_survival / previous_survival)))
                remaining_allowed_losses = max(0, total_allowed_losses - total_events)

                estimated_events = min(max(raw_events, 0), n_current, remaining_allowed_losses)

                if raw_events > estimated_events:
                    diagnostics.append(
                        f"At time {t}, raw estimated events ({raw_events}) were capped to "
                        f"{estimated_events} to remain consistent with the risk table."
                    )

                for _ in range(estimated_events):
                    ipd_rows.append({"time": t, "event": 1})

                n_current -= estimated_events
                total_events += estimated_events

            previous_survival = current_survival

        estimated_censors = max(0, total_allowed_losses - total_events)

        if estimated_censors > 0:
            censor_times = np.linspace(interval_start, interval_end, estimated_censors + 2)[1:-1]
            for ct in censor_times:
                ipd_rows.append({"time": float(ct), "event": 0})

    final_time = max(float(risk_table["time"].max()), max_km_time)
    final_n_risk = int(risk_table["n_risk"].iloc[-1])

    if final_n_risk > 0:
        for _ in range(final_n_risk):
            ipd_rows.append({"time": final_time, "event": 0})

    pseudo_ipd = pd.DataFrame(ipd_rows)

    if len(pseudo_ipd) == 0:
        raise ValueError("Pseudo-IPD reconstruction failed. Check KM and risk table inputs.")

    pseudo_ipd = pseudo_ipd.sort_values("time").reset_index(drop=True)

    expected_n = int(risk_table["n_risk"].iloc[0])
    actual_n = len(pseudo_ipd)

    if expected_n != actual_n:
        diagnostics.append(
            f"Expected reconstructed N = {expected_n}, but got {actual_n}. "
            "This can happen because of digitisation error or inconsistent risk-table values."
        )

    return pseudo_ipd, diagnostics, km_data, risk_table


# -----------------------------
# Parametric survival fitting
# -----------------------------

def fit_survival_models(ipd):
    ipd = validate_ipd(ipd)

    T = ipd["time"].astype(float).values
    E = ipd["event"].astype(int).values

    if np.any(T < 0):
        raise ValueError("IPD times must be non-negative.")

    if np.any((E != 0) & (E != 1)):
        raise ValueError("IPD event values must be 0 or 1.")

    T = np.maximum(T, 1e-8)

    candidate_fitters = {
        "Exponential": ExponentialFitter(),
        "Weibull": WeibullFitter(),
        "Log-normal": LogNormalFitter(),
        "Log-logistic": LogLogisticFitter(),
        "Generalized gamma": GeneralizedGammaFitter(),
    }

    fitted_models = {}
    rows = []
    n = len(ipd)

    for name, fitter in candidate_fitters.items():
        try:
            fitter.fit(T, E)
            fitted_models[name] = fitter

            try:
                k = len(fitter.params_)
            except Exception:
                k = np.nan

            loglik = float(fitter.log_likelihood_)
            aic = float(fitter.AIC_)
            bic = float(k * np.log(n) - 2 * loglik) if pd.notna(k) else np.nan

            try:
                median_survival = float(fitter.median_survival_time_)
            except Exception:
                median_survival = np.nan

            rows.append({
                "model": name,
                "log_likelihood": loglik,
                "AIC": aic,
                "BIC": bic,
                "n_parameters": k,
                "median_survival": median_survival,
                "status": "Fitted",
            })

        except Exception as e:
            rows.append({
                "model": name,
                "log_likelihood": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "n_parameters": np.nan,
                "median_survival": np.nan,
                "status": f"Failed: {str(e)}",
            })

    fit_table = pd.DataFrame(rows)
    fitted_table = fit_table[fit_table["status"] == "Fitted"].copy()

    if len(fitted_table) == 0:
        raise ValueError("No parametric models fitted successfully.")

    fit_table = fit_table.sort_values(by=["AIC"], ascending=True, na_position="last").reset_index(drop=True)
    return fitted_models, fit_table


# -----------------------------
# Parameter extraction and formula export
# -----------------------------

def get_param(model, names, default=np.nan):
    """Extract a parameter by trying attributes first and params_ second."""
    for name in names:
        if hasattr(model, name):
            try:
                return float(getattr(model, name))
            except Exception:
                pass

    try:
        params = model.params_
        for name in names:
            if name in params.index:
                return float(params.loc[name])
    except Exception:
        pass

    return default


def model_parameter_dict(model_name, model):
    if model_name == "Exponential":
        return {"lambda_": get_param(model, ["lambda_"])}

    if model_name == "Weibull":
        return {
            "lambda_": get_param(model, ["lambda_"]),
            "rho_": get_param(model, ["rho_"]),
        }

    if model_name == "Log-normal":
        return {
            "mu_": get_param(model, ["mu_"]),
            "sigma_": get_param(model, ["sigma_"]),
        }

    if model_name == "Log-logistic":
        return {
            "alpha_": get_param(model, ["alpha_"]),
            "beta_": get_param(model, ["beta_"]),
        }

    if model_name == "Generalized gamma":
        mu = get_param(model, ["mu_"])
        ln_sigma = get_param(model, ["ln_sigma_"])
        sigma = float(np.exp(ln_sigma)) if pd.notna(ln_sigma) else np.nan
        lam = get_param(model, ["lambda_"])
        return {
            "mu_": mu,
            "ln_sigma_": ln_sigma,
            "sigma_=EXP(ln_sigma_)": sigma,
            "lambda_": lam,
        }

    return {}


def formula_table_for_model(model_name, params):
    """Return formulas using t as time. Excel assumes t is an Excel cell/value."""
    if model_name == "Exponential":
        lam = params.get("lambda_", "lambda_")
        return {
            "mathematical_survival_function": "S(t) = exp(-t / lambda)",
            "excel_survival_formula": f"=EXP(-t/{lam})",
            "treeage_survival_expression": f"Exp(-t/{lam})",
            "r_survival_expression": f"exp(-t/{lam})",
            "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
        }

    if model_name == "Weibull":
        lam = params.get("lambda_", "lambda_")
        rho = params.get("rho_", "rho_")
        return {
            "mathematical_survival_function": "S(t) = exp(-((t / lambda)^rho))",
            "excel_survival_formula": f"=EXP(-POWER(t/{lam},{rho}))",
            "treeage_survival_expression": f"Exp(-Power(t/{lam},{rho}))",
            "r_survival_expression": f"pweibull(t, shape={rho}, scale={lam}, lower.tail=FALSE)",
            "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
        }

    if model_name == "Log-normal":
        mu = params.get("mu_", "mu_")
        sigma = params.get("sigma_", "sigma_")
        return {
            "mathematical_survival_function": "S(t) = 1 - Phi((ln(t) - mu) / sigma)",
            "excel_survival_formula": f"=IF(t<=0,1,1-NORM.S.DIST((LN(t)-{mu})/{sigma},TRUE))",
            "treeage_survival_expression": f"If(t<=0,1,1-NormalCDF((Ln(t)-{mu})/{sigma}))",
            "r_survival_expression": f"plnorm(t, meanlog={mu}, sdlog={sigma}, lower.tail=FALSE)",
            "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
        }

    if model_name == "Log-logistic":
        alpha = params.get("alpha_", "alpha_")
        beta = params.get("beta_", "beta_")
        return {
            "mathematical_survival_function": "S(t) = 1 / (1 + (t / alpha)^beta)",
            "excel_survival_formula": f"=1/(1+POWER(t/{alpha},{beta}))",
            "treeage_survival_expression": f"1/(1+Power(t/{alpha},{beta}))",
            "r_survival_expression": f"1/(1+(t/{alpha})^{beta})",
            "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
        }

    if model_name == "Generalized gamma":
        mu = params.get("mu_", "mu_")
        sigma = params.get("sigma_=EXP(ln_sigma_)", "sigma_")
        lam = params.get("lambda_", "lambda_")
        # Excel GAMMA.DIST(x, a, 1, TRUE) gives the regularized lower incomplete gamma CDF.
        z = f"EXP({lam}*((LN(t)-{mu})/{sigma}))/POWER({lam},2)"
        a = f"1/POWER({lam},2)"
        return {
            "mathematical_survival_function": "If lambda > 0: S(t)=1-Gamma_RL(1/lambda^2, exp(lambda*((ln(t)-mu)/sigma))/lambda^2); if lambda <= 0: S(t)=Gamma_RL(...). sigma=exp(ln_sigma).",
            "excel_survival_formula": f"=IF(t<=0,1,IF({lam}>0,1-GAMMA.DIST({z},{a},1,TRUE),GAMMA.DIST({z},{a},1,TRUE)))",
            "treeage_survival_expression": "Use exported survival_predictions table or implement the regularized lower incomplete gamma function; TreeAge installations differ in gamma-CDF syntax.",
            "r_survival_expression": f"z <- exp({lam}*((log(t)-{mu})/{sigma}))/{lam}^2; a <- 1/{lam}^2; ifelse({lam}>0, 1-pgamma(z, shape=a, scale=1), pgamma(z, shape=a, scale=1))",
            "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
        }

    return {
        "mathematical_survival_function": "Not available",
        "excel_survival_formula": "Not available",
        "treeage_survival_expression": "Not available",
        "r_survival_expression": "Not available",
        "cycle_probability_formula": "p_cycle = 1 - S(t_end) / S(t_start)",
    }


def make_parameter_export(fitted_models):
    rows = []
    formula_rows = []

    for model_name, model in fitted_models.items():
        params = model_parameter_dict(model_name, model)
        formulas = formula_table_for_model(model_name, params)

        for p_name, p_value in params.items():
            rows.append({
                "model": model_name,
                "parameter": p_name,
                "value": p_value,
            })

        formula_rows.append({
            "model": model_name,
            **formulas,
        })

    return pd.DataFrame(rows), pd.DataFrame(formula_rows)


# -----------------------------
# Predictions and cycle probabilities
# -----------------------------

def make_survival_predictions(fitted_models, horizon, n_points=301):
    times = np.linspace(0, horizon, n_points)
    rows = []

    for model_name, model in fitted_models.items():
        for t in times:
            s = safe_model_survival(model, t)
            rows.append({
                "model": model_name,
                "time": float(t),
                "survival": s,
            })

    return pd.DataFrame(rows)


def make_cycle_probabilities(fitted_models, n_cycles, cycle_length):
    rows = []

    for model_name, model in fitted_models.items():
        for cycle in range(1, int(n_cycles) + 1):
            start_t = (cycle - 1) * cycle_length
            end_t = cycle * cycle_length

            s_start = safe_model_survival(model, start_t)
            s_end = safe_model_survival(model, end_t)

            if pd.isna(s_start) or pd.isna(s_end) or s_start <= 0.0:
                p_event = np.nan
            else:
                p_event = 1.0 - min(max(s_end / s_start, 0.0), 1.0)

            rows.append({
                "model": model_name,
                "cycle": cycle,
                "start_time": start_t,
                "end_time": end_t,
                "S_start": s_start,
                "S_end": s_end,
                "cycle_event_probability": float(p_event) if pd.notna(p_event) else np.nan,
            })

    return pd.DataFrame(rows)


def weighted_survival_at_time(fitted_models, model_a, model_b, weight_a, t):
    weight_a = float(weight_a)
    weight_b = 1.0 - weight_a
    s_a = safe_model_survival(fitted_models[model_a], t)
    s_b = safe_model_survival(fitted_models[model_b], t)

    if pd.isna(s_a) or pd.isna(s_b):
        return np.nan

    return float(np.clip(weight_a * s_a + weight_b * s_b, 0.0, 1.0))


def make_weighted_survival_predictions(fitted_models, model_a, model_b, weight_a, horizon, n_points=301):
    times = np.linspace(0, horizon, n_points)
    label = f"Weighted: {weight_a:.2f}*{model_a} + {1-weight_a:.2f}*{model_b}"
    rows = []

    for t in times:
        s_a = safe_model_survival(fitted_models[model_a], t)
        s_b = safe_model_survival(fitted_models[model_b], t)
        s_w = weighted_survival_at_time(fitted_models, model_a, model_b, weight_a, t)
        rows.append({
            "model": label,
            "model_a": model_a,
            "model_b": model_b,
            "weight_a": float(weight_a),
            "weight_b": float(1.0 - weight_a),
            "time": float(t),
            "S_model_a": s_a,
            "S_model_b": s_b,
            "survival": s_w,
        })

    return pd.DataFrame(rows)


def make_weighted_cycle_probabilities(fitted_models, model_a, model_b, weight_a, n_cycles, cycle_length):
    label = f"Weighted: {weight_a:.2f}*{model_a} + {1-weight_a:.2f}*{model_b}"
    rows = []

    for cycle in range(1, int(n_cycles) + 1):
        start_t = (cycle - 1) * cycle_length
        end_t = cycle * cycle_length
        s_start = weighted_survival_at_time(fitted_models, model_a, model_b, weight_a, start_t)
        s_end = weighted_survival_at_time(fitted_models, model_a, model_b, weight_a, end_t)

        if pd.isna(s_start) or pd.isna(s_end) or s_start <= 0:
            p_event = np.nan
        else:
            p_event = 1.0 - min(max(s_end / s_start, 0.0), 1.0)

        rows.append({
            "model": label,
            "cycle": cycle,
            "start_time": start_t,
            "end_time": end_t,
            "S_start": s_start,
            "S_end": s_end,
            "cycle_event_probability": float(p_event) if pd.notna(p_event) else np.nan,
            "model_a": model_a,
            "model_b": model_b,
            "weight_a": float(weight_a),
            "weight_b": float(1.0 - weight_a),
        })

    return pd.DataFrame(rows)


def make_annual_probabilities_from_monthly_cycles(cycle_probs, months_per_year=12):
    """
    Convert monthly cycle probabilities to annual probabilities.

    This uses the product form:
      p_annual = 1 - product(1 - p_month_i)

    For probabilities derived from the same survival curve, this is equivalent to:
      p_annual = 1 - S(month_end) / S(month_start)
    """
    rows = []
    months_per_year = int(months_per_year)

    for model_name, tmp in cycle_probs.groupby("model"):
        tmp = tmp.sort_values("cycle").copy()
        max_year = int(np.floor(len(tmp) / months_per_year))

        for year in range(1, max_year + 1):
            block = tmp.iloc[(year - 1) * months_per_year: year * months_per_year]

            p_months = block["cycle_event_probability"].astype(float).clip(lower=0, upper=1).values
            p_annual_product = 1.0 - float(np.prod(1.0 - p_months))

            s_start = float(block["S_start"].iloc[0])
            s_end = float(block["S_end"].iloc[-1])
            p_annual_survival = np.nan if s_start <= 0 else 1.0 - min(max(s_end / s_start, 0.0), 1.0)

            rows.append({
                "model": model_name,
                "year": year,
                "start_month": int(block["start_time"].iloc[0]),
                "end_month": int(block["end_time"].iloc[-1]),
                "S_start": s_start,
                "S_end": s_end,
                "annual_event_probability_product_of_monthlies": p_annual_product,
                "annual_event_probability_from_survival_ratio": p_annual_survival,
                "formula": "1 - PRODUCT(1 - monthly probabilities) = 1 - S(end month) / S(start month)",
            })

    return pd.DataFrame(rows)


def make_conversion_examples():
    return pd.DataFrame([
        {
            "conversion": "Rate to probability",
            "formula": "p = 1 - exp(-r * cycle_length)",
            "excel_formula": "=1-EXP(-rate*cycle_length)",
            "use_case": "Convert a constant hazard/rate to a cycle probability.",
        },
        {
            "conversion": "Probability to rate",
            "formula": "r = -ln(1-p) / cycle_length",
            "excel_formula": "=-LN(1-p)/cycle_length",
            "use_case": "Convert a cycle probability to a constant rate.",
        },
        {
            "conversion": "Monthly probability to annual probability, constant monthly risk",
            "formula": "p_annual = 1 - (1-p_month)^12",
            "excel_formula": "=1-POWER(1-p_month,12)",
            "use_case": "Use only if monthly probability is assumed constant across the year.",
        },
        {
            "conversion": "Monthly probabilities to annual probability, varying monthly risk",
            "formula": "p_annual = 1 - product_i(1-p_month_i)",
            "excel_formula": "=1-PRODUCT(1-range_of_monthly_probabilities)",
            "use_case": "Preferred when probabilities vary by month/cycle from a survival distribution.",
        },
        {
            "conversion": "Annual probability directly from survival curve",
            "formula": "p_year = 1 - S(t+12) / S(t), if time is measured in months",
            "excel_formula": "=1-S_end/S_start",
            "use_case": "Preferred for parametric survival curves when input time is in months.",
        },
    ])


# -----------------------------
# Plotting
# -----------------------------

def plot_reconstructed_vs_digitised(ipd, km_data):
    kmf = KaplanMeierFitter()
    kmf.fit(ipd["time"], event_observed=ipd["event"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], where="post", label="Reconstructed KM")
    ax.scatter(km_data["time"], km_data["survival"], label="Digitised published KM")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.set_title("Digitised KM vs reconstructed pseudo-IPD KM")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_parametric_fits(ipd, survival_predictions, selected_models=None):
    kmf = KaplanMeierFitter()
    kmf.fit(ipd["time"], event_observed=ipd["event"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(kmf.survival_function_.index, kmf.survival_function_["KM_estimate"], where="post", label="Observed/reconstructed KM")

    plot_data = survival_predictions.copy()
    if selected_models is not None and len(selected_models) > 0:
        plot_data = plot_data[plot_data["model"].isin(selected_models)]

    for model_name in plot_data["model"].unique():
        tmp = plot_data[plot_data["model"] == model_name]
        ax.plot(tmp["time"], tmp["survival"], label=model_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.set_title("Parametric survival fits")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_hazard_proxy(cycle_probs, selected_models=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = cycle_probs.copy()
    if selected_models is not None and len(selected_models) > 0:
        plot_data = plot_data[plot_data["model"].isin(selected_models)]

    for model_name in plot_data["model"].unique():
        tmp = plot_data[plot_data["model"] == model_name]
        ax.plot(tmp["cycle"], tmp["cycle_event_probability"], marker="o", label=model_name)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cycle event probability")
    ax.set_title("Cycle event probabilities by fitted model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------
# Excel export
# -----------------------------

def make_excel_export(
    ipd,
    fit_table,
    parameter_table,
    formula_table,
    survival_predictions,
    cycle_probs,
    annual_probs=None,
    weighted_survival_predictions=None,
    weighted_cycle_probs=None,
    km_data=None,
    risk_table=None,
    diagnostics=None,
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ipd.to_excel(writer, index=False, sheet_name="pseudo_ipd_or_ipd")
        fit_table.to_excel(writer, index=False, sheet_name="fit_ranking")
        parameter_table.to_excel(writer, index=False, sheet_name="estimated_parameters")
        formula_table.to_excel(writer, index=False, sheet_name="formula_export")
        survival_predictions.to_excel(writer, index=False, sheet_name="survival_predictions")
        cycle_probs.to_excel(writer, index=False, sheet_name="cycle_probabilities")
        make_conversion_examples().to_excel(writer, index=False, sheet_name="conversion_formulas")

        if annual_probs is not None and len(annual_probs) > 0:
            annual_probs.to_excel(writer, index=False, sheet_name="annual_from_monthly")

        if weighted_survival_predictions is not None and len(weighted_survival_predictions) > 0:
            weighted_survival_predictions.to_excel(writer, index=False, sheet_name="weighted_survival")

        if weighted_cycle_probs is not None and len(weighted_cycle_probs) > 0:
            weighted_cycle_probs.to_excel(writer, index=False, sheet_name="weighted_cycle_probs")

        if km_data is not None:
            km_data.to_excel(writer, index=False, sheet_name="digitised_km")

        if risk_table is not None:
            risk_table.to_excel(writer, index=False, sheet_name="risk_table")

        if diagnostics:
            pd.DataFrame({"diagnostic_note": diagnostics}).to_excel(writer, index=False, sheet_name="diagnostics")

    return output.getvalue()


# -----------------------------
# Sample data
# -----------------------------

def sample_ipd():
    return pd.DataFrame({
        "time": [2, 4, 5, 7, 9, 11, 13, 15, 18, 20, 22, 25, 27, 30],
        "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    })


def sample_km():
    return pd.DataFrame({
        "time": [0, 6, 12, 18, 24, 30, 36],
        "survival": [1.00, 0.92, 0.84, 0.76, 0.69, 0.62, 0.56],
    })


def sample_risk_table():
    return pd.DataFrame({
        "time": [0, 12, 24, 36],
        "n_risk": [100, 82, 61, 43],
    })


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(page_title="Enhanced Survival Distribution Fitting Agent", layout="wide")

st.title("Enhanced Survival Distribution Fitting Agent")
st.caption(
    "Fit parametric survival curves from IPD or digitised Kaplan-Meier data, "
    "export parameters/formulas, choose displayed distributions, combine two distributions by weight, "
    "and convert monthly cycle probabilities to annual transition probabilities."
)

st.warning(
    "Important: the Guyot-style reconstruction here is approximate. "
    "For regulatory or HTA use, validate the reconstructed KM curve against the published KM, "
    "number-at-risk table, median survival, and reported event counts."
)

with st.sidebar:
    st.header("Model settings")

    input_type = st.radio("Choose input type", ["IPD-style data", "Digitised KM + risk table"])

    time_unit = st.selectbox(
        "Input time unit",
        ["Months", "Years"],
        index=0,
        help="This does not rescale your data automatically. It labels the interpretation and controls annualisation guidance.",
    )

    n_cycles = st.number_input("Number of cycles to export", min_value=1, max_value=600, value=60, step=1)

    cycle_length = st.number_input(
        "Cycle length in same time unit as input",
        min_value=0.01,
        value=1.0,
        step=1.0,
        help="Use 1 for monthly cycles if input time is months. Use 12 for annual cycles if input time is months. Use 1 for annual cycles if input time is years.",
    )

    plot_horizon = st.number_input(
        "Plot horizon in same time unit as input",
        min_value=1.0,
        value=float(n_cycles) * float(cycle_length),
        step=1.0,
    )

    use_sample = st.checkbox("Use sample data", value=False)

st.subheader("Required input format")

with st.expander("See required columns"):
    st.markdown(
        """
        **For IPD-style data**, upload a CSV or Excel file with:

        | time | event |
        |---:|---:|
        | 4.2 | 1 |
        | 7.0 | 0 |

        `event = 1` means event occurred.  
        `event = 0` means censored.

        **For digitised KM data**, upload:

        KM file:

        | time | survival |
        |---:|---:|
        | 0 | 1.00 |
        | 12 | 0.84 |

        Risk table file:

        | time | n_risk |
        |---:|---:|
        | 0 | 100 |
        | 12 | 82 |
        """
    )

ipd = None
km_data = None
risk_table = None
diagnostics = []

try:
    if input_type == "IPD-style data":
        if use_sample:
            ipd = sample_ipd()
            st.info("Using sample IPD-style data.")
        else:
            uploaded_ipd = st.file_uploader("Upload IPD file", type=["csv", "xlsx", "xls"])
            if uploaded_ipd is not None:
                ipd = read_uploaded_table(uploaded_ipd)

        if ipd is None:
            st.stop()

        ipd = validate_ipd(ipd)

    else:
        if use_sample:
            km_data = sample_km()
            risk_table = sample_risk_table()
            st.info("Using sample digitised KM and risk-table data.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                uploaded_km = st.file_uploader("Upload digitised KM file", type=["csv", "xlsx", "xls"])
            with col2:
                uploaded_risk = st.file_uploader("Upload risk table file", type=["csv", "xlsx", "xls"])

            if uploaded_km is not None:
                km_data = read_uploaded_table(uploaded_km)
            if uploaded_risk is not None:
                risk_table = read_uploaded_table(uploaded_risk)

        if km_data is None or risk_table is None:
            st.stop()

        ipd, diagnostics, km_data, risk_table = reconstruct_pseudo_ipd_from_km(
            km_data,
            risk_table,
            add_final_zero_risk=True,
        )

except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()


# -----------------------------
# Display input/pseudo-IPD
# -----------------------------

st.subheader("Input / reconstructed data")
col1, col2 = st.columns(2)

with col1:
    st.write("IPD or reconstructed pseudo-IPD")
    st.dataframe(ipd.head(100), use_container_width=True)

with col2:
    st.metric("Number of individuals", len(ipd))
    st.metric("Number of events", int(ipd["event"].sum()))
    st.metric("Number censored", int((ipd["event"] == 0).sum()))

    if diagnostics:
        st.write("Diagnostics")
        for note in diagnostics:
            st.warning(note)

if input_type == "Digitised KM + risk table":
    st.subheader("Reconstruction check")
    fig_recon = plot_reconstructed_vs_digitised(ipd, km_data)
    st.pyplot(fig_recon)


# -----------------------------
# Fit models
# -----------------------------

try:
    fitted_models, fit_table = fit_survival_models(ipd)
    parameter_table, formula_table = make_parameter_export(fitted_models)
    survival_predictions = make_survival_predictions(fitted_models, horizon=float(plot_horizon), n_points=301)
    cycle_probs = make_cycle_probabilities(fitted_models, n_cycles=int(n_cycles), cycle_length=float(cycle_length))

except Exception as e:
    st.error(f"Model-fitting error: {e}")
    st.stop()


# -----------------------------
# Results
# -----------------------------

st.subheader("Model ranking")
st.dataframe(fit_table, use_container_width=True)

best_model = fit_table.loc[fit_table["status"] == "Fitted", "model"].iloc[0]
st.success(
    f"Best statistical fit by AIC: {best_model}. "
    "Do not select the base case by AIC alone; also check visual fit, hazard shape, "
    "clinical plausibility, and external evidence."
)

st.subheader("Estimated parameters and model formulas")
param_col, formula_col = st.columns(2)
with param_col:
    st.write("Estimated parameters")
    st.dataframe(parameter_table, use_container_width=True)
with formula_col:
    st.write("Survival-function formulas for Excel, TreeAge and R")
    st.dataframe(formula_table, use_container_width=True)

st.subheader("Parametric survival curves")
model_options = list(fitted_models.keys())
default_models = model_options
selected_curve_models = st.multiselect(
    "Choose which fitted distributions to show on the curve",
    options=model_options,
    default=default_models,
)

fig_fits = plot_parametric_fits(ipd, survival_predictions, selected_models=selected_curve_models)
st.pyplot(fig_fits)

st.subheader("Weighted distribution between two fitted curves")
weighted_col1, weighted_col2, weighted_col3 = st.columns(3)

with weighted_col1:
    model_a = st.selectbox("Distribution A", options=model_options, index=0)
with weighted_col2:
    model_b_index = 1 if len(model_options) > 1 else 0
    model_b = st.selectbox("Distribution B", options=model_options, index=model_b_index)
with weighted_col3:
    weight_a = st.slider(
        "Weight on distribution A",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Weighted survival is calculated as w*S_A(t) + (1-w)*S_B(t). Cycle probabilities are then derived from the weighted survival curve.",
    )

weighted_survival_predictions = make_weighted_survival_predictions(
    fitted_models,
    model_a=model_a,
    model_b=model_b,
    weight_a=float(weight_a),
    horizon=float(plot_horizon),
    n_points=301,
)
weighted_cycle_probs = make_weighted_cycle_probabilities(
    fitted_models,
    model_a=model_a,
    model_b=model_b,
    weight_a=float(weight_a),
    n_cycles=int(n_cycles),
    cycle_length=float(cycle_length),
)

weighted_label = weighted_survival_predictions["model"].iloc[0]
combined_predictions_for_plot = pd.concat(
    [survival_predictions, weighted_survival_predictions[["model", "time", "survival"]]],
    ignore_index=True,
)
combined_selected_models = list(selected_curve_models) + [weighted_label]
fig_weighted = plot_parametric_fits(ipd, combined_predictions_for_plot, selected_models=combined_selected_models)
st.pyplot(fig_weighted)

st.markdown(
    "Weighted curve formula: `S_weighted(t) = w*S_A(t) + (1-w)*S_B(t)`. "
    "The transition probability is then calculated from the weighted survival curve: "
    "`p = 1 - S_weighted(t_end) / S_weighted(t_start)`."
)

st.subheader("Cycle transition probabilities")
selected_prob_model = st.selectbox(
    "Select fitted distribution to view transition probabilities",
    options=model_options,
    index=model_options.index(best_model),
)
selected_probs = cycle_probs[cycle_probs["model"] == selected_prob_model].copy()
st.dataframe(selected_probs, use_container_width=True)

with st.expander("View weighted-cycle probabilities"):
    st.dataframe(weighted_cycle_probs, use_container_width=True)

st.markdown(
    "Cycle event probability is calculated as `p = 1 - S(t + cycle length) / S(t)`. "
    "If input time is in months and cycle length is `12`, this directly gives annual event probabilities. "
    "If cycle length is `1`, monthly probabilities can be annualised below."
)

fig_probs = plot_hazard_proxy(
    pd.concat([cycle_probs, weighted_cycle_probs[cycle_probs.columns]], ignore_index=True),
    selected_models=combined_selected_models,
)
st.pyplot(fig_probs)

st.subheader("Monthly to annual probability conversion")
annual_probs = pd.DataFrame()

if time_unit == "Months" and abs(float(cycle_length) - 1.0) < 1e-9:
    combined_cycle_probs = pd.concat([cycle_probs, weighted_cycle_probs[cycle_probs.columns]], ignore_index=True)
    annual_probs = make_annual_probabilities_from_monthly_cycles(combined_cycle_probs, months_per_year=12)
    st.write("Annual probabilities generated from monthly cycle probabilities.")
    st.dataframe(annual_probs, use_container_width=True)
else:
    st.info(
        "To generate annual probabilities from monthly cycles, set input time unit to Months and cycle length to 1. "
        "If you set cycle length to 12 with month-based data, your cycle probabilities are already annual probabilities."
    )

with st.expander("General conversion formulas"):
    st.dataframe(make_conversion_examples(), use_container_width=True)


# -----------------------------
# Downloads
# -----------------------------

st.subheader("Downloads")
excel_bytes = make_excel_export(
    ipd=ipd,
    fit_table=fit_table,
    parameter_table=parameter_table,
    formula_table=formula_table,
    survival_predictions=survival_predictions,
    cycle_probs=cycle_probs,
    annual_probs=annual_probs,
    weighted_survival_predictions=weighted_survival_predictions,
    weighted_cycle_probs=weighted_cycle_probs,
    km_data=km_data,
    risk_table=risk_table,
    diagnostics=diagnostics,
)

st.download_button(
    label="Download Excel output",
    data=excel_bytes,
    file_name="enhanced_survival_fitting_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.download_button(
    label="Download IPD / pseudo-IPD as CSV",
    data=ipd.to_csv(index=False).encode("utf-8"),
    file_name="pseudo_ipd.csv",
    mime="text/csv",
)

st.download_button(
    label="Download cycle probabilities as CSV",
    data=cycle_probs.to_csv(index=False).encode("utf-8"),
    file_name="cycle_transition_probabilities.csv",
    mime="text/csv",
)

st.download_button(
    label="Download estimated parameters as CSV",
    data=parameter_table.to_csv(index=False).encode("utf-8"),
    file_name="estimated_survival_parameters.csv",
    mime="text/csv",
)

st.download_button(
    label="Download formula export as CSV",
    data=formula_table.to_csv(index=False).encode("utf-8"),
    file_name="survival_formula_export.csv",
    mime="text/csv",
)

if len(annual_probs) > 0:
    st.download_button(
        label="Download annual probabilities from monthly cycles as CSV",
        data=annual_probs.to_csv(index=False).encode("utf-8"),
        file_name="annual_probabilities_from_monthly_cycles.csv",
        mime="text/csv",
    )

st.download_button(
    label="Download weighted cycle probabilities as CSV",
    data=weighted_cycle_probs.to_csv(index=False).encode("utf-8"),
    file_name="weighted_cycle_probabilities.csv",
    mime="text/csv",
)


# -----------------------------
# Report text
# -----------------------------

st.subheader("Draft report wording")
report_text = f"""
Parametric survival models were fitted to the available time-to-event data. The candidate distributions included exponential, Weibull, log-normal, log-logistic and generalized gamma models. The best statistical fit according to AIC was {best_model}. However, the base-case extrapolation should not be selected on statistical fit alone. Visual fit to the observed Kaplan-Meier curve, the implied hazard over time, clinical plausibility, and consistency with external long-term evidence should also be considered.

The fitted model parameters were exported together with model-specific survival functions that can be implemented in Excel, TreeAge or R. Cycle-specific event probabilities were derived from the fitted survival function using p = 1 - S(t + cycle length) / S(t). Where input time was measured in months, annual probabilities can either be generated directly by setting the cycle length to 12 months or by combining twelve monthly cycle probabilities using p_annual = 1 - product(1 - p_month_i).

A weighted survival curve can be generated between two selected distributions using S_weighted(t) = w*S_A(t) + (1-w)*S_B(t). The corresponding transition probabilities are derived from the weighted survival curve rather than by directly averaging the monthly cycle probabilities.
"""

st.text_area("You can copy this into a report", value=report_text, height=280)
