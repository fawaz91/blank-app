# app.py
# Survival Distribution Fitting Agent
# Inputs:
#   1) IPD-style data: time, event
#   2) Digitised KM data + number-at-risk table
#
# Output:
#   - Reconstructed pseudo-IPD
#   - Parametric survival fits
#   - AIC/BIC ranking
#   - Survival curves
#   - Cycle transition probabilities
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
            ignore_index=True
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


# -----------------------------
# Guyot-style pseudo-IPD reconstruction
# -----------------------------

def reconstruct_pseudo_ipd_from_km(km_data, risk_table, add_final_zero_risk=True):
    """
    First-pass Guyot-style reconstruction.

    This is approximate and auditable, not a replacement for a fully validated
    Guyot/IPDfromKM implementation.

    Input:
        km_data: columns time, survival
        risk_table: columns time, n_risk

    Output:
        pseudo-IPD: columns time, event
        diagnostics: reconstruction notes
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
            [
                risk_table,
                pd.DataFrame({"time": [max_km_time], "n_risk": [0]})
            ],
            ignore_index=True
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

                estimated_events = min(
                    max(raw_events, 0),
                    n_current,
                    remaining_allowed_losses
                )

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
            censor_times = np.linspace(
                interval_start,
                interval_end,
                estimated_censors + 2
            )[1:-1]

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

            if pd.notna(k):
                bic = float(k * np.log(n) - 2 * loglik)
            else:
                bic = np.nan

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
                "status": "Fitted"
            })

        except Exception as e:
            rows.append({
                "model": name,
                "log_likelihood": np.nan,
                "AIC": np.nan,
                "BIC": np.nan,
                "n_parameters": np.nan,
                "median_survival": np.nan,
                "status": f"Failed: {str(e)}"
            })

    fit_table = pd.DataFrame(rows)

    fitted_table = fit_table[fit_table["status"] == "Fitted"].copy()

    if len(fitted_table) == 0:
        raise ValueError("No parametric models fitted successfully.")

    fit_table = fit_table.sort_values(
        by=["AIC"],
        ascending=True,
        na_position="last"
    ).reset_index(drop=True)

    return fitted_models, fit_table


def make_survival_predictions(fitted_models, horizon, n_points=301):
    times = np.linspace(0, horizon, n_points)
    rows = []

    for model_name, model in fitted_models.items():
        try:
            surv = model.survival_function_at_times(times).values.astype(float)
            surv = np.clip(surv, 0, 1)

            for t, s in zip(times, surv):
                rows.append({
                    "model": model_name,
                    "time": float(t),
                    "survival": float(s)
                })

        except Exception:
            pass

    return pd.DataFrame(rows)


def make_cycle_probabilities(fitted_models, n_cycles, cycle_length):
    rows = []

    for model_name, model in fitted_models.items():
        for cycle in range(1, int(n_cycles) + 1):
            start_t = (cycle - 1) * cycle_length
            end_t = cycle * cycle_length

            s_start = float(model.survival_function_at_times([start_t]).values[0])
            s_end = float(model.survival_function_at_times([end_t]).values[0])

            s_start = max(s_start, 0.0)
            s_end = max(s_end, 0.0)

            if s_start <= 0.0:
                p_event = 0.0
            else:
                p_event = 1.0 - min(max(s_end / s_start, 0.0), 1.0)

            rows.append({
                "model": model_name,
                "cycle": cycle,
                "start_time": start_t,
                "end_time": end_t,
                "S_start": s_start,
                "S_end": s_end,
                "cycle_event_probability": float(p_event)
            })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def plot_reconstructed_vs_digitised(ipd, km_data):
    kmf = KaplanMeierFitter()
    kmf.fit(ipd["time"], event_observed=ipd["event"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.step(
        kmf.survival_function_.index,
        kmf.survival_function_["KM_estimate"],
        where="post",
        label="Reconstructed KM"
    )

    ax.scatter(
        km_data["time"],
        km_data["survival"],
        label="Digitised published KM"
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.set_title("Digitised KM vs reconstructed pseudo-IPD KM")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_parametric_fits(ipd, survival_predictions):
    kmf = KaplanMeierFitter()
    kmf.fit(ipd["time"], event_observed=ipd["event"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.step(
        kmf.survival_function_.index,
        kmf.survival_function_["KM_estimate"],
        where="post",
        label="Observed/reconstructed KM"
    )

    for model_name in survival_predictions["model"].unique():
        tmp = survival_predictions[survival_predictions["model"] == model_name]
        ax.plot(tmp["time"], tmp["survival"], label=model_name)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.set_title("Parametric survival fits")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_hazard_proxy(cycle_probs):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in cycle_probs["model"].unique():
        tmp = cycle_probs[cycle_probs["model"] == model_name]
        ax.plot(
            tmp["cycle"],
            tmp["cycle_event_probability"],
            marker="o",
            label=model_name
        )

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
    survival_predictions,
    cycle_probs,
    km_data=None,
    risk_table=None,
    diagnostics=None
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ipd.to_excel(writer, index=False, sheet_name="pseudo_ipd_or_ipd")
        fit_table.to_excel(writer, index=False, sheet_name="fit_ranking")
        survival_predictions.to_excel(writer, index=False, sheet_name="survival_predictions")
        cycle_probs.to_excel(writer, index=False, sheet_name="cycle_probabilities")

        if km_data is not None:
            km_data.to_excel(writer, index=False, sheet_name="digitised_km")

        if risk_table is not None:
            risk_table.to_excel(writer, index=False, sheet_name="risk_table")

        if diagnostics:
            pd.DataFrame({"diagnostic_note": diagnostics}).to_excel(
                writer,
                index=False,
                sheet_name="diagnostics"
            )

    return output.getvalue()


# -----------------------------
# Sample data
# -----------------------------

def sample_ipd():
    return pd.DataFrame({
        "time": [2, 4, 5, 7, 9, 11, 13, 15, 18, 20, 22, 25, 27, 30],
        "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    })


def sample_km():
    return pd.DataFrame({
        "time": [0, 6, 12, 18, 24, 30, 36],
        "survival": [1.00, 0.92, 0.84, 0.76, 0.69, 0.62, 0.56]
    })


def sample_risk_table():
    return pd.DataFrame({
        "time": [0, 12, 24, 36],
        "n_risk": [100, 82, 61, 43]
    })


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(
    page_title="Survival Distribution Fitting Agent",
    layout="wide"
)

st.title("Survival Distribution Fitting Agent")
st.caption(
    "Fit parametric survival curves from IPD or digitised Kaplan–Meier data. "
    "Includes a first-pass Guyot-style pseudo-IPD reconstruction."
)

st.warning(
    "Important: the Guyot-style reconstruction here is approximate. "
    "For regulatory or HTA use, validate the reconstructed KM curve against the published KM, "
    "number-at-risk table, median survival, and reported event counts."
)

with st.sidebar:
    st.header("Model settings")

    input_type = st.radio(
        "Choose input type",
        [
            "IPD-style data",
            "Digitised KM + risk table"
        ]
    )

    n_cycles = st.number_input(
        "Number of cycles to export",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    cycle_length = st.number_input(
        "Cycle length in same time unit as input",
        min_value=0.01,
        value=1.0,
        step=1.0,
        help="Use 1 if your data are in years. Use 12 if your data are in months and you want annual cycles."
    )

    plot_horizon = st.number_input(
        "Plot horizon in same time unit as input",
        min_value=1.0,
        value=float(n_cycles) * float(cycle_length),
        step=1.0
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
            uploaded_ipd = st.file_uploader(
                "Upload IPD file",
                type=["csv", "xlsx", "xls"]
            )
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
                uploaded_km = st.file_uploader(
                    "Upload digitised KM file",
                    type=["csv", "xlsx", "xls"]
                )

            with col2:
                uploaded_risk = st.file_uploader(
                    "Upload risk table file",
                    type=["csv", "xlsx", "xls"]
                )

            if uploaded_km is not None:
                km_data = read_uploaded_table(uploaded_km)

            if uploaded_risk is not None:
                risk_table = read_uploaded_table(uploaded_risk)

        if km_data is None or risk_table is None:
            st.stop()

        ipd, diagnostics, km_data, risk_table = reconstruct_pseudo_ipd_from_km(
            km_data,
            risk_table,
            add_final_zero_risk=True
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
    survival_predictions = make_survival_predictions(
        fitted_models,
        horizon=float(plot_horizon),
        n_points=301
    )
    cycle_probs = make_cycle_probabilities(
        fitted_models,
        n_cycles=int(n_cycles),
        cycle_length=float(cycle_length)
    )

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

st.subheader("Parametric survival curves")
fig_fits = plot_parametric_fits(ipd, survival_predictions)
st.pyplot(fig_fits)

st.subheader("Cycle transition probabilities")

selected_model = st.selectbox(
    "Select model to view transition probabilities",
    options=list(fitted_models.keys()),
    index=list(fitted_models.keys()).index(best_model)
)

selected_probs = cycle_probs[cycle_probs["model"] == selected_model].copy()

st.dataframe(selected_probs, use_container_width=True)

st.markdown(
    """
    The cycle event probability is calculated as:

    `p = 1 - S(t + cycle length) / S(t)`

    For example, if your input data are in months and you set cycle length to `12`,
    this gives annual event probabilities.
    """
)

fig_probs = plot_hazard_proxy(cycle_probs)
st.pyplot(fig_probs)


# -----------------------------
# Downloads
# -----------------------------

st.subheader("Downloads")

excel_bytes = make_excel_export(
    ipd=ipd,
    fit_table=fit_table,
    survival_predictions=survival_predictions,
    cycle_probs=cycle_probs,
    km_data=km_data,
    risk_table=risk_table,
    diagnostics=diagnostics
)

st.download_button(
    label="Download Excel output",
    data=excel_bytes,
    file_name="survival_fitting_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.download_button(
    label="Download IPD / pseudo-IPD as CSV",
    data=ipd.to_csv(index=False).encode("utf-8"),
    file_name="pseudo_ipd.csv",
    mime="text/csv"
)

st.download_button(
    label="Download cycle probabilities as CSV",
    data=cycle_probs.to_csv(index=False).encode("utf-8"),
    file_name="cycle_transition_probabilities.csv",
    mime="text/csv"
)


# -----------------------------
# Report text
# -----------------------------

st.subheader("Draft report wording")

report_text = f"""
Parametric survival models were fitted to the available time-to-event data.
The candidate distributions included exponential, Weibull, log-normal,
log-logistic and generalized gamma models. The best statistical fit according
to AIC was {best_model}. However, the base-case extrapolation should not be
selected on statistical fit alone. Visual fit to the observed Kaplan–Meier
curve, the implied hazard over time, clinical plausibility, and consistency
with external long-term evidence should also be considered.

Cycle-specific event probabilities were derived from the fitted survival
function using the formula:

p = 1 - S(t + cycle length) / S(t)

These probabilities can be used as transition probabilities in a Markov model,
provided that the survival endpoint corresponds to the event being modelled.
"""

st.text_area(
    "You can copy this into a report",
    value=report_text,
    height=240
)
