"""
KM digitization, manual editing, and registry mortality adjustment module.
Provides functions for KM curve cleaning, editing, interval probability derivation,
Weibull fitting, and background mortality adjustment.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

# Optional imports for image processing (digitization)
try:
    import cv2
    from PIL import Image
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

import io

EPS = 1e-12


# ==============================
# AUTOMATIC KM CURVE DIGITIZATION
# ==============================

def preprocess_km_image(image_array):
    """
    Preprocesses KM plot image for curve detection.
    Converts to grayscale and applies contrast enhancement.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def detect_curve_pixels(image_array, curve_color_range="dark"):
    """
    Detects curve pixels using edge detection and filtering.
    curve_color_range: "dark" or "light"
    """
    try:
        preprocessed = preprocess_km_image(image_array)
        
        # Edge detection using Canny
        edges = cv2.Canny(preprocessed, 50, 150)
        
        # Morphological operations to connect curve pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the longest contour (main curve)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Extract points from contour
        points = largest_contour.squeeze()
        if len(points.shape) == 1:
            # Single point, reshape
            return None
        
        # Ensure points is 2D
        if len(points.shape) != 2 or points.shape[1] != 2:
            return None
            
        return points.astype(np.float32)
    except Exception as e:
        return None


def detect_axes(image_array, curve_points=None):
    """
    Detects axis ranges from the image.
    Returns dict with x_min, x_max, y_min, y_max in pixel coordinates.
    """
    height, width = image_array.shape[:2]
    
    # Simple heuristic: assume plot area is 80% of image
    # and axes are at edges
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    
    plot_left = margin_x
    plot_right = width - margin_x
    plot_top = margin_y
    plot_bottom = height - margin_y
    
    return {
        "pixel_x_min": plot_left,
        "pixel_x_max": plot_right,
        "pixel_y_min": plot_top,
        "pixel_y_max": plot_bottom,
        "image_width": width,
        "image_height": height
    }


def extract_km_points_from_image(image_array, time_min=0, time_max=None, 
                                  survival_min=0, survival_max=1):
    """
    Extracts KM points from image automatically.
    
    Args:
        image_array: numpy array of image
        time_min, time_max: time axis range (user provides)
        survival_min, survival_max: survival axis range
    
    Returns:
        DataFrame with time and survival columns, or None if extraction fails
    """
    try:
        if image_array is None:
            return None
        
        # Detect curve pixels
        curve_pixels = detect_curve_pixels(image_array)
        if curve_pixels is None or len(curve_pixels) < 10:
            return None
        
        # Detect axes
        axes_info = detect_axes(image_array, curve_pixels)
        
        # Convert pixel coordinates to data coordinates
        px_x = curve_pixels[:, 0].astype(float)
        px_y = curve_pixels[:, 1].astype(float)
        
        # Map pixels to data space
        x_pixel_range = axes_info["pixel_x_max"] - axes_info["pixel_x_min"]
        y_pixel_range = axes_info["pixel_y_max"] - axes_info["pixel_y_min"]
        
        # Avoid division by zero
        if x_pixel_range <= 0 or y_pixel_range <= 0:
            return None
        
        # Set time_max if not provided
        if time_max is None:
            time_max = 100.0
        
        # Convert to data coordinates
        time = time_min + (px_x - axes_info["pixel_x_min"]) / x_pixel_range * (time_max - time_min)
        # Y-axis is inverted in image coordinates
        survival = survival_max - (px_y - axes_info["pixel_y_min"]) / y_pixel_range * (survival_max - survival_min)
        
        # Create DataFrame
        df = pd.DataFrame({
            "time": time,
            "survival": survival
        })
        
        # Remove any NaN or infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) == 0:
            return None
        
        # Filter to valid ranges
        df = df[(df["survival"] >= survival_min - 0.1) & (df["survival"] <= survival_max + 0.1)]
        df = df[df["time"] >= time_min - 0.1]
        
        if len(df) == 0:
            return None
        
        # Clip to valid ranges
        df["survival"] = df["survival"].clip(survival_min, survival_max)
        df["time"] = df["time"].clip(time_min, None)
        
        # Remove duplicates and sort
        df = df.drop_duplicates().sort_values("time").reset_index(drop=True)
        
        # Subsample points for smoothness (keep every Nth point)
        if len(df) > 50:
            df = df.iloc[::len(df)//50].reset_index(drop=True)
        
        return df if len(df) > 0 else None
        
    except Exception as e:
        return None


# -----
# Basic probability conversions
# -----

def probability_to_rate(p, cycle_length=1.0):
    """
    Converts probability over a cycle to a constant hazard rate.
    p = 1 - exp(-r * cycle_length)
    """
    p = np.clip(p, 0, 1 - EPS)
    return -np.log(1 - p) / cycle_length


def rate_to_probability(r, cycle_length=1.0):
    """
    Converts constant hazard rate to probability over a cycle.
    """
    r = np.maximum(r, 0)
    return 1 - np.exp(-r * cycle_length)


# -----------------------------
# KM curve cleaning and editing
# -----------------------------

def prepare_km_points(df, time_col="time", survival_col="survival", add_origin=True):
    """
    Cleans digitized KM points.

    Expected columns:
    - time
    - survival

    Survival can be given as 0-1 or 0-100.
    """
    out = df[[time_col, survival_col]].copy()
    out.columns = ["time", "survival"]

    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["survival"] = pd.to_numeric(out["survival"], errors="coerce")
    out = out.dropna()

    # Convert percentage survival to proportion
    if out["survival"].max() > 1.5:
        out["survival"] = out["survival"] / 100

    out["survival"] = out["survival"].clip(0, 1)
    out = out[out["time"] >= 0]
    out = out.sort_values("time")

    # Average duplicate time points
    out = out.groupby("time", as_index=False)["survival"].mean()

    if add_origin and not np.isclose(out["time"].min(), 0):
        out = pd.concat(
            [pd.DataFrame({"time": [0.0], "survival": [1.0]}), out],
            ignore_index=True
        )

    # Force monotonic non-increasing survival
    out["survival_monotone"] = np.minimum.accumulate(out["survival"].values)

    return out


def survival_at_times(km_df, times, survival_col="survival_monotone"):
    """
    Linear interpolation of survival at selected times.
    """
    x = km_df["time"].values
    y = km_df[survival_col].values
    times = np.asarray(times)

    return np.interp(times, x, y, left=y[0], right=y[-1])


# -----------------------------
# Derive interval probabilities
# -----------------------------

def derive_interval_probabilities(km_df, interval_times=None):
    """
    Derives interval survival and event probabilities from KM survival.

    Conditional event probability:
        p_event(t1 to t2) = 1 - S(t2) / S(t1)

    Conditional survival probability:
        p_survive(t1 to t2) = S(t2) / S(t1)
    """
    if interval_times is None:
        interval_times = km_df["time"].values

    interval_times = np.asarray(sorted(set(interval_times)))
    s = survival_at_times(km_df, interval_times)

    rows = []

    for i in range(len(interval_times) - 1):
        t0 = interval_times[i]
        t1 = interval_times[i + 1]

        s0 = max(s[i], EPS)
        s1 = max(s[i + 1], 0)

        p_survive = s1 / s0
        p_survive = np.clip(p_survive, 0, 1)

        p_event = 1 - p_survive

        rows.append({
            "time_start": t0,
            "time_end": t1,
            "S_start": s0,
            "S_end": s1,
            "p_survive_interval": p_survive,
            "p_event_interval": p_event,
            "rate_interval": probability_to_rate(p_event, cycle_length=t1 - t0)
        })

    return pd.DataFrame(rows)


def derive_at_risk_probabilities(km_df, at_risk_df):
    """
    Uses the number-at-risk table and KM survival to derive interval probabilities.

    Expected at_risk_df columns:
    - time
    - n_risk

    The event and censoring estimates are approximate because digitized KM curves
    usually do not contain exact censoring information.
    """
    ar = at_risk_df.copy()
    ar.columns = [c.lower().strip() for c in ar.columns]

    if "time" not in ar.columns or "n_risk" not in ar.columns:
        raise ValueError("At-risk table must contain columns: time, n_risk")

    ar["time"] = pd.to_numeric(ar["time"], errors="coerce")
    ar["n_risk"] = pd.to_numeric(ar["n_risk"], errors="coerce")
    ar = ar.dropna().sort_values("time")

    interval_df = derive_interval_probabilities(km_df, ar["time"].values)

    rows = []

    for i in range(len(interval_df)):
        n_start = ar.iloc[i]["n_risk"]

        if i + 1 < len(ar):
            n_next_observed = ar.iloc[i + 1]["n_risk"]
        else:
            n_next_observed = np.nan

        p_event = interval_df.iloc[i]["p_event_interval"]
        p_survive = interval_df.iloc[i]["p_survive_interval"]

        expected_events_no_censor = n_start * p_event
        expected_remaining_no_censor = n_start * p_survive

        if not np.isnan(n_next_observed):
            implied_censoring = expected_remaining_no_censor - n_next_observed
            implied_censoring = max(implied_censoring, 0)
        else:
            implied_censoring = np.nan

        rows.append({
            "time_start": interval_df.iloc[i]["time_start"],
            "time_end": interval_df.iloc[i]["time_end"],
            "n_risk_start": n_start,
            "n_risk_next_observed": n_next_observed,
            "S_start": interval_df.iloc[i]["S_start"],
            "S_end": interval_df.iloc[i]["S_end"],
            "p_survive_interval": p_survive,
            "p_event_interval": p_event,
            "expected_events_no_censor": expected_events_no_censor,
            "implied_censoring_approx": implied_censoring,
            "rate_interval": interval_df.iloc[i]["rate_interval"]
        })

    return pd.DataFrame(rows)


# -----------------------------
# Weibull fitting
# -----------------------------

def weibull_survival(t, shape, scale):
    t = np.asarray(t)
    return np.exp(-((np.maximum(t, 0) / scale) ** shape))


def weibull_hazard(t, shape, scale):
    t = np.asarray(t)
    t = np.maximum(t, EPS)
    return (shape / scale) * ((t / scale) ** (shape - 1))


def fit_weibull_to_km(km_df):
    """
    Fits a Weibull curve to digitized KM survival points.

    Survival:
        S(t) = exp(-(t / scale)^shape)

    Hazard:
        h(t) = shape / scale * (t / scale)^(shape - 1)
    """
    df = km_df.copy()
    df = df[(df["time"] > 0) & (df["survival_monotone"] > 0.001) & (df["survival_monotone"] < 0.999)]

    if len(df) < 3:
        raise ValueError("Need at least 3 usable KM points to fit Weibull.")

    t = df["time"].values
    s_obs = df["survival_monotone"].values

    def objective(par):
        log_shape, log_scale = par
        shape = np.exp(log_shape)
        scale = np.exp(log_scale)

        s_pred = weibull_survival(t, shape, scale)

        # Fit on log survival scale
        err = np.log(np.clip(s_obs, EPS, 1)) - np.log(np.clip(s_pred, EPS, 1))
        return np.sum(err ** 2)

    init_shape = 1.2
    init_scale = np.median(t) / max((-np.log(np.median(s_obs))) ** (1 / init_shape), EPS)

    result = minimize(
        objective,
        x0=np.log([init_shape, init_scale]),
        method="Nelder-Mead"
    )

    shape, scale = np.exp(result.x)

    return {
        "shape": shape,
        "scale": scale,
        "success": result.success,
        "sse": result.fun
    }


# -----------------------------
# Background mortality
# -----------------------------

def prepare_background_mortality(bg_df):
    """
    Expected background mortality columns:

    Option A:
    - age
    - qx

    where qx is annual mortality probability.

    Option B:
    - age
    - rate

    where rate is annual hazard rate.
    """
    bg = bg_df.copy()
    bg.columns = [c.lower().strip() for c in bg.columns]

    if "age" not in bg.columns:
        raise ValueError("Background mortality file must contain column: age")

    bg["age"] = pd.to_numeric(bg["age"], errors="coerce")

    if "qx" in bg.columns:
        bg["qx"] = pd.to_numeric(bg["qx"], errors="coerce")
        bg["rate"] = probability_to_rate(bg["qx"], cycle_length=1.0)
    elif "rate" in bg.columns:
        bg["rate"] = pd.to_numeric(bg["rate"], errors="coerce")
        bg["qx"] = rate_to_probability(bg["rate"], cycle_length=1.0)
    else:
        raise ValueError("Background mortality file must contain either qx or rate.")

    bg = bg.dropna().sort_values("age")
    return bg[["age", "qx", "rate"]]


def background_rate_at_age(bg_df, age):
    """
    Interpolates annual background mortality hazard by age.
    """
    age_grid = bg_df["age"].values
    rate_grid = bg_df["rate"].values

    return np.interp(
        age,
        age_grid,
        rate_grid,
        left=rate_grid[0],
        right=rate_grid[-1]
    )


# -----------------------------
# Extrapolation with background mortality and RR
# -----------------------------

def build_extrapolated_survival_table(
    km_df,
    weibull_params,
    start_age,
    bg_df=None,
    registry_rr=1.0,
    max_time=30,
    cycle_length=1.0,
    combine_mode="excess_plus_background"
):
    """
    Builds extrapolated survival using Weibull hazard and optional background mortality.

    combine_mode options:

    1. "parametric_only"
       h_total = h_weibull

    2. "excess_plus_background"
       h_total = h_weibull + registry_rr * h_background

       Use this when the fitted curve represents excess or disease-related mortality,
       and you want to add expected age-specific background mortality.

    3. "background_rr_only_after_km"
       h_total = registry_rr * h_background

       Use this when after the observed KM period you assume mortality follows
       registry/general-population mortality multiplied by RR.
    """
    shape = weibull_params["shape"]
    scale = weibull_params["scale"]

    km_last_time = float(km_df["time"].max())
    km_last_survival = float(km_df["survival_monotone"].iloc[-1])

    times = np.arange(0, max_time + cycle_length, cycle_length)

    rows = []
    S_total = 1.0

    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]
        t_mid = (t0 + t1) / 2
        age_mid = start_age + t_mid

        # During KM observed period, use digitized KM directly
        if t1 <= km_last_time:
            s0 = survival_at_times(km_df, [t0])[0]
            s1 = survival_at_times(km_df, [t1])[0]
            p_total = 1 - (s1 / max(s0, EPS))
            h_total = probability_to_rate(p_total, cycle_length=t1 - t0)

            h_weibull = np.nan
            h_bg = np.nan
            h_registry = np.nan

            S_total = s1

        else:
            h_weibull = weibull_hazard(t_mid, shape, scale)

            if bg_df is not None:
                h_bg = background_rate_at_age(bg_df, age_mid)
                h_registry = registry_rr * h_bg
            else:
                h_bg = 0.0
                h_registry = 0.0

            if combine_mode == "parametric_only":
                h_total = h_weibull

            elif combine_mode == "excess_plus_background":
                h_total = h_weibull + h_registry

            elif combine_mode == "background_rr_only_after_km":
                h_total = h_registry

            else:
                raise ValueError("Invalid combine_mode.")

            p_total = rate_to_probability(h_total, cycle_length=t1 - t0)
            S_total = S_total * (1 - p_total)

        rows.append({
            "cycle": i + 1,
            "time_start": t0,
            "time_end": t1,
            "age_mid": age_mid,
            "h_weibull": h_weibull,
            "h_background": h_bg,
            "registry_rr": registry_rr,
            "h_registry_rr": h_registry,
            "h_total": h_total,
            "p_death_cycle": p_total,
            "p_survive_cycle": 1 - p_total,
            "S_total": S_total
        })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------

def plot_km_and_extrapolation(km_df, extrapolated_df=None, weibull_params=None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=km_df["time"],
        y=km_df["survival"],
        mode="markers",
        name="Digitized points"
    ))

    fig.add_trace(go.Scatter(
        x=km_df["time"],
        y=km_df["survival_monotone"],
        mode="lines+markers",
        name="Edited monotone KM"
    ))

    if weibull_params is not None:
        max_t = max(km_df["time"].max(), 30)
        t_grid = np.linspace(0, max_t, 300)
        s_weibull = weibull_survival(
            t_grid,
            weibull_params["shape"],
            weibull_params["scale"]
        )

        fig.add_trace(go.Scatter(
            x=t_grid,
            y=s_weibull,
            mode="lines",
            name="Weibull fitted curve"
        ))

    if extrapolated_df is not None:
        fig.add_trace(go.Scatter(
            x=extrapolated_df["time_end"],
            y=extrapolated_df["S_total"],
            mode="lines+markers",
            name="Final survival with background/RR"
        ))

    fig.update_layout(
        title="KM curve, manual edits, and extrapolated survival",
        xaxis_title="Time",
        yaxis_title="Survival",
        yaxis=dict(range=[0, 1.02]),
        template="plotly_white"
    )

    return fig


# -----------------------------
# Streamlit UI block
# -----------------------------

def render_km_registry_tool():
    st.header("KM digitization, manual editing, and registry mortality adjustment")

    st.markdown("""
    Upload digitized KM points, manually edit the curve, derive interval probabilities,
    and apply background mortality plus registry mortality RR to extrapolated survival.
    """)

    # Add tabs for manual upload or automatic digitization
    if IMAGE_PROCESSING_AVAILABLE:
        tabs_list = ["📤 Manual CSV Upload", "🤖 Auto-Digitize from Image"]
    else:
        tabs_list = ["📤 Manual CSV Upload"]
    
    if len(tabs_list) == 2:
        tab1, tab2 = st.tabs(tabs_list)
    else:
        tab1 = st.tabs(tabs_list)[0]
        tab2 = None
    
    if tab2 is not None:
        with tab2:
            st.subheader("Automatic KM Curve Digitization")
            st.markdown("""
            Upload a KM plot image and the digitizer will automatically extract curve points.
            """)
            
            image_file = st.file_uploader(
                "Upload KM plot image (PNG, JPG, etc.)",
                type=["png", "jpg", "jpeg", "bmp"],
                key="km_image_file"
            )
            
            if image_file is not None:
                try:
                    # Load image
                    image = Image.open(image_file)
                    image_array = np.array(image)
                    
                    # Display uploaded image
                    st.image(image, caption="Uploaded KM plot", use_container_width=True)
                    
                    # Axis parameter input
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("X-Axis (Time)")
                        time_min = st.number_input("Time minimum", value=0.0, key="time_min")
                        time_max = st.number_input("Time maximum", value=10.0, key="time_max")
                    
                    with col2:
                        st.subheader("Y-Axis (Survival)")
                        survival_min = st.number_input("Survival minimum", value=0.0, min_value=0.0, max_value=1.0, key="surv_min")
                        survival_max = st.number_input("Survival maximum", value=1.0, min_value=0.0, max_value=1.0, key="surv_max")
                    
                    if st.button("🔍 Auto-Digitize", key="auto_digitize_btn"):
                        with st.spinner("Extracting KM curve points..."):
                            try:
                                digitized_points = extract_km_points_from_image(
                                    image_array,
                                    time_min=time_min,
                                    time_max=time_max,
                                    survival_min=survival_min,
                                    survival_max=survival_max
                                )
                                
                                if digitized_points is not None and isinstance(digitized_points, pd.DataFrame) and len(digitized_points) > 0:
                                    st.success(f"✅ Extracted {len(digitized_points)} KM points!")
                                    st.dataframe(digitized_points, use_container_width=True)
                                    
                                    # Store in session for later use
                                    st.session_state.auto_digitized_points = digitized_points
                                    
                                    # Download option
                                    st.download_button(
                                        "📥 Download Extracted Points as CSV",
                                        data=digitized_points.to_csv(index=False),
                                        file_name="auto_digitized_km_points.csv",
                                        mime="text/csv",
                                        key="download_auto_digitized"
                                    )
                                else:
                                    st.error("❌ Could not detect curve in image. Try adjusting axis parameters or uploading a clearer image.")
                            except Exception as e:
                                st.error(f"❌ Error during digitization: {str(e)}")
                                
                except Exception as e:
                    st.error(f"❌ Error loading image: {str(e)}")
    
    with tab1:
        st.subheader("Manual KM Upload")

        km_file = st.file_uploader(
            "Upload digitized KM CSV with columns: time, survival",
            type=["csv"],
            key="km_digitized_file"
        )

        if km_file is None:
            if "auto_digitized_points" not in st.session_state:
                st.info("Upload a digitized KM CSV or use auto-digitization from the Image tab.")
                return
            raw_km = st.session_state.auto_digitized_points
            st.info("✅ Using auto-digitized points from image")
        else:
            raw_km = pd.read_csv(km_file)

    st.subheader("1. Manual KM curve editing")

    km_prepared = prepare_km_points(raw_km)

    edited = st.data_editor(
        km_prepared[["time", "survival"]],
        num_rows="dynamic",
        use_container_width=True,
        key="km_curve_editor"
    )

    km_clean = prepare_km_points(edited)

    st.write("Edited and monotone-adjusted KM points")
    st.dataframe(km_clean, use_container_width=True)

    st.subheader("2. Derived interval probabilities from KM")

    interval_df = derive_interval_probabilities(km_clean)
    st.dataframe(interval_df, use_container_width=True)

    st.download_button(
        "Download KM interval probabilities",
        data=interval_df.to_csv(index=False),
        file_name="km_interval_probabilities.csv",
        mime="text/csv"
    )

    st.subheader("3. Optional number-at-risk adjustment")

    at_risk_file = st.file_uploader(
        "Optional: upload at-risk CSV with columns: time, n_risk",
        type=["csv"],
        key="at_risk_file"
    )

    if at_risk_file is not None:
        at_risk_df = pd.read_csv(at_risk_file)
        at_risk_prob_df = derive_at_risk_probabilities(km_clean, at_risk_df)

        st.write("At-risk interval probability table")
        st.dataframe(at_risk_prob_df, use_container_width=True)

        st.download_button(
            "Download at-risk interval probability table",
            data=at_risk_prob_df.to_csv(index=False),
            file_name="at_risk_interval_probabilities.csv",
            mime="text/csv"
        )

    st.subheader("4. Weibull extrapolation")

    fit = fit_weibull_to_km(km_clean)

    st.write({
        "Weibull shape": fit["shape"],
        "Weibull scale": fit["scale"],
        "Fit success": fit["success"],
        "SSE": fit["sse"]
    })

    st.latex(r"S(t)=\exp\left[-\left(\frac{t}{\lambda}\right)^\gamma\right]")
    st.latex(r"h(t)=\frac{\gamma}{\lambda}\left(\frac{t}{\lambda}\right)^{\gamma-1}")

    st.subheader("5. Background mortality and registry RR")

    start_age = st.number_input(
        "Starting age",
        min_value=0.0,
        max_value=120.0,
        value=65.0,
        step=1.0
    )

    max_time = st.number_input(
        "Maximum extrapolation time",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

    cycle_length = st.selectbox(
        "Cycle length",
        options=[1.0, 1 / 12, 0.25, 0.5],
        format_func=lambda x: {
            1.0: "Annual",
            1 / 12: "Monthly",
            0.25: "Quarterly",
            0.5: "Half-year"
        }[x]
    )

    registry_rr = st.number_input(
        "Registry mortality RR versus background population",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1
    )

    combine_mode = st.selectbox(
        "Mortality combination mode",
        options=[
            "parametric_only",
            "excess_plus_background",
            "background_rr_only_after_km"
        ],
        index=1,
        help=(
            "Use excess_plus_background when the extrapolated curve represents excess disease mortality. "
            "Use parametric_only if the fitted OS curve already contains all-cause mortality. "
            "Use background_rr_only_after_km if mortality after KM follows registry RR times background mortality."
        )
    )

    bg_file = st.file_uploader(
        "Optional: upload background mortality CSV with columns age,qx or age,rate",
        type=["csv"],
        key="background_mortality_file"
    )

    bg_df = None
    if bg_file is not None:
        bg_df = prepare_background_mortality(pd.read_csv(bg_file))
        st.write("Background mortality table")
        st.dataframe(bg_df, use_container_width=True)

    extrapolated_df = build_extrapolated_survival_table(
        km_df=km_clean,
        weibull_params=fit,
        start_age=start_age,
        bg_df=bg_df,
        registry_rr=registry_rr,
        max_time=max_time,
        cycle_length=cycle_length,
        combine_mode=combine_mode
    )

    st.subheader("6. Final extrapolated survival table")

    st.dataframe(extrapolated_df, use_container_width=True)

    st.download_button(
        "Download final extrapolated survival table",
        data=extrapolated_df.to_csv(index=False),
        file_name="final_survival_with_background_rr.csv",
        mime="text/csv"
    )

    st.subheader("7. Survival plot")

    fig = plot_km_and_extrapolation(
        km_df=km_clean,
        extrapolated_df=extrapolated_df,
        weibull_params=fit
    )

    st.plotly_chart(fig, use_container_width=True)
