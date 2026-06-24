# Enhanced Survival Distribution Fitting Agent

A Streamlit app for survival distribution fitting, Kaplan-Meier curve digitization, treatment-specific curve extraction, background mortality adjustment, registry calibration, and export-ready survival formulas/probabilities.

## View the app

This repository does not currently contain a verified production deployment URL. The previous template badge pointed to the generic Streamlit blank-app template, not this customized KM registry app.

To view the latest committed changes, deploy this repository to Streamlit Community Cloud or run it locally.

### Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

### Deploy on Streamlit Community Cloud

1. Push this repository and branch to GitHub.
2. Open https://share.streamlit.io.
3. Create or edit the app deployment.
4. Select this repository and the branch containing the latest commits.
5. Set the main file to `streamlit_app.py`.
6. Set the requirements file to `requirements.txt`.
7. Reboot or redeploy the app.

After deployment, use the sidebar selector and choose **KM Registry Mortality** to see the new KM digitization, registry calibration, and parametric fitting workflow.
