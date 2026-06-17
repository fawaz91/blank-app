# Deployment

This repository contains a Streamlit app that can be deployed locally, on Streamlit Community Cloud, or in a Docker container.

## Local deployment

```bash
cd /workspaces/blank-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io.
3. Connect your GitHub account.
4. Select this repository.
5. Set the main file to `streamlit_app.py`.
6. Set the requirements file to `requirements.txt`.

The service will build and deploy automatically.

## Docker deployment

Build the image:

```bash
docker build -t blank-app .
```

Run it:

```bash
docker run -p 8501:8501 blank-app
```

Then open `http://localhost:8501`.

## Render deployment

1. Push the repository to GitHub.
2. Create a new Web Service on Render.
3. Set the build command to:

```bash
pip install -r requirements.txt
```

4. Set the start command to:

```bash
streamlit run streamlit_app.py --server.port $PORT --server.headless true
```

5. Choose Python 3.11 or newer.
