# Quickstart

## Windows

```powershell
git clone https://github.com/engyusufayman06/loan-prediction-nti.git
cd loan-prediction-nti
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run the app with the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Verify the ML core

```bash
python -m pytest -q
```

## Streamlit Cloud

Deploy the repository with:

- Branch: `main`
- Main file: `app.py`

The app reads `loan_data.csv` from the repository root and trains the reusable model bundle on startup, with Streamlit resource caching preventing repeated training during normal interaction.
