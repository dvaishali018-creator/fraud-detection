# Contributing

## Set up your environment

Use Python 3.10 or later. Create an isolated environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Generate a local development dataset when needed:

```powershell
python -m src.generate_data --rows 12000
```

## Team workflow

1. Start from the latest `main` branch.
2. Create a focused branch, for example `feature/improve-dashboard` or `fix/api-validation`.
3. Keep generated data, models, reports, virtual environments, and temporary files out of commits; `.gitignore` already covers them.
4. Run the relevant tests before opening a pull request:

   ```powershell
   python -m pytest -q
   ```

5. Open a pull request with a short description, the validation performed, and any setup or data changes reviewers need to reproduce the work.

Never commit real customer or transaction data. Use synthetic or approved, de-identified datasets only.
