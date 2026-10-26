# Adetector
414 Find Inciders in Polymarket Program

## Run stage A (local, no database)

1. `pip install -r requirements.txt`
2. Create a `data/` folder in the repo root and copy the course file `wallet_features.csv` into it. The folder is git-ignored: this repo is public, so never commit course data, `.env` files or passwords.
3. Open `code/stage_a.py` in VS Code and run it cell by cell (each `# %%`), or run `python code/stage_a.py`.

Outputs (two figures and three CSVs) are written to `stage_a_out/`.
