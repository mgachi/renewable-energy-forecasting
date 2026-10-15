# Renewable Energy Generation Forecasting

An exploratory deep-learning project for forecasting Germany's normalized solar photovoltaic and onshore wind generation over a 24-hour horizon. The project compares simple baselines with recurrent neural networks and a Transformer using historical generation, weather, and calendar features.

> **Work in progress:** The current results are preliminary and should not be treated as operational forecast performance. The urgent data-alignment and leakage checks below have not yet been completed.

## Project status and scope

The current implementation includes:

- A data pipeline for hourly SMARD generation data and historical weather from the Open-Meteo Archive API.
- A nine-point weather grid over Germany, with temperature, wind speed at 10 m and 100 m, shortwave radiation, and cloud cover.
- Calendar features encoded as sine/cosine pairs for hour, month, and day of week.
- Chronological train, validation, and test splits: 2019–2022, 2023, and 2024.
- Sliding windows with a configurable historical lookback and a 24-hour prediction horizon.
- Seasonal persistence and Ridge regression baselines.
- GRU, LSTM, and encoder-decoder Transformer models implemented in PyTorch.
- Notebook-based experiments, plots, a written report, and a presentation.

The target variables are generation divided by estimated installed capacity, clipped to `[0, 1]`. **The current code estimates capacity with a linear formula; it does not load official historical capacity values.** The current inputs include archive weather observations for the future prediction window. They are not archived weather forecasts issued at the time of prediction.

## Repository contents

| File or folder | Purpose |
| --- | --- |
| `01_eda_and_visualization.ipynb.ipynb` | Exploratory data analysis and visualizations |
| `02_model_training_and_eval.ipynb` | Baselines, model training, tuning, evaluation, and plots |
| `data_pipeline.py` | Data loading, feature engineering, chronological splitting, and scaling |
| `dataset.py` | PyTorch sliding-window dataset and data loaders |
| `baselines.py` | Persistence and Ridge baselines, plus MSE/MAE/RMSE metrics |
| `models.py` | Recurrent forecaster and temporal Transformer |
| `trainer.py` | PyTorch training loop, early stopping, and local model caching |
| `data_cache/` | Local input data and derived cache files; not committed |
| `saved_models/` | Locally cached model checkpoints and metadata; not committed |
| `plot_*.png` | Generated plots currently included in the project folder |
| `deep_learning_project_bericht.pdf` | Current project report; treat results as preliminary |
| `Deep_Learning_Project_presentation.pdf` | Current project presentation; treat results as preliminary |

## Data

The pipeline expects an SMARD CSV export named:

```text
Realisierte_Erzeugung_201812010000_202601010000_Stunde.csv
```

Place the file in this project directory before running the pipeline. The loader also checks `data_cache/` and the parent directory. The current project requests data from 2019-01-01 through 2024-12-31.

Historical weather is requested from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) for nine locations. On first use, the pipeline downloads it and stores it under `data_cache/`. Running the pipeline therefore requires network access unless a suitable local cache already exists.

Raw data, derived caches, and model checkpoints are intentionally excluded from Git. Obtain the SMARD export and let the pipeline create its cache locally. Check source terms and attribution requirements before redistributing datasets or generated materials.

## Environment setup

This project folder is self-contained for GitHub: its `requirements.txt` lists the packages used by the project code and notebooks. After cloning the project repository, open PowerShell in the cloned project folder (the repository root) and create the environment there:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The requirements file allows compatible PyTorch 2.x releases starting at 2.7. The following optional commands replace the installed PyTorch package with a CPU-only or CUDA 12.6 build:

```powershell
python -m pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a driver compatible with CUDA 12.6, install the CUDA build instead:

```powershell
python -m pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Choose only one PyTorch build for the environment. Cartopy is included because the exploratory notebook imports it for map visualizations.

## Running the project

Run these commands from the cloned project folder (the repository root) with its virtual environment activated:

```powershell
python data_pipeline.py
```

This loads the SMARD CSV, downloads historical weather if needed, builds the features, and prints coverage and split summaries. Place the required SMARD CSV in this directory before running the pipeline.

To open the notebooks from this directory:

```powershell
python -m jupyter notebook
```

The Jupyter file browser opens in the project directory; open either notebook there. The training and evaluation workflow is in `02_model_training_and_eval.ipynb`; exploratory analysis is in `01_eda_and_visualization.ipynb.ipynb`. Select the `.venv` Python interpreter as the notebook kernel. The training notebook can take substantial time and compute when models are not already cached. Checkpoints are stored locally under `saved_models/` and reused based on experiment hyperparameters.

## Model and evaluation details

- **Input windows:** configurable historical weather, calendar, and normalized generation; future-window weather and calendar features.
- **Forecast horizon:** 24 hourly steps for solar and wind.
- **Baselines:** persistence from recent generation and multi-output Ridge regression.
- **Neural models:** GRU/LSTM encoder-decoder models and a Transformer encoder-decoder.
- **Training:** mean squared error, AdamW, gradient clipping, and early stopping against validation loss.
- **Current metrics:** MSE, MAE, and RMSE, reported overall and separately for solar and wind.
- **Splits:** train 2019–2022, validation 2023, test 2024, in chronological order.

The current code does **not** implement every item in the original plan: for example, it does not currently use `ReduceLROnPlateau`, Huber loss, sMAPE, directional accuracy, a financial penalty simulation, or an operational weather-forecast archive. Do not infer these features from the project plan or existing report.

## Urgent TODOs

- [ ] **Prevent future-weather leakage before reporting forecast performance.** Replace archive observations in the forecast window with weather forecasts that were actually available at each prediction time, or explicitly reframe the experiment as an oracle-weather study.
- [ ] **Verify and correct SMARD timestamp semantics.** Establish whether source timestamps are UTC or local German time, handle daylight-saving transitions explicitly, and validate alignment against the UTC weather index before joining data.
- [ ] **Fix target-time indexing in evaluation plots.** Dataset targets start after the historical lookback, but some plots currently label predictions from the beginning of the test index. Align every prediction and metric with its actual target timestamps.
- [ ] **Remove the machine-specific absolute path in the EDA notebook.** Make notebook setup work from the project/repository location on other machines, then remove or refresh saved notebook outputs that expose local filesystem paths or stale results.
- [ ] **Make checkpoint caching data-aware.** The cache is currently keyed by model hyperparameters and can reuse weights after input data or preprocessing changes. Include a dataset/preprocessing version or fingerprint in the cache key, and retrain after fixing the issues above.
- [ ] **Resolve missing-hour handling.** The current pipeline drops rows missing after the timestamp join. Quantify the resulting gaps and ensure sliding windows never treat non-consecutive timestamps as adjacent hours.
- [ ] **Document and validate capacity values.** Replace the approximate capacity formula with a sourced historical capacity series or clearly justify and quantify the approximation.
- [x] **Declare all direct project dependencies.** Project dependencies are listed in this folder's `requirements.txt`, including `requests` and the notebook's Cartopy dependency.
- [ ] **Re-run the full workflow and regenerate the report, presentation, plots, and notebook outputs** after the data and evaluation fixes. Until then, treat existing metrics and documents as drafts.
- [ ] **Add data-source attribution and usage terms** for SMARD, Open-Meteo, and any materials included in the report or presentation.

## GitHub publishing notes

This folder has its own `.gitignore` for Python environments, local datasets/caches, saved models, credentials, and editor state. Keep source code, notebooks, documentation, and intentionally selected figures under version control. Before publishing, inspect notebook outputs and the report/presentation for local paths, personal data, credentials, and unsupported result claims. Do not commit downloaded datasets, generated caches, or trained checkpoints.
