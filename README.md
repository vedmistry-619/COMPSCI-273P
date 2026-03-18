## Project Overview

This project predicts Apple (AAPL) stock closing prices using time‑series deep learning. It builds features from historical price data (technical indicators + Fourier components) and merges daily news sentiment, then trains one of several models:

- GRU (`3. GRU.py`)
- Bidirectional LSTM (`3. LSTM.py`)
- GAN (`4. GAN.py`) with a GRU‑based generator
- WGAN‑GP (`5. WGAN.py`)
- WGAN evaluation (`6. Test.py`)

Scripts are intended to be run from the repository root in sequence.

---

## Repository Structure

- `1. Data Loading.py`: Reads `Apple_Data.csv`, computes technical indicators and Fourier components, writes `Final_Fourier.csv`.
- `2. Preprocessing.py`: Merges in news sentiment (`News.csv`), fills missing values, scales features/labels, creates train/test arrays and indices.
- `3. GRU.py`: Trains a GRU regressor, saves `GRUModel.h5`, plots predictions, prints RMSE.
- `3. LSTM.py`: Trains a Bidirectional LSTM, saves `LSTMModel.h5`, plots predictions, prints RMSE.
- `4. GAN.py`: Trains a GAN; periodically saves `GAN_model_<epoch>.h5`, plots D/G loss and training fit.
- `5. WGAN.py`: Trains a WGAN‑GP; periodically saves `WGAN_model_<epoch>.h5`, writes `train_loss.png`, `train_plot.png`, prints RMSE.
- `6. Test.py`: Loads a saved WGAN generator (e.g., `WGAN_model_89.h5`) and evaluates on the test set; saves `test_plot.png`, `test_predicted.csv`, prints RMSE.
- `NLP/finbert.ipynb`: Placeholder for generating news sentiment (if you need to recreate `News.csv`).

Expected data files in repo root:
- `Apple_Data.csv`
- `News.csv`

---

## Setup Instructions (macOS)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels tensorflow
```

Notes:
- TensorFlow wheels differ by macOS chip/OS/Python. If install errors occur, install the appropriate TensorFlow build for your platform, then re-run the command above.

---

## Required Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn
- statsmodels
- tensorflow (Keras)

---

## Dataset: Where to Download and Format Requirements

The two datasets are provided in the repository root:

1) `Apple_Data.csv`

2) `News.csv`

---

## How to Preprocess the Data

Run from the repo root. Filenames contain spaces; keep the quotes.

1) Build technical indicators + Fourier components
```bash
python "1. Data Loading.py"
```
Inputs: `Apple_Data.csv`  
Outputs: `Final_Fourier.csv` and a plot of Apple closing price.

2) Merge news, scale, and create train/test artifacts
```bash
python "2. Preprocessing.py"
```
Inputs: `Final_Fourier.csv`, `News.csv`  
Outputs:
- `dataset.csv`
- `X_scaler.pkl`, `y_scaler.pkl`
- `X_train.npy`, `X_test.npy`
- `y_train.npy`, `y_test.npy`
- `yc_train.npy`, `yc_test.npy`
- `index_train.npy`, `index_test.npy`
- `train_predict_index.npy`, `test_predict_index.npy`

---

## How to Train the Model

After preprocessing, choose one or more trainers:

GRU:
```bash
python "3. GRU.py"
```
Outputs: `GRUModel.h5`, loss curves, train/test prediction plots, printed RMSE.

Bidirectional LSTM:
```bash
python "3. LSTM.py"
```
Outputs: `LSTMModel.h5`, loss curves, train/test prediction plots, printed RMSE.

GAN:
```bash
python "4. GAN.py"
```
Outputs: `GAN_model_<epoch>.h5` (every 15 epochs), D/G loss plot, training fit plot, printed Train RMSE.

WGAN‑GP:
```bash
python "5. WGAN.py"
```
Outputs: `WGAN_model_<epoch>.h5` (every 15 epochs), `train_loss.png`, `train_plot.png`, printed RMSE.

---

## How to Evaluate the Model

GRU / LSTM:
- Both training scripts evaluate on `X_test`, inverse‑transform with `y_scaler.pkl`, plot predicted vs real, and print RMSE.
- No extra step needed beyond running the training script.

WGAN evaluation (requires a saved generator checkpoint):
1) Ensure a file like `WGAN_model_89.h5` exists (produced by `5. WGAN.py`).
2) Run:
```bash
python "6. Test.py"
```
Outputs: test plot window, `test_plot.png`, `test_predicted.csv`, printed RMSE.
Note: `6. Test.py` currently loads `WGAN_model_89.h5`. Rename your checkpoint or update the filename in the script to match your saved epoch.

---

## Expected Outputs (Summary)

From the full GRU/LSTM pipeline:
- Feature file: `Final_Fourier.csv`
- Preprocessing artifacts: `dataset.csv`, scalers (`X_scaler.pkl`, `y_scaler.pkl`), arrays (`X_*`, `y_*`, `yc_*`), indices (`index_*`, `train_predict_index.npy`, `test_predict_index.npy`)
- Models: `GRUModel.h5` or `LSTMModel.h5`
- Plots: training/validation loss and prediction curves shown during runs

From GAN/WGAN:
- Checkpoints: `GAN_model_<epoch>.h5` or `WGAN_model_<epoch>.h5`
- Plots/CSVs: `train_loss.png`, `train_plot.png`, `test_plot.png`, `test_predicted.csv`
- Printed metrics: RMSE (and RMSPE in GAN training return)

---

## Reproduce Results (End‑to‑End Commands)

GRU:
```bash
python "1. Data Loading.py" && \
python "2. Preprocessing.py" && \
python "3. GRU.py"
```

LSTM:
```bash
python "1. Data Loading.py" && \
python "2. Preprocessing.py" && \
python "3. LSTM.py"
```

WGAN + Evaluation (example):
```bash
python "1. Data Loading.py" && \
python "2. Preprocessing.py" && \
python "5. WGAN.py" && \
python "6. Test.py"
```
