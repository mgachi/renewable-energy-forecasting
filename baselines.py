import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_metrics(y_true, y_pred):
    """
    Berechnet MSE, MAE und RMSE zwischen Ground Truth und Vorhersage.
    Args:
        y_true (np.ndarray): Shape (N, 24, 2)
        y_pred (np.ndarray): Shape (N, 24, 2)
    Returns:
        dict: Metriken aufgeteilt nach Solar, Wind und Gesamt.
    """
    mse = mean_squared_error(y_true.reshape(-1, 2), y_pred.reshape(-1, 2))
    mae = mean_absolute_error(y_true.reshape(-1, 2), y_pred.reshape(-1, 2))
    rmse = np.sqrt(mse)

    # Separat für Solar (Index 0) und Wind (Index 1)
    solar_mae = mean_absolute_error(y_true[:, :, 0], y_pred[:, :, 0])
    solar_rmse = np.sqrt(mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0]))

    wind_mae = mean_absolute_error(y_true[:, :, 1], y_pred[:, :, 1])
    wind_rmse = np.sqrt(mean_squared_error(y_true[:, :, 1], y_pred[:, :, 1]))

    return {
        'overall': {'MSE': mse, 'MAE': mae, 'RMSE': rmse},
        'solar':   {'MAE': solar_mae, 'RMSE': solar_rmse},
        'wind':    {'MAE': wind_mae, 'RMSE': wind_rmse}
    }


# =====================================================================
# 1. PERSISTENCE BASELINE (Saisonal: Y_{t+h} = Y_{t+h-24})
# =====================================================================
class SeasonalPersistenceBaseline:
    """
    Saisonale Persistence Baseline:
    Prognostiziert die nächsten 24h basierend auf den letzten 24h der Historie.
    """
    def predict(self, test_loader):
        all_preds = []
        all_targets = []

        for x_hist, x_future, y_target in test_loader:
            # x_hist Shape: (Batch, 48, Num_Features)
            # Die letzten beiden Spalten in x_hist sind die historischen Targets [Solar, Wind]
            # Wir nehmen die letzten 24h der Historie als Prognose für die kommenden 24h
            last_24h_targets = x_hist[:, -24:, -2:].numpy()
            
            all_preds.append(last_24h_targets)
            all_targets.append(y_target.numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        return y_pred, y_true


# =====================================================================
# 2. RIDGE REGRESSION BASELINE
# =====================================================================
class RidgeRegressionBaseline:
    """
    Lineares Baseline-Modell:
    Verwendet flachgedrückte (flattened) Historien- und Zukunfts-Features.
    """
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def _flatten_inputs(self, loader):
        X_list, Y_list = [], []
        for x_hist, x_future, y_target in loader:
            batch_size = x_hist.shape[0]
            
            # Flachdrücken: (Batch, 48 * F_hist) und (Batch, 24 * F_future)
            x_hist_flat = x_hist.reshape(batch_size, -1).numpy()
            x_future_flat = x_future.reshape(batch_size, -1).numpy()
            
            # Konkatenieren zu einem großen Input-Vektor pro Sample
            X_concat = np.hstack([x_hist_flat, x_future_flat])
            y_flat = y_target.reshape(batch_size, -1).numpy() # Shape: (Batch, 24 * 2)
            
            X_list.append(X_concat)
            Y_list.append(y_flat)

        return np.vstack(X_list), np.vstack(Y_list)

    def fit(self, train_loader):
        print("--> Trainiere Ridge Regression Baseline...")
        X_train, Y_train = self._flatten_inputs(train_loader)
        self.model.fit(X_train, Y_train)
        print("✓ Ridge Regression erfolgreich trainiert.")

    def predict(self, test_loader):
        X_test, Y_test = self._flatten_inputs(test_loader)
        Y_pred_flat = self.model.predict(X_test)
        
        # Zurück in Shape (N, 24, 2) bringen
        num_samples = Y_test.shape[0]
        y_pred = Y_pred_flat.reshape(num_samples, 24, 2)
        y_true = Y_test.reshape(num_samples, 24, 2)
        
        # Auf [0, 1] clippen (physikalische Grenze)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        return y_pred, y_true