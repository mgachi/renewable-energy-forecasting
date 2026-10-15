import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset für Multi-Horizon Time-Series Forecasting.
    
    Setup:
      - Historical Window: 48 Stunden (Wetter + Target-Historie + Kalender)
      - Forecast Horizon:  24 Stunden (Future Weather + Kalender -> Future Targets)
    """
    def __init__(self, df, feature_cols, target_cols=['target_solar_norm', 'target_wind_norm'], 
                 hist_steps=48, pred_steps=24):
        """
        Args:
            df (pd.DataFrame): Skaliertes DataFrame (z.B. train_df, val_df oder test_df).
            feature_cols (list): Liste der skalierten Input-Features (Wetter + Kalender).
            target_cols (list): Spaltennamen der Zielgrößen (Solar + Wind normiert).
            hist_steps (int): Anzahl der historischen Zeitschritte (Default: 48h).
            pred_steps (int): Anzahl der zu prognostizierenden Zeitschritte (Default: 24h).
        """
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.total_window = hist_steps + pred_steps
        
        # 1. Feature-Gruppen identifizieren
        # Future-Features sind Wettervariablen + Kalender-Sin/Cos (alles außer reine Erzeugung/Kapazität)
        self.future_feature_cols = feature_cols
        
        # Historical Features umfassen alle Input-Features + die historischen Target-Werte
        self.target_cols = target_cols
        self.hist_feature_cols = feature_cols + target_cols
        
        # Indices für schnelles Slicing extrahieren
        self.hist_data = df[self.hist_feature_cols].values.astype(np.float32)
        self.future_data = df[self.future_feature_cols].values.astype(np.float32)
        self.target_data = df[self.target_cols].values.astype(np.float32)
        
        # Anzahl gültiger Sliding Windows
        self.num_samples = len(df) - self.total_window + 1
        
        if self.num_samples <= 0:
            raise ValueError(
                f"DataFrame ist zu kurz ({len(df)} Zeilen) für Fenstergröße "
                f"von {self.total_window} (48h Hist + 24h Pred)."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Liefert ein Tupel aus Tensors für ein einzelnes Sliding Window:
          - x_hist:   Shape (48, num_hist_features)  -> Vergangenes Wetter + historische Produktion
          - x_future: Shape (24, num_future_features)-> Zukünftiges Wetter (Wettervorhersage)
          - y_target: Shape (24, num_targets)        -> Tatsächliche Produktion (Solar + Wind)
        """
        hist_end = idx + self.hist_steps
        pred_end = hist_end + self.pred_steps
        
        # Historical Input: [t-48 : t]
        x_hist = self.hist_data[idx:hist_end]
        
        # Future Input (Weather Forecast): [t : t+24]
        x_future = self.future_data[hist_end:pred_end]
        
        # Ground Truth Targets: [t : t+24]
        y_target = self.target_data[hist_end:pred_end]
        
        return (
            torch.tensor(x_hist, dtype=torch.float32),
            torch.tensor(x_future, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )


def create_dataloaders(train_df, val_df, test_df, feature_cols, 
                       batch_size=64, hist_steps=48, pred_steps=24, num_workers=0):
    """
    Hilfsfunktion zur Erstellung aller drei DataLoader (Train, Val, Test).
    """
    train_dataset = TimeSeriesDataset(train_df, feature_cols, hist_steps=hist_steps, pred_steps=pred_steps)
    val_dataset   = TimeSeriesDataset(val_df, feature_cols, hist_steps=hist_steps, pred_steps=pred_steps)
    test_dataset  = TimeSeriesDataset(test_df, feature_cols, hist_steps=hist_steps, pred_steps=pred_steps)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader