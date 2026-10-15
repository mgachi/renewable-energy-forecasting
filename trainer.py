import os
import time
import json
import copy
import torch
import torch.nn as nn
import numpy as np


def generate_experiment_name(model_name, params_dict):
    """
    Erstellt einen eindeutigen String anhand der Modellparameter.
    Beispiel: RecurrentForecaster_cell-LSTM_hid-64_layers-2_lr-0.001
    """
    param_str = "_".join([f"{k}-{v}" for k, v in sorted(params_dict.items())])
    return f"{model_name}_{param_str}"


class ModelTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-5, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for x_hist, x_future, y_target in train_loader:
            x_hist = x_hist.to(self.device)
            x_future = x_future.to(self.device)
            y_target = y_target.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(x_hist, x_future)
            loss = self.criterion(y_pred, y_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * x_hist.size(0)
            
        return total_loss / len(train_loader.dataset)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x_hist, x_future, y_target in val_loader:
                x_hist = x_hist.to(self.device)
                x_future = x_future.to(self.device)
                y_target = y_target.to(self.device)
                
                y_pred = self.model(x_hist, x_future)
                loss = self.criterion(y_pred, y_target)
                
                total_loss += loss.item() * x_hist.size(0)
                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_target.cpu().numpy())
                
        val_loss = total_loss / len(val_loader.dataset)
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        return val_loss, y_pred, y_true

    def fit_or_load(self, train_loader, val_loader, hyperparams, epochs=25, patience=5, save_dir="./saved_models"):
        """
        Prüft vor dem Training, ob das Modell mit denselben Parametern bereits trainiert wurde.
        Lädt es gegebenenfalls oder führt ein neues Training mit Zeit-Tracking aus.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Generiere eindeutigen Experiment-Namen
        model_class_name = self.model.__class__.__name__
        exp_name = generate_experiment_name(model_class_name, hyperparams)
        
        model_path = os.path.join(save_dir, f"{exp_name}.pt")
        meta_path = os.path.join(save_dir, f"{exp_name}_meta.json")
        
        # 1. PRÜFUNG: EXISTEREN DIE DATEIEN BEREITS?
        if os.path.exists(model_path) and os.path.exists(meta_path):
            print(f"--> [CACHE HIT] Modell '{exp_name}' existiert bereits. Lade Gewichte & Metadaten...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        
        # 2. NEUES TRAINING MIT ZEITMESSUNG
        print(f"--> [TRAINING START] Keinen Cache gefunden. Trainiere '{exp_name}'...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0
        best_epoch = 0
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"--> Early Stopping getriggert nach Epoche {epoch}.")
                    break
                    
        training_time_sec = round(time.time() - start_time, 2)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Bestes Modell im Arbeitsspeicher wiederherstellen
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            torch.save(best_model_weights, model_path)
            
        # Metadaten zusammenstellen & speichern
        metadata = {
            'experiment_name': exp_name,
            'hyperparameters': hyperparams,
            'total_trainable_params': total_params,
            'training_time_seconds': training_time_sec,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history': history
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"✓ Training in {training_time_sec}s beendet. Modell & Metadaten unter '{save_dir}/' gespeichert.")
        return metadata