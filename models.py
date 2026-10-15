import torch
import torch.nn as nn

class RecurrentForecaster(nn.Module):
    """
    Encoder-Decoder Architektur (LSTM oder GRU) mit Future-Weather-Fusion,
    konfigurierbarem Dropout und optionaler Skip-Connection.
    """
    def __init__(self, hist_dim, future_dim, hidden_dim=64, num_layers=2, 
                 cell_type='LSTM', dropout=0.2, use_skip=True, target_dim=2):
        """
        Args:
            hist_dim (int): Input-Dimension der Historie (48h).
            future_dim (int): Input-Dimension der Zukunft (24h Wetterprognose).
            hidden_dim (int): Größe des versteckten Zustands.
            num_layers (int): Anzahl gestapelter Recurrent Layer.
            cell_type (str): 'LSTM' oder 'GRU'.
            dropout (float): Dropout-Rate (0.0 bis 0.5).
            use_skip (bool): Ob eine Skip-Connection zwischen Zukunfts-Wetter & Output genutzt wird.
            target_dim (int): Anzahl der Zielgrößen (2: Solar & Wind).
        """
        super().__init__()
        self.cell_type = cell_type.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_skip = use_skip
        
        rnn_cls = nn.LSTM if self.cell_type == 'LSTM' else nn.GRU
        
        # 1. ENCODER (Historie 48h)
        self.encoder = rnn_cls(
            input_size=hist_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 2. DECODER (Zukunft 24h)
        self.decoder = rnn_cls(
            input_size=future_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Input-Dimension für den Output-Head berechnen
        # Wenn Skip-Connection aktiv: Combined State aus Decoder Output + Roh-Future-Features
        head_input_dim = (hidden_dim + future_dim) if use_skip else hidden_dim
        
        # 3. PROJEKTIONS-HEAD (mit explizitem Dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, target_dim),
            nn.Sigmoid()  # Hält Targets sauber in [0, 1]
        )

    def forward(self, x_hist, x_future):
        # 1. Encoding
        if self.cell_type == 'LSTM':
            _, (h_n, c_n) = self.encoder(x_hist)
            decoder_init = (h_n, c_n)
        else: # GRU
            _, h_n = self.encoder(x_hist)
            decoder_init = h_n
            
        # 2. Decoding mit Future-Weather-Inputs
        dec_out, _ = self.decoder(x_future, decoder_init)
        
        # 3. Optional Skip-Connection (Verschmelzung von Hidden Representation & direct Future Features)
        if self.use_skip:
            combined = torch.cat([dec_out, x_future], dim=-1)
        else:
            combined = dec_out
            
        # 4. Target Vorhersage (24h)
        y_pred = self.fc_out(combined)
        return y_pred


import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Fügt den Zeitreihen-Tokens Sinus/Cosinus-Positionsinformationen hinzu.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x Shape: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    """
    Encoder-Decoder Transformer für Multi-Horizon Forecasting mit Future-Weather Cross-Attention.
    """
    def __init__(self, hist_dim, future_dim, d_model=64, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, 
                 dim_feedforward=128, dropout=0.1, target_dim=2):
        """
        Args:
            hist_dim (int): Anzahl Features der 48h Historie.
            future_dim (int): Anzahl Features der 24h Zukunfts-Wetterprognose.
            d_model (int): Projektions-Dimension für den Transformer.
            nhead (int): Anzahl der Attention Heads (d_model muss durch nhead teilbar sein!).
            num_encoder_layers (int): Anzahl Encoder-Layer.
            num_decoder_layers (int): Anzahl Decoder-Layer.
            dim_feedforward (int): Dimension des inneren Linear-Layers im Transformer block.
            dropout (float): Dropout-Rate.
            target_dim (int): Anzahl Targets (2: Solar & Wind).
        """
        super().__init__()
        self.d_model = d_model
        
        # 1. PROJEKTION AUF D_MODEL & POSITIONAL ENCODING
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.future_proj = nn.Linear(future_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. TRANSFORMER ENCODER & DECODER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 3. PROJEKTIONS-HEAD
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, target_dim),
            nn.Sigmoid()  # Hält Targets sauber in [0, 1]
        )

    def forward(self, x_hist, x_future):
        """
        Args:
            x_hist: Shape (Batch, 48, hist_dim)
            x_future: Shape (Batch, 24, future_dim)
        Returns:
            y_pred: Shape (Batch, 24, target_dim)
        """
        # 1. Embedding & Positional Encoding
        memory_in = self.pos_encoder(self.hist_proj(x_hist))    # Shape: (Batch, 48, d_model)
        target_in = self.pos_encoder(self.future_proj(x_future)) # Shape: (Batch, 24, d_model)
        
        # 2. Encoder: Self-Attention über die 48h Historie
        memory = self.encoder(memory_in)  # Shape: (Batch, 48, d_model)
        
        # 3. Decoder: Self-Attention (Zukunft) + Cross-Attention über Encoder-Memory
        dec_out = self.decoder(tgt=target_in, memory=memory) # Shape: (Batch, 24, d_model)
        
        # 4. Final Head
        y_pred = self.fc_out(dec_out) # Shape: (Batch, 24, target_dim)
        return y_pred