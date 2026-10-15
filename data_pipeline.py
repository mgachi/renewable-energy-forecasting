import os
import time
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1. KONFIGURATION & KOORDINATEN-RASTER (9 Punkte über Deutschland)
# =====================================================================
GRID_POINTS = {
    'NW': (53.5, 7.5),   # Nordwest (Küste/Wind)
    'N':  (54.0, 10.0),  # Nord (Schleswig-Holstein)
    'NE': (53.5, 13.5),  # Nordost (Ostsee)
    'W':  (51.5, 7.0),   # West (NRW)
    'C':  (51.0, 10.5),  # Mitte (Hessen/Thüringen)
    'E':  (51.0, 14.0),  # Ost (Sachsen)
    'SW': (48.5, 8.0),   # Südwest (Baden-Württemberg)
    'S':  (48.5, 11.5),  # Süd (Bayern Solar)
    'SE': (48.5, 13.5)   # Südost (Niederbayern)
}

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
CACHE_DIR = "./data_cache"

SMARD_CSV_FILENAME = "Realisierte_Erzeugung_201812010000_202601010000_Stunde.csv"

os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================================
# HELPER: DATA COVERAGE CHECK
# =====================================================================
def _print_coverage_report(source_name, df, start_date, end_date):
    """
    Analysiert und druckt die zeitliche Abdeckung eines geladenen DataFrames
    inklusive genauer Erkennung zusammenhängender Lücken.
    """
    expected_range = pd.date_range(
        start=f"{start_date} 00:00:00", 
        end=f"{end_date} 23:00:00", 
        freq='h', 
        tz='UTC'
    )
    expected_count = len(expected_range)
    
    if df is None or df.empty:
        print(f"\n[COVERAGE REPORT - {source_name}]")
        print(f"  ❌ FEHLER: Keine Daten empfangen/vorhanden!")
        print('-'*50)
        return
    
    actual_index = df.index
    if actual_index.tz is None:
        actual_index = actual_index.tz_localize('UTC')
        
    # Nur Datenpunkte berücksichtigen, die EXAKT im erwarteten Zeitraum liegen
    relevant_index = actual_index.intersection(expected_range)
    actual_count = len(relevant_index)
    coverage_pct = (actual_count / expected_count) * 100
    
    missing_stamps = expected_range.difference(relevant_index)
    missing_count = len(missing_stamps)
    
    print(f"\n[COVERAGE REPORT - {source_name}]")
    print(f"  Angefragter Zeitraum: {start_date} bis {end_date} ({expected_count:,} Stunden)")
    print(f"  Erhaltene Datenpunkte: {actual_count:,} / {expected_count:,} Stunden ({coverage_pct:.2f}% Abdeckung)")
    
    if missing_count == 0:
        print(f"  ✓ Perfekt: Der Datensatz ist 100% lückenlos!")
    else:
        print(f"  ⚠️  Fehlende Stunden: {missing_count:,} Datenpunkte fehlen.")
        
        # Zusammenhängende Lücken (Gaps) identifizieren
        # Eine neue Lücke beginnt, wenn der Abstand zur vorherigen Stunde > 1h ist
        missing_series = pd.Series(missing_stamps)
        gap_starts = missing_series[missing_series.diff() != pd.Timedelta(hours=1)]
        
        num_gaps = len(gap_starts)
        print(f"      Verteilung: Verteilt auf {num_gaps} einzelne Lücke(n)/Unterbrechung(en).")
        
        # Die ersten 3 Lücken mit jeweiliger Dauer ausgeben
        print("      Beispiel-Lücken:")
        for start_ts in gap_starts.iloc[:3]:
            # Ende dieses zusammenhängenden Blocks ermitteln
            block = missing_series[(missing_series >= start_ts)]
            # Wo bricht die 1-Stunden-Kontinuität ab?
            cont_mask = (block.diff().fillna(pd.Timedelta(hours=1)) == pd.Timedelta(hours=1))
            # Das Ende ist das letzte Element der kontinuierlichen Kette
            if not cont_mask.all():
                first_break_idx = cont_mask.idxmin()
                block_end = missing_series.loc[first_break_idx - 1]
            else:
                block_end = block.iloc[-1]
                
            block_len = len(missing_series[(missing_series >= start_ts) & (missing_series <= block_end)])
            
            if start_ts == block_end:
                print(f"        - Einzelausfall am {start_ts.strftime('%Y-%m-%d %H:%M UTC')}")
            else:
                print(f"        - Lücke von {start_ts.strftime('%Y-%m-%d %H:%M')} bis {block_end.strftime('%Y-%m-%d %H:%M UTC')} ({block_len} Std.)")
                
        if num_gaps > 3:
            print(f"        ... und {num_gaps - 3} weitere Lücke(n).")
            
    print('-'*50)

# =====================================================================
# 2. DATENBESCHAFFUNG (OPEN-METEO & SMARD)
# =====================================================================

def fetch_weather_grid(start_date=START_DATE, end_date=END_DATE):
    """
    Holt historische Wetterdaten für alle 9 Rasterpunkte von Open-Meteo
    inkl. automatischem Coverage-Check.
    """
    cache_path = os.path.join(CACHE_DIR, "weather_grid.csv")
    if os.path.exists(cache_path):
        print("--> Lade Wetterdaten aus Cache...")
        weather_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        _print_coverage_report("Open-Meteo Weather API (Cache)", weather_df, start_date, end_date)
        return weather_df

    print("--> Lade Wetterdaten von Open-Meteo API (9 Punkte)...")
    dfs = []
    hourly_params = "temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,cloud_cover"
    
    for name, (lat, lon) in GRID_POINTS.items():
        print(f"   Abrufe Punkt {name} ({lat}°, {lon}°)...")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_params,
            "timezone": "UTC"
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            res = requests.get(url, params=params)
            if res.status_code == 200:
                break
            elif res.status_code == 429:
                wait_time = (attempt + 1) * 3
                print(f"      Rate-Limit erreicht. Warte {wait_time} Sekunden...")
                time.sleep(wait_time)
            else:
                print(f"API Fehler bei Punkt {name}: {res.text}")
                res.raise_for_status()
                
        data = res.json()["hourly"]
        
        df_point = pd.DataFrame(data)
        df_point['time'] = pd.to_datetime(df_point['time'], utc=True)
        df_point = df_point.set_index('time')
        
        if 'shortwave_radiation' in df_point.columns:
            df_point = df_point.rename(columns={'shortwave_radiation': 'surface_solar_radiation'})
            
        df_point = df_point.add_suffix(f"_{name}")
        dfs.append(df_point)
        time.sleep(1.2)

    weather_df = pd.concat(dfs, axis=1)
    weather_df.to_csv(cache_path)
    
    _print_coverage_report("Open-Meteo Weather API (Live Download)", weather_df, start_date, end_date)
    return weather_df


def fetch_smard_energy_data(start_date=START_DATE, end_date=END_DATE):
    """
    Liest die offiziellen, realen SMARD-Erzeugungsdaten der Bundesnetzagentur 
    aus der lokalen CSV-Datei ein.
    """
    cache_path = os.path.join(CACHE_DIR, "energy_data_cleaned.csv")
    if os.path.exists(cache_path):
        print("--> Lade bereinigte REALE SMARD-Daten aus Cache...")
        energy_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        _print_coverage_report("REALE SMARD-Daten (Cache)", energy_df, start_date, end_date)
        return energy_df

    print(f"--> Verarbeite originale SMARD-CSV der Bundesnetzagentur ({SMARD_CSV_FILENAME})...")
    
    possible_paths = [
        SMARD_CSV_FILENAME,
        os.path.join(CACHE_DIR, SMARD_CSV_FILENAME),
        os.path.join("..", SMARD_CSV_FILENAME)
    ]
    
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            break
            
    if csv_path is None:
        raise FileNotFoundError(
            f"❌ Die Datei '{SMARD_CSV_FILENAME}' wurde nicht gefunden! "
            f"Bitte lege sie in das Projektverzeichnis oder den Ordner '{CACHE_DIR}'."
        )

    raw_df = pd.read_csv(
        csv_path, 
        sep=';', 
        decimal=',', 
        thousands='.', 
        dtype=str
    )
    
    raw_df['timestamp'] = pd.to_datetime(raw_df['Datum von'], format='%d.%m.%Y %H:%M', utc=True)
    raw_df = raw_df.set_index('timestamp')
    
    col_solar = [c for c in raw_df.columns if 'Photovoltaik' in c][0]
    col_wind_on = [c for c in raw_df.columns if 'Wind Onshore' in c][0]
    
    def clean_num(series):
        s = series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s, errors='coerce').fillna(0.0)

    solar_gen = clean_num(raw_df[col_solar])
    wind_gen = clean_num(raw_df[col_wind_on]) # Rein Onshore Wind Power für konsistente Festland-Features
    
    energy_df = pd.DataFrame({
        'generation_solar_mw': solar_gen,
        'generation_wind_mw': wind_gen
    }, index=raw_df.index)
    
    energy_df = energy_df.loc[f"{start_date} 00:00:00":f"{end_date} 23:00:00"]
    
    # Dynamischer Zubau der installierten Kapazitäten DE (2019-2024)
    years_frac = energy_df.index.year.values + (energy_df.index.dayofyear.values / 365.25)
    energy_df['capacity_solar_mw'] = 48000 + (years_frac - 2019) * (37000 / 6)
    energy_df['capacity_wind_mw']  = 59000 + (years_frac - 2019) * (11000 / 6)
    
    # Duplikate filtern
    energy_df = energy_df[~energy_df.index.duplicated(keep='first')].sort_index()

    energy_df.to_csv(cache_path)
    _print_coverage_report("REALE SMARD-Daten Bundesnetzagentur", energy_df, start_date, end_date)
    
    return energy_df


def _create_mock_energy_data():
    """Fallback für Lokale Tests ohne Live-Download"""
    dates = pd.date_range(start=START_DATE, end=f"{END_DATE} 23:00:00", freq='h', tz='UTC')
    years_frac = dates.year.values + (dates.dayofyear.values / 365.25)
    
    cap_solar = 48000 + (years_frac - 2019) * (37000 / 6)
    cap_wind  = 59000 + (years_frac - 2019) * (11000 / 6)
    
    solar_rad = np.maximum(0, np.sin((dates.hour - 6) * np.pi / 12)) * np.sin(dates.dayofyear * 2 * np.pi / 365)
    solar_gen = np.clip(solar_rad ** 1.2 + np.random.normal(0, 0.03, len(dates)), 0, 1) * cap_solar
    
    wind_speed = 6 + 3 * np.cos((dates.dayofyear - 30) * 2 * np.pi / 365) + np.random.normal(0, 2, len(dates))
    wind_gen = np.clip(((wind_speed - 3) / 10.0) ** 2, 0, 1) * cap_wind
    
    return pd.DataFrame({
        'generation_solar_mw': solar_gen,
        'generation_wind_mw': wind_gen,
        'capacity_solar_mw': cap_solar,
        'capacity_wind_mw': cap_wind
    }, index=dates)


# =====================================================================
# 3. FEATURE ENGINEERING, NORMIERUNG & CHRONOLOGISCHER SPLIT
# =====================================================================

def prepare_pipeline():
    # 1. Daten laden
    weather_df = fetch_weather_grid()
    energy_df = fetch_smard_energy_data()
    
    # Indizes eindeutig machen
    weather_df = weather_df[~weather_df.index.duplicated(keep='first')].sort_index()
    energy_df = energy_df[~energy_df.index.duplicated(keep='first')].sort_index()

    # 2. Zusammenführen (Inner Join auf Timestamp)
    df = pd.concat([energy_df, weather_df], axis=1, join='inner').dropna()
    
    # 3. NORMIERUNG DER PRODUKTION (Target-Berechnung [0, 1])
    df['target_solar_norm'] = (df['generation_solar_mw'] / df['capacity_solar_mw']).clip(0.0, 1.0)
    df['target_wind_norm']  = (df['generation_wind_mw'] / df['capacity_wind_mw']).clip(0.0, 1.0)
    
    # 4. ZYKLISCHE KALENDER-FEATURES (Sinus/Cosinus Transformation)
    dates = df.index
    df['hour_sin']        = np.sin(2 * np.pi * dates.hour / 24.0)
    df['hour_cos']        = np.cos(2 * np.pi * dates.hour / 24.0)
    df['month_sin']       = np.sin(2 * np.pi * dates.month / 12.0)
    df['month_cos']       = np.cos(2 * np.pi * dates.month / 12.0)
    df['day_of_week_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7.0)
    
    # 5. CHRONOLOGISCHER SPLIT
    train_mask = df.index.year <= 2022
    val_mask   = df.index.year == 2023
    test_mask  = df.index.year == 2024

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    # Dynamic Fallback falls Daten unvollständig
    if len(val_df) == 0 or len(test_df) == 0:
        print("\nHinweis: Eingeschränkter Zeitbereich in den Rohdaten erkannt. Passen dynamischen Split an...")
        total_len = len(df)
        train_end = int(total_len * 0.70)
        val_end   = int(total_len * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df   = df.iloc[train_end:val_end].copy()
        test_df  = df.iloc[val_end:].copy()

    # 6. FEATURE SKALIERUNG (StandardScaler nur auf Train anpassen!)
    feature_cols = [c for c in df.columns if c not in [
        'generation_solar_mw', 'generation_wind_mw', 
        'capacity_solar_mw', 'capacity_wind_mw',
        'target_solar_norm', 'target_wind_norm'
    ]]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols]) # Fit AUSSCHEISSLICH auf Train!

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    
    # Helper für sauberes Datumsformat
    fmt_range = lambda d: f"{d.index.min().strftime('%Y-%m-%d')} bis {d.index.max().strftime('%Y-%m-%d')}"
    
    print("\n=== PIPELINE STATUS ===")
    print(f"Features insgesamt ({len(feature_cols)}): {feature_cols[:5]} ...")
    print(f"Train-Set : {fmt_range(train_df)} ({len(train_df):,} Stunden)")
    print(f"Val-Set   : {fmt_range(val_df)} ({len(val_df):,} Stunden)")
    print(f"Test-Set  : {fmt_range(test_df)} ({len(test_df):,} Stunden)")
    print(f"Solar Target Span: [{train_df['target_solar_norm'].min():.2f}, {train_df['target_solar_norm'].max():.2f}]")
    print(f"Wind Target Span : [{train_df['target_wind_norm'].min():.2f}, {train_df['target_wind_norm'].max():.2f}]")
    
    return train_df, val_df, test_df, feature_cols, scaler

def get_grid_coordinates():
    """Gibt das 9-Punkte-Raster mit Breitengraden und Längengraden zurück."""
    return GRID_POINTS


if __name__ == "__main__":
    train_df, val_df, test_df, feature_cols, scaler = prepare_pipeline()