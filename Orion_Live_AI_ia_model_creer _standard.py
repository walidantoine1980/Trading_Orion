# =====================================================
# PORTFOLIO OPTIMIZER AI - v13.09 (Sans Calibration)
# =====================================================
#
# OBJECTIF DE CE SCRIPT (APPLICATION COMPLÈTE) :
#
# v13.09 (Demande utilisateur) :
# 1. Suppression complète de `sklearn.calibration.CalibratedClassifierCV`
#    suite à une erreur `ValueError` persistante.
# 2. Le `xgb.XGBClassifier` est maintenant utilisé directement.
#    Son objectif `binary:logistic` produit déjà les probabilités
#    nécessaires via `.predict_proba()`.
# 3. Simplification de l'extraction de "feature_importance".
#
# v13.08 (Correctif) :
# ... (notes précédentes)
# =====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
# --- MODIFIÉ v13.09 : Suppression de l'import de calibration ---
# from sklearn.calibration import CalibratedClassifierCV 
import warnings
from time import time
import joblib
import sys
import threading
import os
import csv
import traceback
import requests # <-- AJOUTÉ POUR L'AGENT DE RECHERCHE
import re     # <-- AJOUTÉ v12.31 POUR REGEX FOREX

# --- NOUVEAU v13.00 : Import pour la config Alpaca ---
import configparser

# --- NOUVEAU v12.14 : Import pour Alpaca ---
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("ERREUR: La bibliothèque 'alpaca-trade-api' n'est pas installée.")
    print("Veuillez l'installer avec : pip install alpaca-trade-api")
    # On ne quitte pas, l'utilisateur peut vouloir seulement entraîner
    
# --- Imports pour l'IHM (v11 + Dashboard) ---
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# --- Ignorer les avertissements ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# =====================================================
# === 🌍 DICTIONNAIRE DE CONVERSION (du Dashboard) ===
# =====================================================
TICKER_CONVERSION_MAP = {
    # Noms communs vers symboles Yahoo Finance (Actions FR)
    "LVMH": "MC.PA", "TOTAL": "TTE.PA", "AIRBUS": "AIR.PA", 
    "KERING": "KER.PA", "ENGIE": "ENGI.PA", "PEUGEOT": "STLA", # MODIFIÉ v12.27
    "CREDIT AGRICOLE": "ACA.PA", "BNP PARIBAS": "BNP.PA", "SOCIETE GENERALE": "GLE.PA",
    "TOTALENERGIES": "TTE.PA", "SANOFI": "SAN.PA", "L'OREAL": "OR.PA",
    "AIR LIQUIDE": "AI.PA", "AXA": "CS.PA", "ESSILOR": "EL.PA",
    "HERMES": "RMS.PA", "SCHNEIDER": "SU.PA", "SCHNEIDER ELECTRIC": "SU.PA",
    "DANONE": "BN.PA", "MICHELIN": "ML.PA", "VINCI": "DG.PA",
    "TELEPERFORMANCE": "TEP.PA", "DASSAULT SYSTEMES": "DSY.PA",
    "STELLANTIS": "STLA", # MODIFIÉ v12.27
    
    # Noms communs vers symboles Yahoo Finance (Actions US)
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META",
    "TEAM": "TEAM", # Ajouté v12.25
    
    # Indices et ETF courants
    "CAC40": "^FCHI", "DAX": "^GDAXI", "S&P 500": "^GSPC", "SP500": "^GSPC",
    "NASDAQ": "^IXIC", "NASDAQ 100": "QQQ", "ETF CAC40": "C40.PA",
    "ETF S&P 500": "SPY", "AMUNDI ETF S&P 500": "500.PA",
    
    # Symboles communs (sans suffixe) vers symbole complet
    "ENGI": "ENGI.PA", "OR": "OR.PA", "AI": "AI.PA", "SAN": "SAN.PA",
    "TTE": "TTE.PA", "MC": "MC.PA", "KER": "KER.PA", "BNP": "BNP.PA",
    "GLE": "GLE.PA", "ACA": "ACA.PA", "CS": "CS.PA", "EL": "EL.PA",
    "RMS": "RMS.PA", "SU": "SU.PA", "BN": "BN.PA", "ML": "ML.PA", "DG": "DG.PA",
    "AAPL": "AAPL", "BTC-USD": "BTC-USD", "GC=F": "GC=F",
    
    # Typos courants (Ajoutés v12.25)
    "ENGIE.PA": "ENGI.PA", # Correction de l'erreur de ticker
    "CAGR.PA": "ACA.PA",
    "CAGR": "ACA.PA",
    "OREP.PA": "OR.PA",
    "OREP": "OR.PA",
    "PEUP.PA": "STLA", # MODIFIÉ v12.27
    "PEUP": "STLA", # MODIFIÉ v12.27
    "LDOF.MI": "LDO.MI", # Leonardo
    "LDOF": "LDO.MI",
    
    # NOUVEAU v13.05 (Correction Erreurs 2) : Ajout de tickers courants
    "SAF": "SAF.PA",          # Safran
    "BNPP": "BNP.PA",         # BNP Paribas
    "RENA": "RNO.PA",         # Renault
    "RNO": "RNO.PA",          # Renault (doublon pour être sûr)
    "BAES": "BA.L",           # BAE Systems
    "SOGN": "GLE.PA",         # Société Générale
    "KARSN": "KARSN.IS",      # Karsan Otomotiv
    "7201": "7201.T",         # Nissan Motor
    "FDJU": "FDJ.PA",         # Française des Jeux
    
    # NOUVEAU v13.05 (Correction Erreurs 3) :
    "FDJ": "FDJ.PA",          # Ajout du ticker nettoyé
    "FDJ.PA": "FDJ.PA",       # NOUVEAU v13.05 (Correctif 4) : Ajout explicite
    "MTXGN": "MTX.DE",        # MTU Aero Engines
    "TTEF": "TTE.PA",         # Typo Total
    "TTEF.PA": "TTE.PA",      # Typo Total
    "RHMG": "CFR.SW",         # Richemont
    "AIRP": "AIR.PA",         # Typo Airbus
    "AIRP.PA": "AIR.PA",      # Typo Airbus
    "SAABB.ST": "SAAB-B.ST",  # Typo Saab
    "SAABB": "SAAB-B.ST",     # Typo Saab
    
    # Forex (Map de base, mais la logique regex gère les autres)
    "EUR/USD": "EURUSD=X", "EURUSD": "EURUSD=X", "EURUSD=X": "EURUSD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "GBP/EUR": "GBPEUR=X", # AJOUTÉ v12.28
    
    # Commodities
    "GC": "GC=F", "GOLD": "GC=F", 
    "SI": "SI=F", "SILVER": "SI=F", "ARGENT": "SI=F",
    "CL": "CL=F", "OIL": "CL=F", "PETROLE": "CL=F",
    
    # Crypto
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
}

# Dictionnaire global pour la watchlist chargée
# Clé: "Nom Affiché" (ex: "LVMH"), Valeur: "Ticker Yahoo" (ex: "MC.PA")
watchlist_map = {}

# --- NOUVELLE FONCTION (v12.4) ---
# Un cache pour les recherches, pour éviter de surcharger l'API
YAHOO_SEARCH_CACHE = {}

def search_yahoo_finance(query):
    """
    Interroge l'API de recherche (non-officielle) de Yahoo Finance pour trouver un ticker.
    """
    if query in YAHOO_SEARCH_CACHE:
        return YAHOO_SEARCH_CACHE[query]

    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"  [Agent] Recherche Yahoo Finance pour : '{query}'...")
    
    try:
        response = requests.get(url, headers=headers, timeout=5) 
        response.raise_for_status() 
        data = response.json()
        
        if 'quotes' in data and len(data['quotes']) > 0:
            first_result = data['quotes'][0]
            if 'symbol' in first_result:
                ticker = first_result['symbol']
                name = first_result.get('shortname', first_result.get('longname', ''))
                print(f"  [Agent] Trouvé : {ticker} ({name})")
                YAHOO_SEARCH_CACHE[query] = ticker
                return ticker
                
    except requests.exceptions.RequestException as e:
        print(f"  [Agent] Erreur réseau lors de la recherche pour '{query}': {e}")
    except Exception as e:
        print(f"  [Agent] Erreur lors de l'analyse de la recherche pour '{query}': {e}")

    print(f"  [Agent] Aucune correspondance trouvée pour '{query}'.")
    YAHOO_SEARCH_CACHE[query] = None 
    return None

# =====================================================
# === 🛠️ FONCTIONS UTILITAIRES (du Dashboard) ===
# =====================================================

# --- MODIFIÉ v13.05 (Correction Erreur) ---
def convert_csv_symbol_to_yahoo(symbol):
    """
    Convertit un symbole (nom commun, ticker partiel, ou ticker complet) 
    en format Yahoo Finance.
    v13.05 : Ajout de .K et .N à la liste des suffixes.
    v12.31 : Ajout d'une règle REGEX pour convertir automatiquement
             les paires Forex (ex: "EUR/AED..." -> "EURAED=X").
    """
    symbol_str = str(symbol).strip().upper()

    # 1. Si le symbole exact est une CLÉ dans la map, le retourner immédiatement.
    #    (Ex: "LVMH" -> "MC.PA", "EURUSD" -> "EURUSD=X", "STELLANTIS" -> "STLA")
    if symbol_str in TICKER_CONVERSION_MAP:
        return TICKER_CONVERSION_MAP[symbol_str]

    # 2. NOUVEAU v12.31 : Règle de détection Forex
    #    Cherche un motif "XXX/YYY" au début de la chaîne.
    forex_match = re.match(r'^([A-Z]{3})/([A-Z]{3})', symbol_str)
    if forex_match:
        base_currency = forex_match.group(1)
        quote_currency = forex_match.group(2)
        forex_ticker = f"{base_currency}{quote_currency}=X"
        print(f"  [Agent] Détection Forex : '{symbol_str}' -> '{forex_ticker}'")
        # Ajoute à la map pour accélérer les prochaines fois
        TICKER_CONVERSION_MAP[symbol_str] = forex_ticker
        return forex_ticker

    # 3. Nettoyer les suffixes courants (.PA, .DE, .O, etc.)
    cleaned_symbol = symbol_str
    # --- MODIFICATION v13.05 (Correction Erreur) ---
    # Ajout de .K et .N pour gérer les tickers comme ANET.K ou les tickers NYSE
    suffixes = [".PA", ".DE", ".L", ".MI", ".AS", ".SW", ".T", ".HK", ".AX", ".IS", ".O", ".K", ".N"]
    # --- FIN MODIFICATION v13.05 ---
    
    for suffix in suffixes:
        if symbol_str.endswith(suffix):
            cleaned_symbol = symbol_str[:-len(suffix)]
            break # Important: ne retirer qu'un seul suffixe
            
    # 4. Vérifier si le symbole NETTOYÉ est une CLÉ dans la map.
    #    (Ex: L'utilisateur tape "MC.PA" ou "TTEF.PA". cleaned_symbol devient "MC" ou "TTEF".
    #     "MC" est dans la map -> "MC.PA". "TTEF" est dans la map -> "TTE.PA")
    if cleaned_symbol in TICKER_CONVERSION_MAP:
        return TICKER_CONVERSION_MAP[cleaned_symbol]

    # 5. Vérifier si c'est une requête "simple" à rechercher (un seul mot)
    #    (Ex: "RENAULT", "KER", "AAPL")
    #    Empêche la recherche de "EUR/USD - EURO US DOLLAR" (déjà géré par la règle 2)
    is_searchable = ' ' not in symbol_str and \
                      '/' not in symbol_str and \
                      '.' not in symbol_str and \
                      '=' not in symbol_str and \
                      '^' not in symbol_str
                      
    if is_searchable:
        # 'cleaned_symbol' peut être le même que 'symbol_str' ici (ex: "AAPL")
        # On recherche d'abord le mot nettoyé, puis le mot original si besoin.
        
        # Recherche sur le mot nettoyé (ex: "RENA" de "RENA.PA")
        if cleaned_symbol != symbol_str: 
            found_ticker = search_yahoo_finance(cleaned_symbol)
            if found_ticker:
                TICKER_CONVERSION_MAP[symbol_str] = found_ticker
                TICKER_CONVERSION_MAP[cleaned_symbol] = found_ticker
                return found_ticker

        # Recherche sur le mot original (ex: "RENAULT" ou "AAPL")
        found_ticker = search_yahoo_finance(symbol_str)
        if found_ticker:
            TICKER_CONVERSION_MAP[symbol_str] = found_ticker
            return found_ticker

    # 6. Si aucune règle n'a fonctionné, retourner le symbole original.
    #    - S'il est valide (ex: "RNO.PA" tapé manuellement), yfinance le trouvera.
    #    - S'il est invalide (ex: "EUR/AED - ..."), yfinance échouera
    #      (car la règle 2 n'a pas matché), et l'erreur apparaîtra dans le log.
    
    # --- MODIFICATION v13.05 (Correction Erreur) ---
    # Si nous avons nettoyé un symbole (ex: ANET.K -> ANET),
    # et que ANET n'était pas dans la map, nous retournons 
    # le symbole NETTOYÉ (ANET), et non l'original (ANET.K).
    # C'est ce symbole nettoyé que yfinance doit utiliser.
    if cleaned_symbol != symbol_str:
        print(f"  [Agent] Symbole '{symbol_str}' nettoyé en '{cleaned_symbol}'.")
        return cleaned_symbol
    # --- FIN MODIFICATION v13.05 ---
        
    return symbol_str


def detect_csv_delimiter(filepath):
    """Détecte automatiquement le délimiteur du CSV"""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            first_line = f.readline().strip()
        delimiters = [';', ',', '\t', '|']
        counts = {delim: first_line.count(delim) for delim in delimiters}
        detected_delimiter = max(counts, key=counts.get)
        if counts[detected_delimiter] == 0:
            return ';'
        return detected_delimiter
    except:
        return ';'

def load_watchlist_to_gui(listbox_widget):
    """Charge un fichier CSV ou Excel et remplit la Listbox (fonction adaptée pour l'IHM)"""
    global watchlist_map
    
    filepath = filedialog.askopenfilename(
        title="Ouvrir un fichier Watchlist (CSV ou Excel)",
        filetypes=(("Fichiers Watchlist", "*.csv *.xlsx"),
                   ("Fichiers CSV", "*.csv"),
                   ("Fichiers Excel", "*.xlsx"),
                   ("Tous les fichiers", "*.*"))
    )
    if not filepath:
        return

    try:
        watchlist_map = {}
        listbox_widget.delete(0, tk.END) 
        loaded_count = 0

        if filepath.endswith('.csv'):
            print(f"Chargement du fichier CSV : {filepath}")
            delimiter = detect_csv_delimiter(filepath)
            print(f"Délimiteur détecté : '{delimiter}'")
            with open(filepath, mode='r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter=delimiter)
                try:
                    header_raw = next(reader)
                    header = [h.strip().lower().replace('﻿', '') for h in header_raw]
                    print(f"En-têtes trouvés : {header}")
                except StopIteration:
                    messagebox.showerror("Erreur CSV", "Le fichier CSV est vide.")
                    return

                name_variants = ['name', 'nom', 'company', 'company name', 'security']
                symbol_variants = ['symbol', 'ticker', 'code', 'isin', 'id']
                
                name_col, symbol_col = None, None
                for i, col_name in enumerate(header):
                    if col_name in name_variants: name_col = i
                    elif col_name in symbol_variants: symbol_col = i
                
                if name_col is None: name_col = 0
                if symbol_col is None: symbol_col = 1 if len(header) > 1 else 0
                print(f"Colonne 'Nom' utilisée : index {name_col}")
                print(f"Colonne 'Symbole' utilisée : index {symbol_col}")

                for i, row in enumerate(reader):
                    if len(row) > max(name_col, symbol_col):
                        display_name = row[name_col].strip()
                        csv_symbol = row[symbol_col].strip()
                        
                        if not display_name or not csv_symbol: 
                            print(f"Ligne {i+2} ignorée : données manquantes.")
                            continue
                        
                        yahoo_ticker = convert_csv_symbol_to_yahoo(csv_symbol)
                        
                        if display_name not in watchlist_map:
                            watchlist_map[display_name] = yahoo_ticker
                            listbox_widget.insert(tk.END, display_name) 
                            loaded_count += 1
                        else:
                            print(f"Ligne {i+2} ignorée : '{display_name}' est déjà dans la liste.")

        elif filepath.endswith('.xlsx'):
            print(f"Chargement du fichier Excel : {filepath}")
            df = pd.read_excel(filepath, engine='openpyxl')
            df.columns = [str(col).strip().lower().replace('﻿', '') for col in df.columns]
            print(f"En-têtes trouvés : {list(df.columns)}")

            name_variants = ['name', 'nom', 'company', 'company name', 'security']
            symbol_variants = ['symbol', 'ticker', 'code', 'isin', 'id']
            
            name_col, symbol_col = None, None
            for col in df.columns:
                if col in name_variants: name_col = col
                elif col in symbol_variants: symbol_col = col
            
            if name_col is None: name_col = df.columns[0]
            if symbol_col is None: symbol_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            print(f"Colonne 'Nom' utilisée : '{name_col}'")
            print(f"Colonne 'Symbole' utilisée : '{symbol_col}'")

            for i, row in df.iterrows():
                display_name = str(row[name_col]).strip()
                csv_symbol = str(row[symbol_col]).strip()

                if not display_name or not csv_symbol or display_name == 'nan' or csv_symbol == 'nan': 
                    print(f"Ligne {i+2} ignorée : données manquantes.")
                    continue

                yahoo_ticker = convert_csv_symbol_to_yahoo(csv_symbol)
                
                if display_name not in watchlist_map:
                    watchlist_map[display_name] = yahoo_ticker
                    listbox_widget.insert(tk.END, display_name)
                    loaded_count += 1
                else:
                    print(f"Ligne {i+2} ignorée : '{display_name}' est déjà dans la liste.")
        else:
            messagebox.showerror("Erreur Fichier", "Type non supporté. Utilisez .csv ou .xlsx")
            return

        if loaded_count == 0:
            messagebox.showwarning("Avertissement", "Aucune donnée valide trouvée dans le fichier.")
        else:
            print(f"Succès : {loaded_count} actions chargées.")
            messagebox.showinfo("Succès", f"{loaded_count} actions chargées depuis le fichier.")

    except Exception as e:
        print(f"[ERREUR] Impossible de lire le fichier: {str(e)}")
        messagebox.showerror("Erreur Fichier", f"Impossible de lire le fichier: {str(e)}\n\nTrace: {traceback.format_exc()}")


# =====================================================
# === CLASSE DE LOGIQUE MÉTIER (le "Cerveau" v11) ===
# =====================================================
class TradingAI:
    def __init__(self):
        print("Initialisation du système AI Optimizer...")
        self.REGIME_TICKER = "^VIX"
        self.start_date_train = "2009-01-01"
        self.end_date_train = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        self.FEATURE_NAMES = ["RSI", "MACD", "Signal", "Momentum_1D", "SMA_20", "BB_Width", "Volume_Change", "Market_Regime"]
        self.FEATURE_NAMES_NO_REGIME = ["RSI", "MACD", "Signal", "Momentum_1D", "SMA_20", "BB_Width", "Volume_Change"]
        
        self.MODEL_FILE = "ia_model.joblib"
        self.SCALER_FILE = "ia_scaler.joblib"
        
        self.current_feature_names = self.FEATURE_NAMES_NO_REGIME 

    # --- Fonctions Indicateurs ---
    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def compute_macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def compute_sma(self, series, period=20):
        return series.rolling(window=period).mean()

    def compute_bollinger(self, series, window=20, num_std=2):
        sma = self.compute_sma(series, window)
        std = series.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return sma, upper, lower

    # --- Fonction de Création des "Features" ---
    def create_features(self, asset_data, regime_df, feature_list, include_target=True):
        df = pd.DataFrame()
        df["Close"] = asset_data["Close"]
        
        if "Volume" in asset_data.columns and not asset_data["Volume"].isnull().all():
            df["Volume"] = asset_data["Volume"]
        else:
            df["Volume"] = 0
        df["Volume"] = df["Volume"].replace(0, 1e-6)

        if not regime_df.empty:
            df = df.join(regime_df, how='left') 
        
        df["RSI"] = self.compute_rsi(df["Close"])
        df["MACD"], df["Signal"] = self.compute_macd(df["Close"])
        df["SMA_20"] = self.compute_sma(df["Close"])
        bb_sma, bb_upper, bb_lower = self.compute_bollinger(df["Close"])
        df["BB_Width"] = (bb_upper - bb_lower) / bb_sma.replace(0, 1e-9)
        
        # --- MODIFIÉ v12.17 : Stocker les bandes pour les graphiques ---
        df["BB_Upper"] = bb_upper
        df["BB_Lower"] = bb_lower
        # --- Fin Modif v12.17 ---
        
        df["Volume_Change"] = df["Volume"].pct_change(fill_method=None)
        df["Momentum_1D"] = df["Close"].pct_change(fill_method=None)
        
        if 'Market_Regime' in df.columns:
            df['Market_Regime'].ffill(inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if include_target:
            df["Target"] = (df["Close"].pct_change(21).shift(-21) > 0).astype(int)
            df.dropna(inplace=True) 
            X = df[feature_list]
            y = df["Target"]
            return X, y
        else:
            df.dropna(subset=feature_list, inplace=True)
            X = df[feature_list]
            return X, df

    # --- Téléchargement VIX ---
    def get_regime_feature(self, start, end):
        print(f"        -> Téléchargement du Régime de Marché ({self.REGIME_TICKER})...")
        try:
            regime_data_raw = yf.download(self.REGIME_TICKER, start=start, end=end, auto_adjust=True)
            if regime_data_raw.empty:
                raise ValueError(f"Aucune donnée pour {self.REGIME_TICKER}")
            
            regime_data_raw.ffill(inplace=True)
            rolling_mean = regime_data_raw['Close'].rolling(window=20).mean()
            
            if isinstance(rolling_mean, pd.Series):
                regime_feature = rolling_mean.to_frame(name='Market_Regime')
            elif isinstance(rolling_mean, pd.DataFrame):
                regime_feature = rolling_mean
                if regime_feature.shape[1] > 0:
                    regime_feature.columns = ['Market_Regime']
                else:
                    raise ValueError("DataFrame VIX vide après calcul.")
            else:
                raise TypeError("Calcul du régime de marché a retourné un type inattendu.")

            regime_feature.ffill(inplace=True)
            regime_feature.dropna(inplace=True)
            
            print(f"        -> Données de Régime de Marché ({self.REGIME_TICKER}) téléchargées.")
            self.current_feature_names = self.FEATURE_NAMES
            return regime_feature
        
        except Exception as e:
            print(f"        -> [!] Erreur lors du téléchargement du {self.REGIME_TICKER}: {e}")
            print("        -> Continuation sans la feature de régime de marché.")
            self.current_feature_names = self.FEATURE_NAMES_NO_REGIME
            return pd.DataFrame() 


    # --- MODE ENTRAÎNEMENT ---
    def run_training_mode(self, assets_to_process, start_date, end_date):
        try:
            print("\n" + "="*50)
            print("--- DÉMARRAGE DU MODE ENTRAÎNEMENT ---")
            print(f"Actifs Ciblés : {assets_to_process}")
            print(f"Période d'entraînement : {start_date} à {end_date}") 
            print("="*50 + "\n")

            print(f"--- [ÉTAPE 1/4] Téléchargement des données complètes ({start_date} à {end_date}) ---")
            start_time = time()
            
            print("        -> Téléchargement des données YFinance pour les actifs...")
            data_full = yf.download(assets_to_process, start=start_date, end=end_date, auto_adjust=True)
            data_full.ffill(inplace=True)
            print(f"        -> Données OHLC des actifs téléchargées. {len(data_full)} lignes.")

            regime_feature = self.get_regime_feature(start_date, end_date)
            print(f"        -> Téléchargement terminé en {time() - start_time:.2f}s.")
            print("-------------------------------------------------------\n")

            
            print(f"--- [ÉTAPE 2/4] Préparation des données d'APPRENTISSAGE ---")
            start_time = time()
            features_train, targets_train = [], []

            # --- DEBUT FIX v12.23 ---
            # Suppression du bloc 'if len(assets_to_process) == 1:'
            # La logique de la boucle (anciennement 'else') est correcte pour 1 ou N actifs
            # car 'assets_to_process' est toujours une liste.

            print(f"        -> Traitement ({len(assets_to_process)} Actif(s)) [Train]...")
            for asset in assets_to_process:
                print(f"                      -> Calcul des features pour [Train] : {asset}...")
                try:
                    # Tenter d'extraire l'actif en supposant un MultiIndex
                    # .xs() extrait un DataFrame simple (sans MultiIndex) pour l'actif
                    asset_train_data = data_full.xs(asset, level=1, axis=1)
                    
                except (KeyError, pd.errors.PerformanceWarning):
                    # Gère le cas où yf.download(['UN_SEUL_TICKER']) ne renvoie PAS de MultiIndex
                    if len(assets_to_process) == 1:
                        print("                      -> [Info] Comportement yfinance détecté (DataFrame Simple pour 1 actif). Traitement direct.")
                        asset_train_data = data_full # data_full EST le DataFrame simple
                    else:
                         print(f"                   [!] Pas de données d'apprentissage du tout pour {asset}.")
                         continue # Passer à l'actif suivant
                except Exception as e:
                    print(f"                   [!] Erreur lors de l'extraction des données pour {asset}: {e}")
                    continue # Passer à l'actif suivant

                try:
                    if not asset_train_data['Close'].dropna().empty:
                        X, y = self.create_features(asset_train_data, regime_feature, self.current_feature_names, include_target=True)
                        if not X.empty:
                            features_train.append(X)
                            targets_train.append(y)
                    else:
                        print(f"                   [!] Pas de données 'Close' valides pour {asset}.")
                except Exception as e:
                    print(f"                   [!] Erreur lors du calcul des features pour {asset}: {e}")
            # --- FIN FIX v12.23 ---

            if not features_train:
                print("[ERREUR] Aucun échantillon d'apprentissage. Vérifiez les tickers ou les dates.")
                return

            print("        -> Concaténation de tous les échantillons d'apprentissage...")
            X_total_train = pd.concat(features_train)
            y_total_train = pd.concat(targets_train)
            print(f"        -> Préparation des données terminée en {time() - start_time:.2f}s.")
            print(f"        -> Taille totale du set d'apprentissage : {len(X_total_train)} échantillons.")
            print("-------------------------------------------------------\n")

            
            print(f"--- [ÉTAPE 3/4] Entraînement du Scaler et du Modèle ---")
            start_time = time()
            print("        -> Standardisation des features (Fit/Transform)...")
            scaler = StandardScaler()
            X_scaled_train = scaler.fit_transform(X_total_train[self.current_feature_names]) 

            print("        -> Lancement de l'entraînement de XGBoost...") # <-- Mis à jour
            
            # --- MODIFICATION v13.09 (Demande utilisateur) ---
            # Suppression de CalibratedClassifierCV suite à une erreur persistante.
            # Nous utilisons XGBClassifier directement.
            # L'objectif 'binary:logistic' produit déjà des probabilités.
            
            model = xgb.XGBClassifier(
                objective='binary:logistic',  # <-- L'objectif de CLASSIFICATION
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                enable_categorical=True      # <-- Gardé pour la compatibilité
            )
            
            # --- FIN MODIFICATION v13.09 ---
            
            print("        -> Entraînement du modèle (direct)...") # <-- Mis à jour
            
            # Pas de calibration, on entraîne le modèle directement.
            model.fit(X_scaled_train, y_total_train) 
            
            print(f"        -> Modèle XGBoost entraîné avec succès en {time() - start_time:.2f}s.") # <-- Mis à jour
            print("-------------------------------------------------------\n")
            
            
            print(f"--- [ÉTAPE 4/4] Sauvegarde des modèles ---")
            print(f"        -> Sauvegarde du modèle dans : {self.MODEL_FILE}")
            joblib.dump(model, self.MODEL_FILE)
            print(f"        -> Modèle sauvegardé avec succès.")
            print(f"        -> Sauvegarde du scaler dans : {self.SCALER_FILE}")
            joblib.dump(scaler, self.SCALER_FILE)
            print(f"        -> Scaler sauvegardé avec succès.")
            
            print("\n--- [ÉTAPE BONUS] Affichage de l'importance des features ---")
            try:
                print("        -> Extraction des features importances...")
                
                # --- MODIFIÉ v13.09 : Le modèle est maintenant l'estimateur de base ---
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=self.current_feature_names)
                    # --- FIN MODIFICATION v13.09 ---
                    importances = importances.sort_values(ascending=False)
                    print("        -> Importance des features :")
                    print(importances)
                    
                    plt.figure(figsize=(10, 6))
                    importances.plot(kind='bar', title='Importance des Features (selon XGBoost)')
                    plt.ylabel('Importance (Gain)')
                    plt.tight_layout()
                    print("        -> Affichage du graphique d'importance des features...")
                    plt.show() 
                else:
                    print("        -> Impossible d'extraire l'importance des features du modèle calibré.")
            except Exception as e:
                print(f"        -> [!] Erreur lors de l'affichage de l'importance des features : {e}")

            print("\n" + "="*50)
            print("--- MODE ENTRAÎNEMENT TERMINÉ ---")
            print("="*50 + "\n")

        except Exception as e:
            print(f"\n[ERREUR CRITIQUE PENDANT L'ENTRAÎNEMENT] : {e}\nTrace: {traceback.format_exc()}")


    # --- MODE PRÉDICTION (MODIFIÉ v12.14) ---
    def run_prediction_mode(self, assets_to_process, display_prefs, ticker_to_name_map=None):
        try:
            print("\n" + "="*50)
            print("--- DÉMARRAGE DU MODE PRÉDICTION ---")
            print(f"Actifs Ciblés : {assets_to_process}")
            print(f"Indicateurs à afficher : {display_prefs}")
            print("="*50 + "\n")
            
            if ticker_to_name_map is None:
                ticker_to_name_map = {ticker: ticker for ticker in assets_to_process}


            print(f"--- [ÉTAPE 1/5] Chargement du Modèle et du Scaler ---")
            try:
                print(f"        -> Chargement du modèle : {self.MODEL_FILE}")
                model = joblib.load(self.MODEL_FILE)
                print(f"        -> Chargement du scaler : {self.SCALER_FILE}")
                scaler = joblib.load(self.SCALER_FILE)
                print(f"        -> Modèle et Scaler chargés.")
            except FileNotFoundError:
                print(f"        -> [ERREUR] Fichiers modèles non trouvés.")
                print(f"        -> Veuillez d'abord exécuter le MODE ENTRAÎNEMENT.")
                return None, None, None, None, None # MODIFIÉ v12.14
            print("-------------------------------------------------------\n")


            print(f"--- [ÉTAPE 2/5] Téléchargement des données 'Live' (2 dernières années) ---") 
            start_time = time()
            
            start_date_live = (pd.Timestamp.now() - pd.DateOffset(days=730)).strftime('%Y-%m-%d')
            end_date_live = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            print(f"        -> Téléchargement des données YFinance pour les actifs ({start_date_live} à {end_date_live})...")
            data_live = yf.download(assets_to_process, start=start_date_live, end=end_date_live, auto_adjust=True)
            data_live.ffill(inplace=True)

            regime_feature = self.get_regime_feature(start_date_live, end_date_live)
            print(f"        -> Téléchargement des données live terminé en {time() - start_time:.2f}s.")
            print("-------------------------------------------------------\n")

            
            print(f"--- [ÉTAPE 3/5] Calcul des Features 'Live' ---")
            live_features_dict = {}
            live_indicators_dict = {}
            full_indicators_history = {}
            
            # --- DEBUT FIX v12.23 ---
            # Suppression du bloc 'if len(assets_to_process) == 1:'
            # La logique de la boucle (anciennement 'else') est correcte pour 1 ou N actifs
            
            print(f"        -> Traitement ({len(assets_to_process)} Actif(s)) [Live]...")
            for asset in assets_to_process:
                print(f"                      -> Calcul des features pour [Live] : {asset}...")
                try:
                    # Tenter d'extraire l'actif en supposant un MultiIndex
                    # .xs() extrait un DataFrame simple (sans MultiIndex) pour l'actif
                    asset_live_data = data_live.xs(asset, level=1, axis=1)

                except (KeyError, pd.errors.PerformanceWarning):
                    # Gère le cas où yf.download(['UN_SEUL_TICKER']) ne renvoie PAS de MultiIndex
                     if len(assets_to_process) == 1:
                         print("                      -> [Info] Comportement yfinance détecté (DataFrame Simple pour 1 actif). Traitement direct.")
                         asset_live_data = data_live # data_live EST le DataFrame simple
                     else:
                         print(f"                   [!] Pas de données live pour {asset}")
                         continue # Passer à l'actif suivant
                except Exception as e:
                    print(f"                   [!] Erreur lors de l'extraction des données pour {asset}: {e}")
                    continue # Passer à l'actif suivant

                try:
                    if not asset_live_data['Close'].dropna().empty:
                        X_live_full, df_indicators = self.create_features(asset_live_data, regime_feature, self.current_feature_names, include_target=False)
                        if not X_live_full.empty:
                            live_features_dict[asset] = X_live_full.iloc[-1:]
                            live_indicators_dict[asset] = df_indicators.iloc[-1:] 
                            full_indicators_history[asset] = df_indicators
                        else:
                            print(f"                   [!] Pas de features pour {asset}")
                    else:
                        print(f"                   [!] Pas de données 'Close' valides pour {asset}.")
                except Exception as e:
                    print(f"                   [!] Erreur lors du calcul des features pour {asset}: {e}")
            # --- FIN FIX v12.23 ---

            if not live_features_dict:
                print("[ERREUR] Impossible de calculer les features live pour aucun actif.")
                return None, None, None, None, None # MODIFIÉ v12.14
            
            valid_assets = list(live_features_dict.keys())
            if not valid_assets:
                 print("[ERREUR] Aucun actif n'a de données valides à la dernière date.")
                 return None, None, None, None, None # MODIFIÉ v12.14
            
            last_date = live_features_dict[valid_assets[0]].index[0].strftime('%Y-%m-%d')
            print(f"        -> Calcul des features terminé pour la date : {last_date}")
            print("-------------------------------------------------------\n")

            print(f"--- [ÉTAPE 3.5] Indicateurs Techniques (pour la date {last_date}) ---")
            for asset, indicators_series in live_indicators_dict.items():
                print(f"        --- {ticker_to_name_map.get(asset, asset)} ({asset}) ---")
                try:
                    if display_prefs.get('rsi', False): 
                        print(f"                      RSI: {indicators_series['RSI'].iloc[0]:.2f}")
                    if display_prefs.get('macd', False): 
                        print(f"                      MACD: {indicators_series['MACD'].iloc[0]:.2f} | Signal: {indicators_series['Signal'].iloc[0]:.2f}")
                    if display_prefs.get('sma', False): 
                        print(f"                      SMA 20: {indicators_series['SMA_20'].iloc[0]:.2f}")
                    if display_prefs.get('bollinger', False): 
                        print(f"                      BB Width: {indicators_series['BB_Width'].iloc[0]:.4f}")
                except Exception as e:
                    print(f"                      [!] Erreur lors de l'affichage des indicateurs pour {asset}: {e}")
            print("-------------------------------------------------------\n")


            print(f"--- [ÉTAPE 4/5] Prédiction des Probabilités ---")
            live_probabilities = {}
            
            for asset, features_df in live_features_dict.items():
                try:
                    print(f"        -> Prédiction pour {ticker_to_name_map.get(asset, asset)} ({asset})...")
                    features_df_ordered = features_df[self.current_feature_names]
                    print(f"                      -> Scaling des features (Transform)...")
                    X_scaled_live = scaler.transform(features_df_ordered)
                    print(f"                      -> Prédiction de la probabilité...")
                    prob_hausse = model.predict_proba(X_scaled_live)[:, 1]
                    live_probabilities[asset] = prob_hausse[0]
                except Exception as e:
                    print(f"                      [!] Erreur de prédiction pour {asset}: {e}")
                    live_probabilities[asset] = np.nan
                    
            print("        -> Prédiction IA terminée.")
            print("-------------------------------------------------------\n")
            
            print(f"--- [ÉTAPE 5/5] SIGNAUX TEMPS RÉEL (pour la période {last_date}) ---")
            
            results_df = pd.Series(live_probabilities, name="Probabilité_Hausse_IA_21J")
            results_df.index = results_df.index.map(lambda t: f"{ticker_to_name_map.get(t, t)} ({t})") 
            
            results_df = results_df.sort_values(ascending=False)
            print(results_df)
            
            print("\n--- Allocation de Risque Recommandée (Stratégie v11) ---")
            print("        -> Calcul des poids bruts (Max(0, (Prob - 0.5) * 2))...")
            
            # Utiliser live_probabilities (clés ticker) pour la logique interne
            weights = {asset: max(0, (prob - 0.5) * 2) for asset, prob in live_probabilities.items()}
            total_raw_weight = sum(weights.values())
            
            final_allocations = {}
            cash_weight = 1.0

            if total_raw_weight > 1.0:
                print(f"        -> Poids > 100%. Normalisation des poids. CASH = 0%.")
                final_allocations = {asset: w / total_raw_weight for asset, w in weights.items()}
                cash_weight = 0.0
            elif total_raw_weight > 0:
                print(f"        -> Poids < 100% ({total_raw_weight:.1%}). CASH = {1.0 - total_raw_weight:.1%}.")
                final_allocations = weights
                cash_weight = 1.0 - total_raw_weight
            else:
                print("        -> Aucun signal positif. Allocation 100% CASH.")
            
            final_allocations['CASH'] = cash_weight
            
            # Créer la série pour l'affichage (avec noms)
            alloc_series = pd.Series(final_allocations, name="Allocation_Recommandée")
            alloc_series.index = alloc_series.index.map(lambda t: f"{ticker_to_name_map.get(t, t)} ({t})" if t != "CASH" else "CASH")
            alloc_series = alloc_series.sort_values(ascending=False)
            
            print(alloc_series.map('{:.1%}'.format))
            print("\n" + "="*50)
            print("--- MODE PRÉDICTION TERMINÉ ---")
            print("="*50 + "\n")
            
            # MODIFIÉ v12.14 : Retourner 'final_allocations' (map de tickers)
            return results_df, alloc_series, last_date, full_indicators_history, final_allocations

        except Exception as e:
            print(f"\n[ERREUR CRITIQUE PENDANT LA PRÉDICTION] : {e}\nTrace: {traceback.format_exc()}")
            return None, None, None, None, None # MODIFIÉ v12.14

# =====================================================
# === CLASSE DE L'IHM (L'Interface Graphique) ===
# =====================================================
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Orion AI v1.0") # MODIFIÉ v13.04 (Titre v1.0 gardé)
        root.geometry("850x950") # MODIFIÉ v12.15 - Fenêtre plus grande

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("Green.TButton", padding=6, relief="flat", background="#28a745", foreground="white")
        style.map("Green.TButton", background=[('active', '#218838')])
        # NOUVEAU v12.14 : Style pour le bouton Alpaca Sync
        style.configure("Orange.TButton", padding=6, relief="flat", background="#fd7e14", foreground="white")
        style.map("Orange.TButton", background=[('active', '#e66800')])
        
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0")
        style.configure("TCheckbutton", background="#f0f0f0")
        style.configure("TLabelFrame", background="#f0f0f0", borderwidth=1, relief="groove")
        style.configure("TLabelFrame.Label", background="#f0f0f0", foreground="#003366")
        style.configure("Danger.TCheckbutton", background="#f0f0f0", foreground="#dc3545")

        # NOUVEAU v12.15 : Style pour le Treeview
        style.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Segoe UI', 9))

        # --- NOUVEAU : Style pour le Titre ---
        style.configure("Title.TLabel", background="#f0f0f0", foreground="#28a745", font=("Segoe UI", 18, "bold"), anchor="center")
        
        # --- NOUVEAU v12.19 : Style pour le Sous-titre ---
        style.configure("Subtitle.TLabel", background="#f0f0f0", foreground="#555555", font=("Segoe UI", 9, "italic"), anchor="center")


        # --- Initialiser le "Cerveau" AI ---
        self.ai_logic = TradingAI()
        
        # --- Variables IHM ---
        self.train_start_date_var = tk.StringVar(value=self.ai_logic.start_date_train)
        self.train_end_date_var = tk.StringVar(value=self.ai_logic.end_date_train)
        self.output_directory = None
        self.log_buffer = []
        self.capture_log = False
        
        # --- NOUVEAU v12.14 : Variables Alpaca ---
        self.api = None # Contiendra l'objet API Alpaca
        self.last_allocations = None # Stockera la dernière allocation (map de tickers)
        self.alpaca_key_id_var = tk.StringVar()
        self.alpaca_secret_key_var = tk.StringVar()

        # --- Structure principale (avec scroll) ---
        container = ttk.Frame(root)
        container.pack(fill=tk.BOTH, expand=True)

        main_canvas = tk.Canvas(container, background="#f0f0f0", highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(container, orient="vertical", command=main_canvas.yview)
        
        main_frame = ttk.Frame(main_canvas, padding=10)
        main_frame_id = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def on_frame_configure(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))

        def on_canvas_configure(event):
            main_canvas.itemconfig(main_frame_id, width=event.width)

        main_frame.bind("<Configure>", on_frame_configure)
        main_canvas.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(event):
            delta = 0
            if sys.platform == "linux":
                if event.num == 4: delta = -1
                elif event.num == 5: delta = 1
            else: 
                if abs(event.delta) >= 120:
                    delta = -1 * int(event.delta / 40)
                else:
                    delta = -1 * event.delta
            main_canvas.yview_scroll(delta, "units")

        main_frame.bind_all("<MouseWheel>", _on_mousewheel) 
        main_frame.bind_all("<Button-4>", _on_mousewheel) 
        main_frame.bind_all("<Button-5>", _on_mousewheel) 

        # --- NOUVEAU : Titre ---
        title_label = ttk.Label(main_frame, text="ORION AI", style="Title.TLabel")
        title_label.pack(pady=(5, 0), fill=tk.X)

        # --- NOUVEAU v12.19 : Sous-titre ---
        subtitle_text = "Optimized Risk & Investment Opportunity Navigator"
        subtitle_label = ttk.Label(main_frame, text=subtitle_text, style="Subtitle.TLabel")
        subtitle_label.pack(pady=(0, 5), fill=tk.X) # Réduction du padding
        # --- Fin NOUVEAU v12.19 ---

        # --- Signature ---
        signature_label = ttk.Label(
            main_frame, text="Designed by Antoine Aoun", 
            font=("Segoe UI", 12, "bold"), foreground="#065F46", background="#f0f0f0",
            anchor="center" # Ajouté pour centrer la signature
        )
        signature_label.pack(pady=(0, 10), fill=tk.X) # fill=tk.X pour que l'ancre fonctionne

        # --- ÉTAPE 1 : Sélection des Actifs ---
        assets_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 1 : Sélection des Actifs", padding=10)
        assets_frame.pack(fill=tk.X, pady=5)
        
        assets_left_panel = ttk.Frame(assets_frame, padding=5)
        assets_left_panel.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=5)
        
        btn_load = ttk.Button(assets_left_panel, text="📂 Charger Watchlist (CSV/Excel)",
                              command=lambda: load_watchlist_to_gui(self.ticker_listbox))
        btn_load.pack(pady=10, fill=tk.X)
        
        list_frame = ttk.Frame(assets_left_panel)
        list_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.ticker_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=8,
                                         yscrollcommand=list_scrollbar.set,
                                         font=("Segoe UI", 9))
        list_scrollbar.config(command=self.ticker_listbox.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        assets_right_panel = ttk.Frame(assets_frame, padding=5)
        assets_right_panel.pack(fill=tk.BOTH, side=tk.RIGHT, expand=True, padx=5)
        
        ttk.Label(assets_right_panel, text="...OU Saisie Manuelle", justify=tk.LEFT, font=("Segoe UI", 10, "bold")).pack(pady=5)
        ttk.Label(assets_right_panel, text="(Séparés par des virgules)\nEx: AAPL, LVMH, BTC-USD, GOLD", justify=tk.LEFT).pack(pady=5, fill=tk.X)
        
        self.manual_ticker_entry = tk.Text(assets_right_panel, height=6, font=("Segoe UI", 9), wrap=tk.WORD)
        self.manual_ticker_entry.pack(pady=5, fill=tk.BOTH, expand=True)

        # --- ÉTAPE 1.5 : Sélection des Indicateurs ---
        indicators_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 1.5 : Indicateurs à Afficher (en Prédiction)", padding=10)
        indicators_frame.pack(fill=tk.X, pady=5)
        
        self.show_rsi = tk.BooleanVar(value=True)
        self.show_macd = tk.BooleanVar(value=True)
        self.show_sma = tk.BooleanVar(value=True)
        self.show_bollinger = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(indicators_frame, text="RSI", variable=self.show_rsi).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(indicators_frame, text="MACD", variable=self.show_macd).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(indicators_frame, text="SMA (Moyenne Mobile)", variable=self.show_sma).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(indicators_frame, text="Bollinger (Largeur)", variable=self.show_bollinger).pack(side=tk.LEFT, padx=10)


        # --- ÉTAPE 2 : Opérations IA ---
        mode_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 2 : Opérations IA", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)

        # Panneau 1: Mode Entraînement
        train_panel = ttk.Frame(mode_frame, padding=5)
        train_panel.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5)
        
        train_expl = (
            "MODE ENTRAÎNEMENT (Train)\n"
            "À lancer 1x par semaine ou 1x par mois.\n"
            "1. Entraîne un NOUVEAU cerveau IA sur les\n"
            "   actifs sélectionnés à l'Étape 1.\n"
            "2. Sauvegarde 'ia_model.joblib' et 'ia_scaler.joblib'."
        )
        ttk.Label(train_panel, text=train_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)
        
        date_frame = ttk.Frame(train_panel)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Date Début (Train):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(date_frame, textvariable=self.train_start_date_var, width=12).pack(side=tk.LEFT)
        
        ttk.Label(date_frame, text="Date Fin (Train):").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(date_frame, textvariable=self.train_end_date_var, width=12).pack(side=tk.LEFT)
        
        self.train_button = ttk.Button(train_panel, text="LANCER L'ENTRAÎNEMENT", command=self.start_training)
        self.train_button.pack(pady=10, fill=tk.X)

        ttk.Separator(mode_frame, orient=tk.VERTICAL).pack(fill=tk.Y, side=tk.LEFT, padx=10)

        # Panneau 2: Mode Prédiction
        predict_panel = ttk.Frame(mode_frame, padding=5)
        predict_panel.pack(fill=tk.X, side=tk.RIGHT, expand=True, padx=5)

        predict_expl = (
            "MODE PRÉDICTION (Predict)\n"
            "À lancer chaque jour après la clôture.\n"
            "1. Charge les modèles IA sauvegardés.\n"
            "2. Calcule les signaux pour les actifs de l'Étape 1."
        )
        ttk.Label(predict_panel, text=predict_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)
        
        self.clear_output_dir = tk.BooleanVar(value=False)
        clear_dir_check = ttk.Checkbutton(predict_panel, 
                                          text="Nettoyer le répertoire d'export avant la prédiction", 
                                          variable=self.clear_output_dir,
                                          style="Danger.TCheckbutton")
        clear_dir_check.pack(pady=5, anchor='w')
        
        self.predict_button = ttk.Button(predict_panel, text="LANCER LA PRÉDICTION", command=self.start_prediction, style="Green.TButton")
        self.predict_button.pack(pady=10, fill=tk.X)


        # --- ÉTAPE 3 : Console de Log ---
        log_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 3 : Console de Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        sys.stdout = TextRedirector(self.log_widget, "stdout", self)
        sys.stderr = TextRedirector(self.log_widget, "stderr", self)
        
        self.export_button = ttk.Button(main_frame, text="Définir le Répertoire d'Export (Rapports)", command=self.select_output_directory)
        self.export_button.pack(pady=5, fill=tk.X)

        # --- NOUVEAU v12.14 / MODIFIÉ v12.15 : ÉTAPE 4 : Connexion Alpaca ---
        alpaca_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 4 : Connexion & Trading (Alpaca Paper)", padding=10)
        alpaca_frame.pack(fill=tk.X, pady=10)
        
        alpaca_expl = (
            "Connectez-vous à votre compte Paper Trading Alpaca.\n"
            "Trouvez vos clés dans 'Paper Trading' -> 'API Keys' sur le site d'Alpaca.\n"
            "N'utilisez PAS vos clés de trading réel (Live) ici."
        )
        ttk.Label(alpaca_frame, text=alpaca_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)
        
        key_frame = ttk.Frame(alpaca_frame)
        key_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(key_frame, text="Alpaca Key ID:", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(key_frame, textvariable=self.alpaca_key_id_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        secret_frame = ttk.Frame(alpaca_frame)
        secret_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(secret_frame, text="Alpaca Secret Key:", width=15).pack(side=tk.LEFT, padx=5)
        ttk.Entry(secret_frame, textvariable=self.alpaca_secret_key_var, width=50, show="*").pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        status_frame = ttk.Frame(alpaca_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.connect_button = ttk.Button(status_frame, text="Tester la Connexion (Paper)", command=self.connect_to_alpaca)
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        # NOUVEAU v12.15 : Bouton "Afficher Positions"
        self.show_positions_button = ttk.Button(status_frame, text="Afficher les Positions Actuelles", command=self.start_fetch_positions)
        self.show_positions_button.pack(side=tk.LEFT, padx=5)
        self.show_positions_button.config(state=tk.DISABLED) 
        
        self.alpaca_status_label = ttk.Label(status_frame, text="Statut : Déconnecté", foreground="red", font=("Segoe UI", 9, "bold"))
        self.alpaca_status_label.pack(side=tk.LEFT, padx=10)
        
        # --- NOUVEAU v12.15 : Cadre pour les positions ---
        positions_frame = ttk.LabelFrame(alpaca_frame, text="Positions Actuelles (Paper)", padding=10)
        positions_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        tree_scroll = ttk.Scrollbar(positions_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.positions_tree = ttk.Treeview(positions_frame, columns=("Symbole", "Qty", "Valeur", "P/L Jour", "P/L Total"), show="headings", yscrollcommand=tree_scroll.set, height=5)
        
        self.positions_tree.heading("Symbole", text="Symbole")
        self.positions_tree.heading("Qty", text="Quantité")
        self.positions_tree.heading("Valeur", text="Valeur Actuelle")
        self.positions_tree.heading("P/L Jour", text="P/L Jour ($)")
        self.positions_tree.heading("P/L Total", text="P/L Total ($)")
        
        self.positions_tree.column("Symbole", width=100, anchor=tk.W)
        self.positions_tree.column("Qty", width=80, anchor=tk.E)
        self.positions_tree.column("Valeur", width=120, anchor=tk.E)
        self.positions_tree.column("P/L Jour", width=100, anchor=tk.E)
        self.positions_tree.column("P/L Total", width=100, anchor=tk.E)
        
        self.positions_tree.pack(fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.positions_tree.yview)
        # --- Fin de l'ajout v12.15 ---
        
        sync_expl = (
            "Ce bouton exécutera les ordres d'achat/vente pour correspondre à la dernière allocation\n"
            "calculée par le mode Prédiction. Annule tous les ordres ouverts avant d'exécuter."
        )
        ttk.Label(alpaca_frame, text=sync_expl, justify=tk.LEFT, font=("Segoe UI", 8, "italic")).pack(fill=tk.X, pady=(10,5))
        
        self.sync_button = ttk.Button(alpaca_frame, text="SYNCHRONISER LE PORTEFEUILLE (Paper)", 
                                      command=self.start_alpaca_sync, style="Orange.TButton")
        self.sync_button.pack(pady=5, fill=tk.X)
        self.sync_button.config(state=tk.DISABLED) # Désactivé jusqu'à connexion
        # --- Fin de la section Alpaca ---

        # --- NOUVEAU v13.00 : Chargement auto de la config Alpaca ---
        self.load_alpaca_config()
        # --- Fin v13.00 ---


    # --- MODIFIÉ v12.13 : Fonction pour récupérer les actifs ET la map de noms ---
    def get_selected_assets(self):
        """Récupère et nettoie les tickers depuis l'IHM."""
        
        print("Récupération des actifs sélectionnés...")
        
        ticker_to_name_map = {}
        
        # 1. Actifs de la Listbox
        selected_indices = self.ticker_listbox.curselection()
        selected_names = [self.ticker_listbox.get(i) for i in selected_indices]
        list_tickers = []
        for name in selected_names:
            if name in watchlist_map:
                ticker = watchlist_map[name]
                list_tickers.append(ticker)
                if ticker not in ticker_to_name_map: 
                    ticker_to_name_map[ticker] = name
        
        if list_tickers:
            print(f"        -> Actifs de la Listbox : {list_tickers}")

        # 2. Actifs de la saisie manuelle (nettoyage)
        manual_input_raw = self.manual_ticker_entry.get("1.0", tk.END)
        manual_input_list = [t.strip().upper() for t in manual_input_raw.replace('\n', ',').split(",") if t.strip()]
        
        manual_tickers = []
        if manual_input_list:
            print(f"        -> Actifs de la Saisie Manuelle (brut) : {manual_input_list}")
            for name_or_ticker in manual_input_list:
                ticker = convert_csv_symbol_to_yahoo(name_or_ticker)
                # MODIFIÉ v12.14 : S'assurer que les tickers YFinance sont convertis en Alpaca si nécessaire
                # (ex: "MC.PA" -> "MC") - C'est une simplification, Alpaca EU gère ".PA"
                # Pour l'instant, on suppose que l'utilisateur utilise des tickers US ou compatibles
                manual_tickers.append(ticker)
                if ticker not in ticker_to_name_map:
                    ticker_to_name_map[ticker] = name_or_ticker 
            print(f"        -> Actifs de la Saisie Manuelle (convertis) : {manual_tickers}")
        
        # 3. Combiner et dé-doublonner
        final_tickers = sorted(list(set(list_tickers + manual_tickers)))
        
        if not final_tickers:
            print("[ERREUR] Aucun actif sélectionné.")
            messagebox.showerror("Aucun Actif Sélectionné", "Veuillez charger une watchlist, sélectionner des actifs, ou en entrer manuellement à l'Étape 1.")
            return None, None 
            
        print(f"        -> Liste finale des tickers : {final_tickers}")
        print(f"        -> Map Ticker->Nom : {ticker_to_name_map}")
        return final_tickers, ticker_to_name_map 

    # --- MODIFIÉ v12.15 / REQUÊTE UTILISATEUR v13.06 ---
    def start_training(self):
        """Lance l'entraînement dans un thread séparé pour ne pas geler l'IHM."""
        assets_to_process, _ = self.get_selected_assets() 
        if not assets_to_process:
            return 

        start_date = self.train_start_date_var.get()
        end_date = self.train_end_date_var.get()
        
        # --- AJOUT REQUÊTE UTILISATEUR v13.06 ---
        output_directory = self.output_directory
        if not output_directory:
            messagebox.showwarning("Export non défini", "Le log d'entraînement ne sera pas sauvegardé.\n\nVeuillez d'abord 'Définir le Répertoire d'Export' (Étape 3) si vous souhaitez un rapport.")
        # --- FIN AJOUT ---
            
        if not (start_date and end_date):
            messagebox.showerror("Erreur", "Les dates de début et de fin d'entraînement sont requises.")
            return

        self.log_widget.delete('1.0', tk.END) 
        print(f"Initialisation du thread d'entraînement (Dates: {start_date} à {end_date})...") 
        
        # --- AJOUT REQUÊTE UTILISATEUR v13.06 (Capture Log) ---
        self.log_buffer = []
        self.capture_log = True
        # --- FIN AJOUT ---
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED, text="Entraînement en cours...")
        self.predict_button.config(state=tk.DISABLED)
        self.sync_button.config(state=tk.DISABLED) 
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        
        # --- MODIFICATION REQUÊTE UTILISATEUR v13.06 (Passer output_directory) ---
        self.train_thread = threading.Thread(target=self.run_training_task, args=(assets_to_process, start_date, end_date, output_directory))
        # --- FIN MODIFICATION ---
        self.train_thread.daemon = True
        self.train_thread.start()

    # --- MODIFIÉ v12.15 / REQUÊTE UTILISATEUR v13.06 ---
    def run_training_task(self, assets_to_process, start_date, end_date, output_directory): 
        """Tâche exécutée par le thread d'entraînement."""
        # --- AJOUT REQUÊTE UTILISATEUR v13.06 ---
        captured_log = ""
        # --- FIN AJOUT ---
        
        try:
            self.ai_logic.run_training_mode(assets_to_process, start_date, end_date) 
            
            # --- AJOUT REQUÊTE UTILISATEUR v13.06 (Sauvegarde du log) ---
            print("\n[Rapport] Entraînement terminé. Sauvegarde du log...")
            self.capture_log = False
            captured_log = "".join(self.log_buffer)
            self.log_buffer = []
            
            if output_directory:
                self.generate_training_report(captured_log, output_directory)
            else:
                print("[Rapport] Aucun répertoire d'export défini. Log non sauvegardé sur disque.")
            # --- FIN AJOUT ---
            
        except Exception as e:
            print(f"[ERREUR FATALE DU THREAD] : {e}\nTrace: {traceback.format_exc()}")
            # --- AJOUT REQUÊTE UTILISATEUR v13.06 (Stopper capture en cas d'erreur) ---
            self.capture_log = False
            self.log_buffer = []
            # --- FIN AJOUT ---
        finally:
            print("Thread d'entraînement terminé. Réactivation des boutons.")
            
            def re_enable():
                self.train_button.config(state=tk.NORMAL, text="LANCER L'ENTRAÎNEMENT")
                self.predict_button.config(state=tk.NORMAL)
                self.connect_button.config(state=tk.NORMAL)
                if self.api: 
                    self.sync_button.config(state=tk.NORMAL)
                    self.show_positions_button.config(state=tk.NORMAL)
            self.root.after_idle(re_enable)

    # --- MODIFIÉ v12.15 : Gestion de l'état des boutons
    def start_prediction(self):
        """Lance la prédiction dans un thread séparé."""
        assets_to_process, ticker_to_name_map = self.get_selected_assets() 
        
        if not assets_to_process:
            return
            
        display_prefs = {
            'rsi': self.show_rsi.get(),
            'macd': self.show_macd.get(),
            'sma': self.show_sma.get(),
            'bollinger': self.show_bollinger.get()
        }
        
        output_directory = self.output_directory
        clear_dir_pref = self.clear_output_dir.get()
        self.log_widget.delete('1.0', tk.END) 
        
        self.log_buffer = []
        self.capture_log = True
        
        print("Initialisation du thread de prédiction...") 
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED, text="Prédiction en cours...")
        self.sync_button.config(state=tk.DISABLED) 
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        
        self.predict_thread = threading.Thread(target=self.run_prediction_task, args=(
            assets_to_process, display_prefs, output_directory, clear_dir_pref, ticker_to_name_map
        ))
        self.predict_thread.daemon = True
        self.predict_thread.start()

    # MODIFIÉ v13.04 : Gestion du popup et aperçu des ordres
    def run_prediction_task(self, assets_to_process, display_prefs, output_directory, clear_dir_pref, ticker_to_name_map):
        """Tâche exécutée par le thread de prédiction."""
        captured_log = "" 
        
        # NOUVEAU v12.16: Variables pour stocker les résultats pour le popup
        results_df_popup = None
        alloc_series_popup = None
        
        # NOUVEAU v13.04: Variable pour l'aperçu des ordres
        trade_preview_list = []
        
        try:
            results_df, alloc_series, last_date, full_indicators_history, final_allocations = self.ai_logic.run_prediction_mode(
                assets_to_process, display_prefs, ticker_to_name_map
            )
            
            # Stocker les résultats pour le popup
            results_df_popup = results_df
            alloc_series_popup = alloc_series
            
            if final_allocations:
                self.last_allocations = final_allocations
            
            # --- NOUVEAU v13.04 : Calculer l'aperçu des ordres ---
            if self.api and self.last_allocations: # Seulement si connecté ET prédiction réussie
                try:
                    print("[Alpaca] Calcul de l'aperçu des ordres pour le popup...")
                    trade_preview_list = self.get_trade_preview_list() # Nouvelle fonction
                except Exception as e:
                    print(f"[ERREUR Aperçu Ordres] : {e}")
                    trade_preview_list = [f"Erreur lors du calcul de l'aperçu : {e}"]
            elif not self.api:
                trade_preview_list = ["Non connecté à Alpaca. Aperçu non disponible."]
            else:
                trade_preview_list = ["Prédiction échouée. Aperçu non disponible."]
            # --- Fin v13.04 ---
            
            if output_directory and results_df is not None:
                print(f"\n[Rapport] Génération du rapport dans : {output_directory}")
                
                self.capture_log = False 
                captured_log = "".join(self.log_buffer)
                
                self.generate_report(
                    results_df, alloc_series, last_date, output_directory, 
                    captured_log, full_indicators_history, clear_dir_pref, ticker_to_name_map
                )
            else:
                self.capture_log = False
            
            self.log_buffer = [] 
                
        except Exception as e:
            print(f"[ERREUR FATALE DU THREAD] : {e}\nTrace: {traceback.format_exc()}")
            self.capture_log = False 
            self.log_buffer = []
        finally:
            print("Thread de prédiction terminé. Réactivation des boutons.") 
            
            # MODIFIÉ v13.04: Fonction pour réactiver ET montrer le popup (avec aperçu)
            def re_enable_and_popup():
                self.train_button.config(state=tk.NORMAL)
                self.predict_button.config(state=tk.NORMAL, text="LANCER LA PRÉDICTION")
                self.connect_button.config(state=tk.NORMAL)
                if self.api: 
                    self.sync_button.config(state=tk.NORMAL)
                    self.show_positions_button.config(state=tk.NORMAL)
                
                # Afficher le popup (avec la liste des trades)
                if results_df_popup is not None and alloc_series_popup is not None:
                    self.show_prediction_popup(results_df_popup, alloc_series_popup, trade_preview_list)

            self.root.after_idle(re_enable_and_popup)

    # --- MODIFIÉ v13.05 : Fenêtre Popup de Résultats ---
    # --- MODIFIÉ (Demande utilisateur) : Ajout de styles vert/rouge gras ---
    def show_prediction_popup(self, results_df, alloc_series, trade_preview_list):
        """
        Affiche une fenêtre popup avec les résultats de la prédiction ET l'aperçu des ordres.
        MODIFIÉ : Ajout de tags de style pour les achats (vert) et les ventes (rouge).
        """
        print("[IHM] Affichage de la fenêtre de résultats de prédiction...")
        
        popup = tk.Toplevel(self.root)
        popup.title("Résultats de la Prédiction et Aperçu des Ordres")
        popup.geometry("600x700") # Agrandie
        popup.transient(self.root) # Reste au-dessus de la fenêtre principale
        popup.grab_set() # Modale

        main_frame = ttk.Frame(popup, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section Probabilités ---
        prob_frame = ttk.LabelFrame(main_frame, text="1. Probabilités de Hausse (21J) - IA", padding=5)
        prob_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        prob_text = scrolledtext.ScrolledText(prob_frame, wrap=tk.NONE, height=6, font=("Consolas", 9))
        prob_text.pack(fill=tk.BOTH, expand=True)
        prob_string = results_df.to_string(float_format="%.4f")
        prob_text.insert(tk.END, prob_string)
        prob_text.config(state=tk.DISABLED)

        # --- Section Allocations ---
        alloc_frame = ttk.LabelFrame(main_frame, text="2. Allocation Recommandée (Cible du Robot)", padding=5)
        alloc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        alloc_text = scrolledtext.ScrolledText(alloc_frame, wrap=tk.NONE, height=6, font=("Consolas", 9))
        alloc_text.pack(fill=tk.BOTH, expand=True)
        alloc_string = alloc_series.map('{:.1%}'.format).to_string()
        alloc_text.insert(tk.END, alloc_string)
        alloc_text.config(state=tk.DISABLED)

        # --- MODIFIÉ v13.05 : Section Aperçu des Ordres ---
        preview_frame = ttk.LabelFrame(main_frame, text="3. Aperçu des Ordres (si 'Synchroniser' est cliqué)", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        preview_text = scrolledtext.ScrolledText(preview_frame, wrap=tk.NONE, height=14, font=("Consolas", 9)) # height=14
        preview_text.pack(fill=tk.BOTH, expand=True)
        
        # --- DÉBUT MODIFICATION (Demande utilisateur : vert/rouge gras) ---
        
        # 1. Définir la police grasse
        # Note : Assurez-vous que la police "Consolas" est disponible sur votre système
        # ou remplacez-la par une police monospac_ée standard comme "Courier".
        try:
            bold_font = ("Consolas", 9, "bold")
            # Tester la police pour éviter une erreur si elle n'existe pas
            from tkinter.font import Font
            Font(font=bold_font).actual()
        except tk.TclError:
            print("[Style Popup] Police 'Consolas' non trouvée, utilisation de 'Courier'.")
            bold_font = ("Courier", 9, "bold")
            
        # 2. Définir les tags de style dans le widget de texte
        preview_text.tag_configure("buy_style", foreground="#008000", font=bold_font) # Vert (ex: #008000)
        preview_text.tag_configure("sell_style", foreground="#FF0000", font=bold_font) # Rouge
        
        # 3. Insérer le texte ligne par ligne en appliquant les tags
        if trade_preview_list:
            # Ancienne méthode :
            # preview_string = "\n".join(trade_preview_list)
            # preview_text.insert(tk.END, preview_string)
            
            # Nouvelle méthode :
            for line in trade_preview_list:
                tag_to_use = () # Style par défaut (pas de tag)
                
                # Appliquer le style si le mot-clé est trouvé
                if "[ACHAT]" in line:
                    tag_to_use = ("buy_style",)
                elif "[VENTE]" in line or "[LIQUIDATION]" in line:
                    tag_to_use = ("sell_style",)
                    
                # Insérer la ligne avec le tag approprié et un saut de ligne
                preview_text.insert(tk.END, line + "\n", tag_to_use)
                
        else:
            preview_text.insert(tk.END, "Aucun aperçu d'ordre à générer.")
        
        # --- FIN MODIFICATION ---
        
        preview_text.config(state=tk.DISABLED)
        # --- Fin v13.05 ---

        # --- Bouton OK ---
        ok_button = ttk.Button(main_frame, text="Fermer", command=popup.destroy)
        ok_button.pack(pady=10)
        
        # Centrer le popup
        popup.update_idletasks()
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() - popup.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - popup.winfo_height()) // 2
            popup.geometry(f"+{x}+{y}")
        except Exception as e:
            print(f"[IHM] Erreur lors du centrage du popup: {e}")

    # --- NOUVEAU v13.00 : Chargement de la config Alpaca ---
    # --- MODIFIÉ v13.02 : Utilisation d'un chemin absolu ---
    def load_alpaca_config(self):
        """
        Lit le fichier API_Alpaca.ini et tente une connexion automatique.
        Recherche le .ini dans le MÊME dossier que le script .py.
        Le format attendu du .ini est :
        [alpaca]
        key_id = VOTRE_CLE_ID
        secret_key = VOTRE_CLE_SECRET
        """
        
        # NOUVEAU v13.02 : Logique pour trouver le chemin absolu
        try:
            # Trouve le répertoire où se trouve le script .py
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ n'est pas défini si exécuté dans certains IDE (ex: IDLE interactif)
            # On se rabat sur le répertoire de travail actuel
            script_dir = os.getcwd()
            print("[Config] Avertissement : Impossible de déterminer le chemin du script via __file__. Utilisation du répertoire de travail actuel.")

        config_file_name = 'API_Alpaca.ini'
        config_file_path = os.path.join(script_dir, config_file_name) # Chemin complet
        
        print(f"[Config] Recherche du fichier de configuration : {config_file_path}") # Log de débogage

        config = configparser.ConfigParser()
        try:
            if not os.path.exists(config_file_path): # MODIFIÉ v13.02
                print(f"[Config] Fichier {config_file_name} non trouvé à l'emplacement du script. Saisie manuelle requise.")
                return
                
            config.read(config_file_path) # MODIFIÉ v13.02
            key_id = config.get('alpaca', 'key_id', fallback=None)
            secret_key = config.get('alpaca', 'secret_key', fallback=None)
            
            if key_id and secret_key:
                self.alpaca_key_id_var.set(key_id)
                self.alpaca_secret_key_var.set(secret_key)
                print(f"[Config] Fichier {config_file_name} chargé. Tentative de connexion auto dans 100ms...")
                
                # Utiliser root.after pour laisser l'IHM s'initialiser avant le popup de connexion
                self.root.after(100, self.connect_to_alpaca) 
            else:
                print(f"[Config] Clés 'key_id' ou 'secret_key' non trouvées dans {config_file_name}. Saisie manuelle requise.")
                
        except Exception as e:
            print(f"[ERREUR Config] Impossible de lire {config_file_name} : {e}")
            messagebox.showwarning("Erreur Config", f"Impossible de lire {config_file_name} : {e}")

    # --- MODIFIÉ v13.00 : Logique de Connexion Alpaca ---
    def connect_to_alpaca(self):
        """Tente de se connecter à l'API Paper Trading d'Alpaca."""
        print("[Alpaca] Tentative de connexion (Mode Paper)...")
        key_id = self.alpaca_key_id_var.get()
        secret_key = self.alpaca_secret_key_var.get()
        
        if not key_id or not secret_key:
            messagebox.showerror("Erreur Alpaca", "Veuillez entrer l'API Key ID ET la Secret Key.")
            return

        try:
            base_url = 'https://paper-api.alpaca.markets'
            self.api = tradeapi.REST(key_id, secret_key, base_url, api_version='v2')
            
            # Tester la connexion en récupérant les infos du compte
            account = self.api.get_account()
            print(f"[Alpaca] Connexion réussie au compte Paper : {account.account_number}")
            print(f"[Alpaca] Equity du compte : ${account.equity}")
            
            self.alpaca_status_label.config(text=f"Statut : Connecté (${float(account.equity):,})", foreground="green")
            self.sync_button.config(state=tk.NORMAL) # Activer le bouton de sync
            self.show_positions_button.config(state=tk.NORMAL) # MODIFIÉ v12.15 : Activer le bouton
            # Ne pas afficher de popup si la connexion auto a réussi (silencieux)
            # messagebox.showinfo("Alpaca Succès", f"Connecté avec succès au compte Paper {account.account_number}.\nEquity: ${account.equity}")
            
            # NOUVEAU v12.15 : Lancer un premier affichage des positions
            self.start_fetch_positions()
            
            # --- NOUVEAU v13.00 ---
            # Démarrer le rafraîchissement automatique
            print("[Alpaca] Démarrage du rafraîchissement auto (5 min)...")
            self.schedule_position_refresh() 
            # --- Fin v13.00 ---
            
        except Exception as e:
            print(f"[ERREUR Alpaca] : {e}")
            self.api = None
            self.alpaca_status_label.config(text="Statut : Échec Connexion", foreground="red")
            self.sync_button.config(state=tk.DISABLED)
            self.show_positions_button.config(state=tk.DISABLED) # MODIFIÉ v12.15 : Désactiver
            messagebox.showerror("Erreur Alpaca", f"Échec de la connexion. Vérifiez vos clés et votre connexion internet.\n\nErreur: {e}")

    # --- NOUVEAU v13.00 : Planificateur de rafraîchissement ---
    def schedule_position_refresh(self):
        """
        Planifie le rafraîchissement des positions toutes les 5 minutes.
        """
        refresh_interval_ms = 300000 # 5 minutes
        
        if self.api:
            print("[Alpaca] Rafraîchissement automatique des positions...")
            self.start_fetch_positions()
            # Re-planifier la prochaine exécution
            self.root.after(refresh_interval_ms, self.schedule_position_refresh)
        else:
            print("[Alpaca] Connexion perdue. Arrêt du rafraîchissement automatique.")

    # --- NOUVELLES FONCTIONS v12.15 : Afficher les Positions ---
    def start_fetch_positions(self):
        """Lance la récupération des positions dans un thread."""
        if not self.api:
            # Ne pas montrer d'erreur si c'est juste le refresh auto qui échoue
            if self.alpaca_status_label.cget("foreground") == "green":
                print("[Alpaca] Impossible de rafraîchir : API non connectée.")
            return

        print("[Alpaca] Récupération des positions actuelles...")
        
        # Geler les boutons de la section Alpaca
        self.connect_button.config(state=tk.DISABLED)
        self.show_positions_button.config(state=tk.DISABLED, text="Chargement...")
        self.sync_button.config(state=tk.DISABLED)
        
        self.fetch_thread = threading.Thread(target=self.run_fetch_positions_task)
        self.fetch_thread.daemon = True
        self.fetch_thread.start()

    def run_fetch_positions_task(self):
        """Tâche de thread pour appeler l'API list_positions."""
        positions_data = []
        try:
            positions = self.api.list_positions()
            for p in positions:
                positions_data.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_intraday_pl": float(p.unrealized_intraday_pl),
                    "unrealized_pl": float(p.unrealized_pl)
                })
            
            # Mettre à jour l'IHM dans le thread principal
            self.root.after_idle(self.update_positions_treeview, positions_data)
            
        except Exception as e:
            print(f"[ERREUR Alpaca] Impossible de récupérer les positions: {e}")
            # Ne pas spammer l'utilisateur avec des popups si c'est le refresh auto
            # messagebox.showerror("Erreur Alpaca", f"Impossible de récupérer les positions:\n{e}")
        finally:
            # Réactiver les boutons dans le thread principal
            def re_enable_buttons():
                self.connect_button.config(state=tk.NORMAL)
                self.show_positions_button.config(state=tk.NORMAL, text="Afficher les Positions Actuelles")
                self.sync_button.config(state=tk.NORMAL)
            
            self.root.after_idle(re_enable_buttons)

    def update_positions_treeview(self, positions_data):
        """Met à jour le Treeview avec les données (appelé depuis le thread principal)."""
        # Vider l'ancien contenu
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
        
        if not positions_data:
            print("[Alpaca] Aucune position ouverte trouvée.")
            self.positions_tree.insert("", tk.END, values=("(Aucune position)", "", "", "", ""))
            return
            
        print(f"[Alpaca] Affichage de {len(positions_data)} positions.")
        
        total_value = 0.0
        total_pl_day = 0.0
        total_pl = 0.0
        
        for p in positions_data:
            self.positions_tree.insert("", tk.END, values=(
                p["symbol"],
                f"{p['qty']}",
                f"${p['market_value']:,.2f}",
                f"${p['unrealized_intraday_pl']:,.2f}",
                f"${p['unrealized_pl']:,.2f}"
            ))
            total_value += p['market_value']
            total_pl_day += p['unrealized_intraday_pl']
            total_pl += p['unrealized_pl']
        
        # Ajouter une ligne de total
        self.positions_tree.insert("", tk.END, values=(
            "TOTAL",
            "",
            f"${total_value:,.2f}",
            f"${total_pl_day:,.2f}",
            f"${total_pl:,.2f}"
        ), tags=('total_row',))
        
        self.positions_tree.tag_configure('total_row', font=('Segoe UI', 9, 'bold'))
    # --- Fin des fonctions v12.15 ---

    # --- NOUVEAU v13.04 / MODIFIÉ v13.05 : Fonction d'aperçu des trades (appelée par un thread) ---
    def get_trade_preview_list(self):
        """
        Calcule les ordres de rebalancement SANS les exécuter.
        Retourne une liste de strings pour l'aperçu.
        Cette fonction FAIT des appels API (get_account, list_positions)
        et doit être appelée depuis un thread.
        """
        print("[Alpaca] Démarrage du calcul de l'aperçu des ordres...")
        if not self.api:
            return ["API non connectée."]
        if not self.last_allocations:
            return ["Allocations non calculées (lancez d'abord une prédiction)."]
        
        order_strings = []
        
        try:
            # 1. Obtenir l'Equity
            account = self.api.get_account()
            equity = float(account.equity)
            order_strings.append(f"Equity total du compte : ${equity:,.2f}")
            order_strings.append("-" * 40) # Séparateur

            # 2. Obtenir les Positions Actuelles (et les formater)
            order_strings.append("Positions Actuelles (Portefeuille)")
            positions = self.api.list_positions()
            current_positions_data = {}
            if not positions:
                order_strings.append("  Aucune position ouverte.")
            else:
                for p in positions:
                    current_positions_data[p.symbol] = {
                        'market_value': float(p.market_value),
                        'avg_entry_price': float(p.avg_entry_price),
                        'current_price': float(p.current_price)
                    }
                    order_strings.append(f"  - {p.symbol}: ${float(p.market_value):,.2f} (Prix d'achat: ${float(p.avg_entry_price):,.2f})")
            order_strings.append("-" * 40) # Séparateur

            # 3. Obtenir les Allocations Cibles (et les formater)
            order_strings.append("Allocations Cibles (IA)")
            target_allocations = self.last_allocations
            for symbol, pct in target_allocations.items():
                # Utiliser le ticker YF (ex: MC.PA) pour l'affichage
                order_strings.append(f"  - {symbol}: {pct:.2%}")
            order_strings.append("-" * 40) # Séparateur

            # 4. Identifier tous les symboles à traiter
            current_alpaca_symbols = set(current_positions_data.keys())
            # Convertir les cibles YF (ex: MC.PA) en tickers Alpaca (ex: MC)
            target_alpaca_symbols = {s.split('.')[0] for s in target_allocations.keys()}
            
            # Cibles YF (ex: 'MC.PA') et Positions Alpaca à liquider (ex: 'TSLA')
            all_symbols_to_process_yf = set(target_allocations.keys())
            symbols_to_liquidate_alpaca = current_alpaca_symbols - target_alpaca_symbols
            
            order_strings.append("--- Ordres de Rebalancement Requis ---")

            # 5. Logique de Rebalancement (sur les cibles de l'IA)
            for yf_symbol in all_symbols_to_process_yf:
                if yf_symbol == 'CASH':
                    continue
                    
                alpaca_symbol = yf_symbol.split('.')[0] 
                
                target_pct = target_allocations.get(yf_symbol, 0)
                target_value = equity * target_pct
                
                position_data = current_positions_data.get(alpaca_symbol)
                current_value = position_data['market_value'] if position_data else 0
                
                diff_value = target_value - current_value
                
                # Tolérance de $1 pour éviter les micro-ordres
                if diff_value > 1: # ACHAT
                    order_strings.append(f"  [ACHAT] : {alpaca_symbol} (cible: {yf_symbol})")
                    order_strings.append(f"    Montant : ${diff_value:,.2f}")
                    order_strings.append(f"    Nouvelle valeur cible : ${target_value:,.2f}")

                elif diff_value < -1: # VENTE (partielle ou totale)
                    notional_to_sell = abs(diff_value)
                    realized_pl_estimate = 0.0
                    
                    if position_data:
                        current_price = position_data.get('current_price', 0)
                        avg_entry_price = position_data.get('avg_entry_price', 0)
                        if current_price > 0 and avg_entry_price > 0: # Éviter division par zéro
                            qty_to_sell = notional_to_sell / current_price
                            realized_pl_estimate = (current_price - avg_entry_price) * qty_to_sell
                    
                    order_strings.append(f"  [VENTE] : {alpaca_symbol} (cible: {yf_symbol})")
                    order_strings.append(f"    Montant : ${notional_to_sell:,.2f}")
                    order_strings.append(f"    Nouvelle valeur cible : ${target_value:,.2f}")
                    order_strings.append(f"    Profit/Perte Réalisé Estimé: ${realized_pl_estimate:,.2f}")

                else: # CONSERVER
                    order_strings.append(f"  [CONSERVER] : {alpaca_symbol} (Allocation déjà correcte)")
            
            # 6. Logique de Liquidation (pour les positions non-IA)
            for alpaca_symbol in symbols_to_liquidate_alpaca:
                position_data = current_positions_data.get(alpaca_symbol)
                if not position_data: continue
                
                notional_to_sell = position_data['market_value']
                realized_pl_estimate = 0.0
                current_price = position_data.get('current_price', 0)
                avg_entry_price = position_data.get('avg_entry_price', 0)
                
                if current_price > 0 and avg_entry_price > 0:
                    qty_to_sell = notional_to_sell / current_price
                    realized_pl_estimate = (current_price - avg_entry_price) * qty_to_sell
            
                order_strings.append(f"  [LIQUIDATION] : {alpaca_symbol} (Non dans les cibles IA)")
                order_strings.append(f"    Montant : ${notional_to_sell:,.2f}")
                order_strings.append(f"    Nouvelle valeur cible : $0.00")
                order_strings.append(f"    Profit/Perte Réalisé Estimé: ${realized_pl_estimate:,.2f}")

            order_strings.append("-" * 40)
            order_strings.append("Note: Les P/L sont des estimations avant exécution.")
            return order_strings
            
        except Exception as e:
            print(f"[ERREUR Aperçu Ordres] : {e}\nTrace: {traceback.format_exc()}")
            return [f"Erreur fatale lors du calcul de l'aperçu : {e}"]
    # --- Fin v13.05 ---


    # --- MODIFIÉ v12.15 : Démarrage du Thread de Sync ---
    def start_alpaca_sync(self):
        """Lance la synchronisation du portefeuille dans un thread."""
        if not self.api:
            messagebox.showerror("Erreur", "Veuillez d'abord vous connecter à Alpaca.")
            return
            
        if not self.last_allocations:
            messagebox.showerror("Erreur", "Veuillez d'abord lancer une 'Prédiction' pour calculer les allocations cibles.")
            return
        
        if not messagebox.askyesno("Confirmation Alpaca",
            "Êtes-vous sûr de vouloir synchroniser votre portefeuille (Paper) ?\n"
            "Cela va annuler tous les ordres ouverts et soumettre de nouveaux ordres (ACHAT/VENTE) pour correspondre à l'allocation de l'IA."):
            print("[Alpaca] Synchronisation annulée par l'utilisateur.")
            return

        print("[Alpaca] Initialisation du thread de synchronisation...")
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)
        self.sync_button.config(state=tk.DISABLED, text="Synchronisation en cours...")
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        
        # Démarrer le thread
        self.sync_thread = threading.Thread(target=self.run_alpaca_sync_task)
        self.sync_thread.daemon = True
        self.sync_thread.start()

    # --- MODIFIÉ v12.15 / v13.05 (ET PAR DEMANDE UTILISATEUR) : Tâche de Sync Alpaca ---
    def run_alpaca_sync_task(self):
        """Exécute la logique de rebalancement du portefeuille."""
        print("[Alpaca] Démarrage de la synchronisation du portefeuille...")
        
        # --- NOUVEAU (Demande utilisateur) : Enregistrer l'aperçu avant exécution ---
        try:
            print("[Alpaca] Génération de l'aperçu pour l'enregistrement...")
            # Appelle la fonction qui génère la liste de strings
            trade_preview_for_log = self.get_trade_preview_list()
            # Appelle la nouvelle fonction d'enregistrement
            self.log_trade_execution(trade_preview_for_log)
        except Exception as e:
            print(f"[ERREUR Log] Impossible de générer ou d'enregistrer l'aperçu des trades : {e}")
            # Continuer quand même, le trading est plus important que le log
        # --- FIN DE L'AJOUT ---
        
        try:
            # 1. Obtenir l'état actuel du compte
            account = self.api.get_account()
            equity = float(account.equity)
            print(f"[Alpaca] Equity total du compte : ${equity:,.2f}")
            
            # 2. Obtenir les positions actuelles
            positions = self.api.list_positions()
            current_positions_map = {p.symbol: float(p.market_value) for p in positions}
            print(f"[Alpaca] Positions actuelles : {current_positions_map}")
            
            # 3. Obtenir les allocations cibles (map de tickers)
            target_allocations = self.last_allocations
            print(f"[Alpaca] Allocations cibles (IA) : {target_allocations}")
            
            # 4. Créer la liste de tous les symboles (actuels + cibles)
            # (Logique dupliquée de get_trade_preview_list pour la robustesse)
            current_alpaca_symbols = set(current_positions_map.keys())
            target_alpaca_symbols = {s.split('.')[0] for s in target_allocations.keys()}
            symbols_to_liquidate_alpaca = current_alpaca_symbols - target_alpaca_symbols
            all_symbols_to_process_yf = set(target_allocations.keys())
            
            # 5. Annuler tous les ordres ouverts pour éviter les conflits
            print("[Alpaca] Annulation de tous les ordres ouverts...")
            self.api.cancel_all_orders()
            
            # 6. Logique de rebalancement (par valeur notionnelle)
            print("[Alpaca] Calcul et exécution du rebalancement...")
            
            # 6a. Traiter les cibles de l'IA
            for yf_symbol in all_symbols_to_process_yf:
                if yf_symbol == 'CASH':
                    continue 
                    
                alpaca_symbol = yf_symbol.split('.')[0] 
                
                target_pct = target_allocations.get(yf_symbol, 0) # Ticker YF
                target_value = equity * target_pct
                
                current_value = current_positions_map.get(alpaca_symbol, 0) # Ticker Alpaca
                
                diff_value = target_value - current_value
                
                # Tolérance de $1 pour éviter les micro-ordres
                if diff_value > 1:
                    # Acheter
                    print(f"[Alpaca] ORDRE ACHAT : {alpaca_symbol}, Montant : ${diff_value:.2f} (Cible: ${target_value:.2f})")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            notional=round(diff_value, 2),
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                    except Exception as e:
                        print(f"[ERREUR Ordre Achat] {alpaca_symbol}: {e}")
                        
                elif diff_value < -1:
                    # Vendre
                    print(f"[Alpaca] ORDRE VENTE : {alpaca_symbol}, Montant : ${abs(diff_value):.2f} (Cible: ${target_value:.2f})")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            notional=round(abs(diff_value), 2),
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    except Exception as e:
                        print(f"[ERREUR Ordre Vente] {alpaca_symbol}: {e}")
                else:
                    # Conserver
                    print(f"[Alpaca] CONSERVER : {alpaca_symbol} (Allocation déjà correcte)")
            
            # 6b. Traiter les liquidations
            for alpaca_symbol in symbols_to_liquidate_alpaca:
                current_value = current_positions_map.get(alpaca_symbol, 0)
                if current_value > 1: # Tolérance de $1
                    print(f"[Alpaca] ORDRE LIQUIDATION : {alpaca_symbol}, Montant : ${current_value:.2f} (Non dans les cibles IA)")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            notional=round(current_value, 2), # Vendre la totalité
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    except Exception as e:
                        print(f"[ERREUR Ordre Liquidation] {alpaca_symbol}: {e}")


            print("[Alpaca] Synchronisation terminée.")
            messagebox.showinfo("Alpaca Succès", "Synchronisation du portefeuille terminée.\nConsultez la console pour les détails des ordres.")

        except Exception as e:
            print(f"[ERREUR FATALE Sync Alpaca] : {e}\nTrace: {traceback.format_exc()}")
            messagebox.showerror("Erreur Sync Alpaca", f"La synchronisation a échoué : \n{e}")
        finally:
            print("Thread de synchronisation terminé. Réactivation des boutons et rafraîchissement des positions.")
            # MODIFIÉ v12.15 : Logique de réactivation plus robuste
            def final_sync_updates():
                self.train_button.config(state=tk.NORMAL)
                self.predict_button.config(state=tk.NORMAL)
                # Ne pas réactiver les boutons Alpaca ici
                # Lancer le rafraîchissement des positions, qui gérera la réactivation
                self.start_fetch_positions()
            
            self.root.after_idle(final_sync_updates)

    # --- NOUVELLE FONCTION (Demandée) : Enregistrement des prédictions ---
    def log_trade_execution(self, trade_preview_list):
        """
        Enregistre l'aperçu des ordres (la prédiction) dans un fichier
        avant l'exécution.
        """
        print("[Log] Enregistrement de la prédiction dans 'Predictions_Transmises.txt'...")
        try:
            # 1. Trouver le répertoire du script (logique de load_alpaca_config)
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()
            
            log_file_name = "Predictions_Transmises.txt"
            log_file_path = os.path.join(script_dir, log_file_name)
            
            # 2. Créer le contenu de l'enregistrement
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            separator = "=" * 70
            
            log_content = []
            log_content.append(separator + "\n")
            log_content.append(f"ENREGISTREMENT DU : {timestamp}\n")
            log_content.append(separator + "\n")
            
            if not trade_preview_list:
                log_content.append("[INFO] Aucune donnée d'aperçu à enregistrer.\n")
            else:
                # Ajouter un saut de ligne à chaque ligne de l'aperçu
                log_content.extend([line + "\n" for line in trade_preview_list])
                
            log_content.append("\n\n") # Ajouter de l'espace avant le prochain enregistrement
            
            # 3. Écrire dans le fichier (mode 'append')
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.writelines(log_content)
                
            print(f"[Log] Succès : Prédiction enregistrée dans {log_file_path}")

        except Exception as e:
            print(f"[ERREUR LOG] Impossible d'écrire dans {log_file_name}: {e}")
            # Ne pas bloquer l'exécution du trade pour une erreur de log
            pass
    # --- FIN DE LA NOUVELLE FONCTION ---

    def select_output_directory(self):
        """Ouvre une boîte de dialogue pour choisir un répertoire d'export."""
        directory = filedialog.askdirectory(title="Choisir un répertoire pour les exports")
        if directory:
            self.output_directory = directory
            print(f"Répertoire d'export défini : {self.output_directory}")
            messagebox.showinfo("Répertoire Défini", f"Les rapports seront sauvegardés dans :\n{directory}")

    # --- NOUVELLE FONCTION (REQUÊTE UTILISATEUR v13.06) ---
    def generate_training_report(self, captured_log, output_directory):
        """Génère un simple rapport .txt avec le log de l'entraînement."""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            txt_filename = os.path.join(output_directory, f"report_TRAINING_{timestamp}.txt")
            print(f"        -> Création du rapport TXT d'entraînement : {txt_filename}")
            
            report_string = f"--- Journal d'Événements (Log) de l'Entraînement ---\n"
            report_string += f"--- Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            report_string += "="*70 + "\n\n"
            report_string += captured_log 
            report_string += "\n" + "="*70 + "\n"
            report_string += "--- Fin du Journal d'Événements ---\n"
            
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(report_string)
            
            print(f"[Rapport] Fichier log d'entraînement sauvegardé avec succès dans {output_directory}")

        except Exception as e:
            print(f"[ERREUR Rapport Entraînement] Impossible de générer le rapport : {e}\nTrace: {traceback.format_exc()}")
            messagebox.showerror("Erreur d'Export (Entraînement)", f"Impossible de sauvegarder le rapport d'entraînement :\n{e}")
    # --- FIN NOUVELLE FONCTION ---

    # --- MODIFIÉ V12.18 : Logique de génération des graphiques ---
    def generate_report(self, results_df, alloc_series, last_date, output_directory, captured_log="", full_indicators_history=None, clear_directory=False, ticker_to_name_map=None):
        """Génère et sauvegarde un rapport TXT et des graphes PNG."""
        
        if ticker_to_name_map is None:
            ticker_to_name_map = {} 
            
        try:
            if clear_directory:
                print(f"        -> [!] NETTOYAGE du répertoire d'export : {output_directory}")
                files_deleted = 0
                for f in os.listdir(output_directory):
                    file_path = os.path.join(output_directory, f)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                            print(f"                      -> Fichier supprimé : {f}")
                            files_deleted += 1
                    except Exception as e:
                        print(f"                   [!] Erreur lors de la suppression de {f}: {e}")
                print(f"        -> Nettoyage terminé. {files_deleted} fichiers supprimés.")

            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            txt_filename = os.path.join(output_directory, f"report_AI_Prediction_{timestamp}.txt") # Nom changé pour clarté
            print(f"        -> Création du rapport TXT de Prédiction : {txt_filename}")
            
            report_string = f"--- Journal d'Événements (Log) de la Prédiction ---\n"
            report_string += "Ce log contient toutes les étapes affichées dans la console de l'application.\n"
            report_string += "-"*70 + "\n"
            report_string += captured_log 
            report_string += "\n" + "-"*70 + "\n"
            report_string += "--- Fin du Journal d'Événements ---\n\n"
            report_string += "="*50 + "\n\n"

            report_string += f"Rapport de Prédiction AI - {last_date}\n"
            report_string += f"Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report_string += "="*50 + "\n\n"
            
            report_string += "--- Probabilités de Hausse (21J) ---\n"
            report_string += results_df.to_string(float_format="%.4f") + "\n\n"
            
            report_string += "--- Allocation Recommandée ---\n"
            report_string += alloc_series.map('{:.1%}'.format).to_string() + "\n"
            
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(report_string)

            png_filename = os.path.join(output_directory, f"graph_AI_Probabilites_{timestamp}.png")
            print(f"        -> Création du graphe PNG (Probabilités) : {png_filename}")
            
            if not results_df.empty:
                # --- DÉBUT MODIF v12.17 : Style du graphique de probabilités ---
                plt.style.use('default') # Forcer le style blanc
                plt.figure(figsize=(10, max(5, len(results_df) * 0.4)))
                results_df.sort_values(ascending=True).plot(kind='barh', title=f'Probabilité de Hausse IA (21J) - {last_date}')
                plt.xlabel('Probabilité')
                plt.ylabel('Actifs')
                plt.axvline(0.5, color='grey', linestyle='--', linewidth=1)
                plt.grid(True, linestyle='--', alpha=0.6) # Ajouter une grille
                plt.tight_layout()
                plt.savefig(png_filename)
                plt.close()
                # --- FIN MODIF v12.17 ---
            else:
                print("        -> [!] Aucune donnée de résultat à tracer pour le graphe.")
                
            if full_indicators_history:
                print("        -> Génération des graphiques d'analyse technique (style v12.18)...")
                
                # --- DÉBUT MODIF v12.17 / v12.18 ---
                plt.style.use('default') # S'assurer que le style est blanc

                for asset, indicator_df in full_indicators_history.items(): 
                    
                    display_name = ticker_to_name_map.get(asset, asset)
                    safe_asset_name = display_name.replace('^', '').replace('=', '_').replace('/', '_').replace('.', '_')
                    
                    indicator_filename = os.path.join(output_directory, f"graph_Analyse_Technique_{safe_asset_name}_{timestamp}.png")
                    print(f"                      -> Création du graphe pour : {display_name} ({asset}) ({indicator_filename})")
                    
                    try:
                        # --- Logique de stratégie (croisement SMA20) ---
                        indicator_df['Position'] = np.where(indicator_df['Close'] > indicator_df['SMA_20'], 1, 0)
                        # shift(1) regarde la position de la veille
                        indicator_df['Buy_Signal'] = (indicator_df['Position'] == 1) & (indicator_df['Position'].shift(1) == 0)
                        indicator_df['Sell_Signal'] = (indicator_df['Position'] == 0) & (indicator_df['Position'].shift(1) == 1)
                        
                        # --- Création des 4 panels ---
                        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
                        
                        # --- MODIFICATION v12.18 : Ajout des sous-titres ---
                        main_title = f'Analyse Technique : {display_name} ({asset}) - {last_date}'
                        strategy_title = "Stratégie de suivi de tendance (Croisement Cours / SMA 20)"
                        context_note = "Note : Ces signaux SMA sont pour le contexte visuel (la décision de l'IA est séparée)."
                        
                        fig.suptitle(main_title, fontsize=16, y=0.99)
                        fig.text(0.5, 0.96, strategy_title, ha='center', fontsize=10, style='italic', color='black')
                        fig.text(0.5, 0.94, context_note, ha='center', fontsize=9, style='italic', color='gray')
                        # --- FIN MODIFICATION v12.18 ---

                        # --- Panel 1: Cours et tendances ---
                        axes[0].plot(indicator_df.index, indicator_df['Close'], label='Cours de clôture', color='black', linewidth=1.5)
                        axes[0].plot(indicator_df.index, indicator_df['SMA_20'], label='SMA (20)', color='orange', linestyle='-')
                        axes[0].plot(indicator_df.index, indicator_df['BB_Upper'], label='Bollinger Haute', color='red', linestyle='--', alpha=0.7)
                        axes[0].plot(indicator_df.index, indicator_df['BB_Lower'], label='Bollinger Basse', color='green', linestyle='--', alpha=0.7)
                        
                        # Signaux d'achat
                        buy_signals = indicator_df[indicator_df['Buy_Signal']]
                        axes[0].plot(buy_signals.index, buy_signals['Close'] * 0.98, '^', markersize=10, color='green', label='Achat', markeredgecolor='black')
                        
                        # Signaux de vente
                        sell_signals = indicator_df[indicator_df['Sell_Signal']]
                        axes[0].plot(sell_signals.index, sell_signals['Close'] * 1.02, 'v', markersize=10, color='red', label='Vente', markeredgecolor='black')
                        
                        axes[0].set_title('Cours et tendances')
                        axes[0].legend()
                        axes[0].grid(True, linestyle='--', alpha=0.6)

                        # --- Panel 2: RSI ---
                        axes[1].plot(indicator_df.index, indicator_df['RSI'], label='RSI (14 jours)', color='purple')
                        axes[1].axhline(70, color='red', linestyle='--', linewidth=0.7, label='Surchat (70)')
                        axes[1].axhline(30, color='green', linestyle='--', linewidth=0.7, label='Survente (30)')
                        axes[1].set_title('RSI (Relative Strength Index)')
                        axes[1].legend()
                        axes[1].grid(True, linestyle='--', alpha=0.6)

                        # --- Panel 3: MACD ---
                        axes[2].plot(indicator_df.index, indicator_df['MACD'], label='MACD', color='blue')
                        axes[2].plot(indicator_df.index, indicator_df['Signal'], label='Signal', color='orange')
                        axes[2].axhline(0, color='black', linestyle='--', linewidth=0.5)
                        axes[2].set_title('MACD et Ligne de Signal')
                        axes[2].legend()
                        axes[2].grid(True, linestyle='--', alpha=0.6)
                        
                        # --- Panel 4: Position de la StratégIE ---
                        axes[3].plot(indicator_df.index, indicator_df['Position'], label='Position (1=Long, 0=Neutre)', color='blue')
                        axes[3].set_ylim(-0.1, 1.1)
                        axes[3].set_title('Position de la Stratégie (Basée sur SMA20)')
                        axes[3].legend()
                        axes[3].grid(True, linestyle='--', alpha=0.6)

                        # --- Finalisation ---
                        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Ajusté pour les sous-titres
                        plt.savefig(indicator_filename)
                        plt.close(fig) 
                        
                    except Exception as e:
                        print(f"                   [!] Erreur lors de la création du graphe pour {asset}: {e}")
                # --- FIN MODIF v12.17 ---

            print(f"[Rapport] Fichiers sauvegardés avec succès dans {output_directory}")

        except Exception as e:
            print(f"[ERREUR Rapport] Impossible de générer le rapport : {e}\nTrace: {traceback.format_exc()}")
            messagebox.showerror("Erreur d'Export", f"Impossible de sauvegarder le rapport :\n{e}")

# =====================================================
# === CLASSE POUR REDIRIGER LES PRINT VERS L'IHM ===
# =====================================================
class TextRedirector:
    def __init__(self, widget, tag="stdout", app_gui=None): 
        self.widget = widget
        self.tag = tag
        self.app_gui = app_gui 

    def write(self, s):
        def task():
            try:
                # v12.20: Ajout d'une vérification d'existence pour éviter les erreurs TclError à la fermeture
                if self.widget.winfo_exists():
                    self.widget.configure(state="normal")
                    self.widget.insert(tk.END, s, (self.tag,))
                    self.widget.see(tk.END) 
                    self.widget.configure(state="disabled")
                    
                    if self.app_gui and self.app_gui.capture_log:
                        self.app_gui.log_buffer.append(s)
            
            except tk.TclError:
                # Le widget a été détruit (l'application se ferme)
                pass
            except Exception as e:
                # Capturer d'autres erreurs inattendues
                # Écrire sur le stderr original pour éviter une boucle
                print(f"[TextRedirector ERREUR]: {e}", file=sys.__stderr__) 
                
        # Utiliser after_idle pour s'assurer que cela s'exécute sur le thread principal de Tkinter
        try:
            # v12.20: S'assurer que le widget existe avant même de planifier la tâche
            if self.widget.winfo_exists():
                self.widget.after_idle(task)
        except Exception:
            # L'application est déjà en cours de fermeture
            pass

    def flush(self):
        pass

# =====================================================
# === POINT D'ENTRÉE PRINCIPAL ===
# =====================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()