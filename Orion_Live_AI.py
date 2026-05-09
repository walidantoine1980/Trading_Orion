# =====================================================
# PORTFOLIO OPTIMIZER AI - v16.0 (HYBRID EDITION)
# =====================================================
#
# OBJECTIF : VERSION HYBRIDE (TECHNIQUE + FONDAMENTALE)
#
# Modifications demandées :
# 1. Intégration des données fondamentales via yfinance :
#    - PER (Price Earning Ratio)
#    - Croissance du Chiffre d'Affaires (Revenue Growth)
# 2. Ces données sont ajoutées comme "Features" dans le moteur XGBoost.
#
# Le reste du code (GUI, Alpaca, Logique wfa) est inchangé.
# =====================================================
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import traceback
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from time import time
import matplotlib
matplotlib.use('Agg') # Force mode Headless pour éviter le plantage sur serveur Cloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# --- MODIFIÉ v13.09 : Suppression de l'import de calibration ---
# from sklearn.calibration import CalibratedClassifierCV 
import warnings
import joblib
import sys
import threading
import os
import csv
import requests # <-- AJOUTÉ POUR L'AGENT DE RECHERCHE
import re     # <-- AJOUTÉ v12.31 POUR REGEX FOREX
from GoogleNews import GoogleNews


# --- NOUVEAU v13.00 : Import pour la config Alpaca ---
import configparser

# --- NOUVEAU v12.14 : Import pour Alpaca ---
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("ERREUR: La bibliothèque 'alpaca-trade-api' n'est pas installée.")
    print("Veuillez l'installer avec : pip install alpaca-trade_api")
    # On ne quitte pas, l'utilisateur peut vouloir seulement entraîner
    
# --- Imports pour l'IHM (v11 + Dashboard) ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, scrolledtext
except ImportError:
    pass # Permet l'importation de la classe TradingAI sur serveur Headless sans erreur

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
# === CLASSE DE LOGIQUE MÉTIER (v22.1 - Hybrid News) ===
# =====================================================
class TradingAI:
    def __init__(self):
        print("Initialisation du système AI Optimizer (Version Grand Trader v22.1 - Hybrid News)...")
        
        # --- Initialisation NLTK (Sentiment Analysis) ---
        try:
            # Vérifie si le lexique est déjà présent
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("  [Init] Téléchargement du lexique VADER pour l'analyse de sentiment...")
            nltk.download('vader_lexicon', quiet=True)

        self.REGIME_TICKER = "^VIX"
        self.start_date_train = "2009-01-01"
        self.end_date_train = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # --- LISTE DES 34 FEATURES ---
        self.FEATURE_NAMES = [
            "RSI", "MACD", "Signal", "Momentum_1D", "SMA_20", "BB_Width", "Volume_Change", "Market_Regime", 
            "PER", "Rev_Growth", "Debt_to_Equity", "Profit_Margin", "Free_Cash_Flow",
            "Price_to_Book", "ROE", "Earnings_Growth", "Current_Ratio", "Beta",
            "ATR_Normalized", "Dist_SMA50", "PEG_Ratio", "Div_Yield",
            "Dist_VWAP", "Z_Score", "Vol_Ratio", "SMA_Slope", "Price_Vol_Impact",
            "Stoch_RSI", "Force_Index", "Ichi_Tenkan", "Keltner_Pos", "Ret_Skew",
            # --- AJOUT v22.0 ---
            "News_Sentiment", "Event_MA"
        ]
        self.FEATURE_NAMES_NO_REGIME = [f for f in self.FEATURE_NAMES if f != "Market_Regime"]
        
        self.MODEL_FILE = "ia_model.joblib"
        self.SCALER_FILE = "ia_scaler.joblib"
        
        self.current_feature_names = self.FEATURE_NAMES_NO_REGIME 
        
        # Cache pour les données fondamentales
        self.fundamental_cache = {}

    # --- NOUVEAU v22.1 : Analyse Hybride (Yahoo + Google) ---
    def get_news_sentiment_and_events(self, ticker):
        """
        Analyse les news via YAHOO FINANCE et GOOGLE NEWS pour extraire :
        1. Le Sentiment Score (de -1 à 1) via NLTK VADER (Moyenne des sources)
        2. Un flag M&A (Fusion/Acquisition) (Si détecté par l'une des sources)
        """
        sia = SentimentIntensityAnalyzer()
        merger_keywords = ['merger', 'acquisition', 'acquire', 'buyout', 'takeover', 'merge', 'bid', 'tender offer']
        
        scores = []
        merger_detected = 0
        
        # --- 1. SOURCE YAHOO FINANCE ---
        try:
            t = yf.Ticker(ticker)
            news_yf = t.news
            if news_yf:
                for item in news_yf:
                    title = item.get('title', '').lower()
                    
                    # Sentiment Yahoo
                    sentiment = sia.polarity_scores(title)
                    scores.append(sentiment['compound'])
                    
                    # Detection M&A Yahoo
                    if any(word in title for word in merger_keywords):
                        merger_detected = 1
        except Exception:
            # Continue silencieusement vers Google si Yahoo échoue
            pass

        # --- 2. SOURCE GOOGLE NEWS ---
        try:
            # Import local pour éviter erreur si librairie absente en haut
            from GoogleNews import GoogleNews
            
            # Config: Anglais, Période 2 jours
            googlenews = GoogleNews(lang='en', period='2d')
            # Recherche précise : Ticker + "stock"
            googlenews.search(f"{ticker} stock news")
            results_gn = googlenews.result()
            
            if results_gn:
                # On prend les 5 premières news pertinentes
                for item in results_gn[:5]:
                    title = item.get('title', '').lower()
                    desc = item.get('desc', '').lower()
                    full_text = f"{title} {desc}"
                    
                    # Sentiment Google
                    sentiment = sia.polarity_scores(full_text)
                    scores.append(sentiment['compound'])
                    
                    # Detection M&A Google
                    if any(word in full_text for word in merger_keywords):
                        merger_detected = 1
            
            googlenews.clear() # Nettoyage important
            
        except ImportError:
            print(f"                   [!] Librairie 'GoogleNews' non installée. Seul Yahoo utilisé.")
        except Exception as e:
            # print(f"                   [!] Erreur Google News pour {ticker}: {e}")
            pass

        # --- 3. AGRÉGATION ---
        avg_sentiment = np.mean(scores) if scores else 0.0
        
        # Logs informatifs
        if merger_detected:
            print(f"                   [INFO] M&A détecté (Hybride) sur {ticker} !")
        if abs(avg_sentiment) > 0.4:
            print(f"                   [INFO] Sentiment News fort pour {ticker} : {avg_sentiment:.2f} ({len(scores)} articles)")

        return avg_sentiment, merger_detected

    # --- Récupération étendue des 12 fondamentaux (Inchangé) ---
    def get_fundamentals(self, ticker):
        if ticker in self.fundamental_cache:
            return self.fundamental_cache[ticker]
        
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            per = info.get('trailingPE', np.nan)
            if per is None: per = info.get('forwardPE', np.nan)
            rev_growth = info.get('revenueGrowth', np.nan)
            debt_equity = info.get('debtToEquity', np.nan)
            profit_margin = info.get('profitMargins', np.nan)
            fcf = info.get('freeCashflow', np.nan)
            pb = info.get('priceToBook', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            earn_growth = info.get('earningsGrowth', np.nan)
            curr_ratio = info.get('currentRatio', np.nan)
            beta = info.get('beta', np.nan)
            peg = info.get('pegRatio', np.nan)
            if pd.isna(peg) and not pd.isna(per) and not pd.isna(earn_growth) and earn_growth > 0:
                peg = per / (earn_growth * 100)
            div_yield = info.get('dividendYield', np.nan)
            
            data_tuple = (per, rev_growth, debt_equity, profit_margin, fcf, 
                          pb, roe, earn_growth, curr_ratio, beta, 
                          peg, div_yield)
            self.fundamental_cache[ticker] = data_tuple
            return data_tuple
            
        except Exception as e:
            fallback = (np.nan,) * 12
            self.fundamental_cache[ticker] = fallback
            return fallback

    # --- Fonctions Indicateurs de base ---
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

    # --- Création des Features ---
    def create_features(self, asset_data, regime_df, feature_list, include_target=True, fundamentals=None, news_data=None):
        df = pd.DataFrame()
        df["Close"] = asset_data["Close"]
        
        if "High" in asset_data.columns: df["High"] = asset_data["High"]
        else: df["High"] = df["Close"] 
        
        if "Low" in asset_data.columns: df["Low"] = asset_data["Low"]
        else: df["Low"] = df["Close"]

        if "Volume" in asset_data.columns and not asset_data["Volume"].isnull().all():
            df["Volume"] = asset_data["Volume"]
        else:
            df["Volume"] = 0
        df["Volume"] = df["Volume"].replace(0, 1e-6)

        if not regime_df.empty:
            df = df.join(regime_df, how='left') 
        
        # --- Indicateurs Techniques Standards ---
        df["RSI"] = self.compute_rsi(df["Close"])
        df["MACD"], df["Signal"] = self.compute_macd(df["Close"])
        df["SMA_20"] = self.compute_sma(df["Close"])
        bb_sma, bb_upper, bb_lower = self.compute_bollinger(df["Close"])
        df["BB_Width"] = (bb_upper - bb_lower) / bb_sma.replace(0, 1e-9)
        df["BB_Upper"] = bb_upper
        df["BB_Lower"] = bb_lower
        
        df["Volume_Change"] = df["Volume"].pct_change(fill_method=None)
        df["Momentum_1D"] = df["Close"].pct_change(fill_method=None)
        
        # --- Features Techniques v19.0 ---
        prev_close = df["Close"].shift(1)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - prev_close).abs()
        tr3 = (df["Low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()
        df["ATR_Normalized"] = df["ATR_14"] / df["Close"]
        
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Dist_SMA50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
        
        # --- Features Techniques v20.0 ---
        hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP_20"] = (hlc3 * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
        df["Dist_VWAP"] = (df["Close"] - df["VWAP_20"]) / df["VWAP_20"]

        rolling_std = df["Close"].rolling(20).std()
        df["Z_Score"] = (df["Close"] - df["SMA_20"]) / rolling_std.replace(0, 1e-9)

        pct_change_series = df["Close"].pct_change()
        vol_short = pct_change_series.rolling(5).std()
        vol_long = pct_change_series.rolling(60).std()
        df["Vol_Ratio"] = vol_short / vol_long.replace(0, 1e-9)

        df["SMA_Slope"] = df["SMA_20"].diff(3) / df["SMA_20"]
        df["Price_Vol_Impact"] = df["Momentum_1D"] * np.log(df["Volume"] + 1)

        # --- Features v21.0 ---
        min_rsi = df["RSI"].rolling(window=14).min()
        max_rsi = df["RSI"].rolling(window=14).max()
        df["Stoch_RSI"] = (df["RSI"] - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)

        raw_force_index = df["Close"].diff(1) * df["Volume"]
        df["Force_Index"] = raw_force_index.ewm(span=13, adjust=False).mean()

        high_9 = df["High"].rolling(window=9).max()
        low_9 = df["Low"].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        df["Ichi_Tenkan"] = (df["Close"] - tenkan_sen) / tenkan_sen.replace(0, 1e-9)

        keltner_ema = df["Close"].ewm(span=20, adjust=False).mean()
        keltner_upper = keltner_ema + (2 * df["ATR_14"])
        keltner_lower = keltner_ema - (2 * df["ATR_14"])
        keltner_range = keltner_upper - keltner_lower
        df["Keltner_Pos"] = (df["Close"] - keltner_lower) / keltner_range.replace(0, 1e-9)

        df["Ret_Skew"] = pct_change_series.rolling(window=20).skew()

        # --- Injection des Fondamentaux ---
        if fundamentals is None: fundamentals = (np.nan,) * 12
        (per_val, growth_val, de_val, pm_val, fcf_val, pb_val, roe_val, earn_growth_val, curr_ratio_val, beta_val, peg_val, div_val) = fundamentals
        
        def fix_nan(val): return 0 if pd.isna(val) else val
        
        df["PER"] = fix_nan(per_val)
        df["Rev_Growth"] = fix_nan(growth_val)
        df["Debt_to_Equity"] = fix_nan(de_val)
        df["Profit_Margin"] = fix_nan(pm_val)
        df["Free_Cash_Flow"] = fix_nan(fcf_val)
        df["Price_to_Book"] = fix_nan(pb_val)
        df["ROE"] = fix_nan(roe_val)
        df["Earnings_Growth"] = fix_nan(earn_growth_val)
        df["Current_Ratio"] = fix_nan(curr_ratio_val)
        df["Beta"] = fix_nan(beta_val)
        df["PEG_Ratio"] = fix_nan(peg_val)
        df["Div_Yield"] = fix_nan(div_val)
        
        # --- Injection News & M&A ---
        if news_data is None:
            sentiment_val = 0.0
            ma_val = 0
        else:
            sentiment_val, ma_val = news_data
            
        df["News_Sentiment"] = sentiment_val
        df["Event_MA"] = ma_val

        # --- Nettoyage final ---
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
            df.fillna(0, inplace=True) 
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
            
            self.current_feature_names = self.FEATURE_NAMES
            return regime_feature
        
        except Exception as e:
            print(f"        -> [!] Erreur lors du téléchargement du {self.REGIME_TICKER}: {e}")
            self.current_feature_names = self.FEATURE_NAMES_NO_REGIME
            return pd.DataFrame() 

    # --- MODE ENTRAÎNEMENT ---
    def run_training_mode(self, assets_to_process, start_date, end_date):
        try:
            print(f"--- Entraînement v22.1 (34 Features Hybrid) sur {len(assets_to_process)} actif(s) ---")

            data_full = yf.download(assets_to_process, start=start_date, end=end_date, auto_adjust=True)
            data_full.ffill(inplace=True)
            regime_feature = self.get_regime_feature(start_date, end_date)

            features_train, targets_train = [], []
            for asset in assets_to_process:
                try:
                    asset_train_data = data_full.xs(asset, level=1, axis=1)
                except (KeyError, pd.errors.PerformanceWarning):
                    if len(assets_to_process) == 1: asset_train_data = data_full 
                    else: continue
                except Exception: continue 

                try:
                    if not asset_train_data['Close'].dropna().empty:
                        fundamentals_tuple = self.get_fundamentals(asset)
                        
                        # News = None pour l'entraînement (valeurs neutres 0.0)
                        X, y = self.create_features(
                            asset_train_data, 
                            regime_feature, 
                            self.current_feature_names, 
                            include_target=True,
                            fundamentals=fundamentals_tuple,
                            news_data=None 
                        )
                        if not X.empty:
                            features_train.append(X)
                            targets_train.append(y)
                except Exception: continue

            if not features_train:
                print("[ERREUR] Aucun échantillon d'apprentissage.")
                return False, None 

            X_total_train = pd.concat(features_train)
            y_total_train = pd.concat(targets_train)
            
            scaler = StandardScaler()
            X_scaled_train = scaler.fit_transform(X_total_train[self.current_feature_names]) 

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                enable_categorical=True     
            )
            
            model.fit(X_scaled_train, y_total_train) 
            
            joblib.dump(model, self.MODEL_FILE)
            joblib.dump(scaler, self.SCALER_FILE)
            
            print(f"[Sauvegarde LIVE] Modèle v22.1 Hybrid mis à jour.")
            return True, model 
            
        except Exception as e:
            print(f"\n[ERREUR CRITIQUE PENDANT L'ENTRAÎNEMENT] : {e}\nTrace: {traceback.format_exc()}")
            return False, None

    # --- MODE PRÉDICTION (News Live Hybride) ---
    def run_prediction_mode(self, assets_to_process, display_prefs, ticker_to_name_map=None, use_live_data=True, custom_start_date=None, custom_end_date=None):
        try:
            print("\n" + "="*50)
            print("--- DÉMARRAGE DU MODE PRÉDICTION (v22.1 Hybrid) ---")
            
            if ticker_to_name_map is None:
                ticker_to_name_map = {ticker: ticker for ticker in assets_to_process}

            try:
                model = joblib.load(self.MODEL_FILE)
                scaler = joblib.load(self.SCALER_FILE)
                if len(scaler.mean_) != len(self.current_feature_names):
                    print(f"        -> [ATTENTION] Mismatch features. Scaler: {len(scaler.mean_)}, Script: {len(self.current_feature_names)}")
                    print("        -> Veuillez relancer un ENTRAÎNEMENT COMPLET.")
            except FileNotFoundError:
                print(f"        -> [ERREUR] Modèles non trouvés. Lancez l'entraînement d'abord.")
                return None, None, None, None, None 

            if use_live_data:
                start_date_data = (pd.Timestamp.now() - pd.DateOffset(days=730)).strftime('%Y-%m-%d')
                end_date_data = pd.Timestamp.now().strftime('%Y-%m-%d')
                last_date = end_date_data 
            else: 
                start_date_data = (pd.to_datetime(custom_start_date) - pd.DateOffset(days=180)).strftime('%Y-%m-%d')
                end_date_data = custom_end_date
                last_date = custom_end_date 

            print(f"        -> Téléchargement des données YFinance...")
            data_live = yf.download(assets_to_process, start=start_date_data, end=end_date_data, auto_adjust=True)
            data_live.ffill(inplace=True)
            regime_feature = self.get_regime_feature(start_date_data, end_date_data)

            live_features_dict = {}
            live_indicators_dict = {}
            full_indicators_history = {}
            
            for asset in assets_to_process:
                try:
                    asset_live_data = data_live.xs(asset, level=1, axis=1)
                except (KeyError, pd.errors.PerformanceWarning):
                     if len(assets_to_process) == 1: asset_live_data = data_live
                     else: continue
                except Exception: continue 

                try:
                    if not asset_live_data['Close'].dropna().empty:
                        fundamentals_tuple = self.get_fundamentals(asset)

                        # --- LIVE : Récupération News HYBRIDE ---
                        news_tuple = None
                        if use_live_data:
                            print(f"                   Analyses des news (Yahoo+Google) pour {asset}...")
                            news_tuple = self.get_news_sentiment_and_events(asset)
                        # ----------------------------------------

                        X_live_full, df_indicators = self.create_features(
                            asset_live_data, 
                            regime_feature, 
                            self.current_feature_names, 
                            include_target=False,
                            fundamentals=fundamentals_tuple,
                            news_data=news_tuple 
                        )
                        if not X_live_full.empty:
                            if use_live_data:
                                features_to_predict = X_live_full.iloc[-1:]
                                indicators_to_display = df_indicators.iloc[-1:]
                            else:
                                prediction_date_str = custom_end_date
                                filtered_df = X_live_full[X_live_full.index <= prediction_date_str].iloc[-1:]
                                filtered_indicators = df_indicators[df_indicators.index <= prediction_date_str].iloc[-1:]
                                if filtered_df.empty: continue
                                features_to_predict = filtered_df
                                indicators_to_display = filtered_indicators
                                last_date = features_to_predict.index[0].strftime('%Y-%m-%d')
                                
                            live_features_dict[asset] = features_to_predict
                            live_indicators_dict[asset] = indicators_to_display
                            full_indicators_history[asset] = df_indicators 
                        else:
                            print(f"                   [!] Pas de features pour {asset}")
                except Exception as e:
                    print(f"                   [!] Erreur calcul features {asset}: {e}")
            
            if not live_features_dict:
                print("[ERREUR] Impossible de calculer les features live.")
                return None, None, None, None, None 
            
            # --- Indicateurs & Infos ---
            if use_live_data:
                print(f"--- [ÉTAPE 3.5] Indicateurs & Infos (v22.1 Hybrid) ---")
                for asset, ind in live_indicators_dict.items():
                    print(f"        --- {ticker_to_name_map.get(asset, asset)} ---")
                    try:
                        per = ind['PER'].iloc[0]
                        pb = ind['Price_to_Book'].iloc[0]
                        stoch = ind['Stoch_RSI'].iloc[0]
                        # Nouveaux affichages
                        sent = ind['News_Sentiment'].iloc[0]
                        ma = ind['Event_MA'].iloc[0]
                        ma_txt = "OUI" if ma == 1 else "Non"
                        
                        print(f"             [VALO]    PER: {per:.2f} | P/B: {pb:.2f}")
                        print(f"             [TECH]    Stoch.RSI: {stoch:.2f}")
                        print(f"             [NEWS]    Sentiment: {sent:.2f} | M&A Détecté: {ma_txt}")
                        
                    except Exception as e:
                        print(f"             [!] Erreur affichage: {e}")
                print("-------------------------------------------------------\n")

            # --- Prédiction ---
            live_probabilities = {}
            for asset, features_df in live_features_dict.items():
                try:
                    features_df_ordered = features_df[self.current_feature_names]
                    X_scaled_live = scaler.transform(features_df_ordered)
                    prob_hausse = model.predict_proba(X_scaled_live)[:, 1]
                    live_probabilities[asset] = prob_hausse[0]
                except Exception as e:
                    print(f"                      [!] Erreur prédiction {asset}: {e}")
                    live_probabilities[asset] = np.nan
            
            # --- Allocation ---
            results_df = pd.Series(live_probabilities, name="Probabilité_Hausse_IA_21J")
            results_df.index = results_df.index.map(lambda t: f"{ticker_to_name_map.get(t, t)} ({t})") 
            results_df = results_df.sort_values(ascending=False)
            
            weights = {asset: max(0, (prob - 0.5) * 2) for asset, prob in live_probabilities.items()}
            total_raw_weight = sum(weights.values())
            
            final_allocations = {}
            cash_weight = 1.0

            if total_raw_weight > 1.0:
                final_allocations = {asset: w / total_raw_weight for asset, w in weights.items()}
                cash_weight = 0.0
            elif total_raw_weight > 0:
                final_allocations = weights
                cash_weight = 1.0 - total_raw_weight
            
            final_allocations['CASH'] = cash_weight
            
            alloc_series = pd.Series(final_allocations, name="Allocation_Recommandée")
            alloc_series.index = alloc_series.index.map(lambda t: f"{ticker_to_name_map.get(t, t)} ({t})" if t != "CASH" else "CASH")
            alloc_series = alloc_series.sort_values(ascending=False)
            
            if use_live_data:
                print(results_df)
                print("\n--- Allocation (v22.1 Grand Trader Hybrid) ---")
                print(alloc_series.map('{:.1%}'.format))
                print("\n" + "="*50)
                print("--- MODE PRÉDICTION TERMINÉ ---")
                print("="*50 + "\n")
            
            return results_df, alloc_series, last_date, full_indicators_history, final_allocations

        except Exception as e:
            print(f"\n[ERREUR CRITIQUE PENDANT LA PRÉDICTION] : {e}\nTrace: {traceback.format_exc()}")
            return None, None, None, None, None
# =====================================================
# === CLASSE DE L'IHM (L'Interface Graphique) ===
# =====================================================
class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Orion AI v16.0 Hybrid") # MODIFIÉ v16.0
        root.geometry("850x950") # MODIFIÉ v12.15 - Fenêtre plus grande

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- NOUVEAU STYLE (Adaptation Lanceur) ---
        # Style Bipolaire pour les opérations de mise à jour (Standard/WFA) et de Trading (LIVE)
        
        # Style Entraînement Standard (Bleu - Opération Globale)
        style.configure("Blue.TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        style.map("Blue.TButton", background=[('active', '#0056b3')])
        
        # Style Prédiction LIVE (Vert - Opération Finale/Trading)
        style.configure("Green.TButton", padding=6, relief="flat", background="#28a745", foreground="white")
        style.map("Green.TButton", background=[('active', '#218838')])
        
        # Style WFA (Violet - Backtest/Adaptatif) - Conservé pour la distinction
        style.configure("Purple.TButton", padding=6, relief="flat", background="#6f42c1", foreground="white")
        style.map("Purple.TButton", background=[('active', '#5a34a3')])
        
        # Style Synchronisation Alpaca (Orange - Opération Délicate) - Conservé
        style.configure("Orange.TButton", padding=6, relief="flat", background="#fd7e14", foreground="white")
        style.map("Orange.TButton", background=[('active', '#e66800')])
        
        # Configuration des cadres et labels pour le fond gris (couleur par défaut de Tkinter/ttk)
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0")
        style.configure("TCheckbutton", background="#f0f0f0")
        style.configure("TLabelFrame", background="#f0f0f0", borderwidth=1, relief="groove")
        style.configure("TLabelFrame.Label", background="#f0f0f0", foreground="#003366")
        style.configure("Danger.TCheckbutton", background="#f0f0f0", foreground="#dc3545")

        # Styles de texte
        style.configure("Title.TLabel", background="#f0f0f0", foreground="#28a745", font=("Segoe UI", 18, "bold"), anchor="center")
        style.configure("Subtitle.TLabel", background="#f0f0f0", foreground="#555555", font=("Segoe UI", 9, "italic"), anchor="center")

        # NOUVEAU v12.15 : Style pour le Treeview
        style.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Segoe UI', 9))
        # --- FIN NOUVEAU STYLE ---

        # --- Initialiser le "Cerveau" AI ---
        self.ai_logic = TradingAI()
        
        # --- Variables IHM ---
        self.train_start_date_var = tk.StringVar(value=self.ai_logic.start_date_train)
        self.train_end_date_var = tk.StringVar(value=self.ai_logic.end_date_train)
        self.output_directory = None
        self.log_buffer = []
        self.capture_log = False
        
        # --- NOUVEAU v14.0 : Variables WFA ---
        self.wfa_window_var = tk.StringVar(value="5Y") # Fenêtre d'entraînement
        self.wfa_step_var = tk.StringVar(value="1M") # Pas d'avancement
        self.wfa_start_var = tk.StringVar(value="2019-01-01") # Date de début WFA (pour le backtest)
        self.wfa_end_var = tk.StringVar(value=pd.Timestamp.now().strftime('%Y-%m-%d')) # Date de fin WFA
        self.wfa_results_df = None # Stocke les allocations WFA
        
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
        subtitle_text = "Optimized Risk & Investment Opportunity Navigator (Hybrid Edition)"
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
        mode_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 2 : Opérations IA - Gestion du Modèle LIVE", padding=10) # TITRE MODIFIÉ
        mode_frame.pack(fill=tk.X, pady=5)
        
        # --- NOUVEAU: Cadre pour l'Entraînement Standard ---
        train_panel = ttk.LabelFrame(mode_frame, text="1. Entraînement Standard (Modèle LIVE global)", padding=5) # Changé en LabelFrame pour plus de clarté
        train_panel.pack(fill=tk.X, pady=5, padx=5)
        
        train_expl = (
            "Entraîne un NOUVEAU modèle XGBoost sur la période historique complète.\n"
            "Ce modèle remplace le fichier 'ia_model.joblib' actuel."
        )
        ttk.Label(train_panel, text=train_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)
        
        date_frame = ttk.Frame(train_panel)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Date Début (Train):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(date_frame, textvariable=self.train_start_date_var, width=12).pack(side=tk.LEFT)
        
        ttk.Label(date_frame, text="Date Fin (Train):").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(date_frame, textvariable=self.train_end_date_var, width=12).pack(side=tk.LEFT)
        
        # Bouton Entraînement Standard (Bleu)
        self.train_button = ttk.Button(train_panel, text="LANCER L'ENTRAÎNEMENT (Mise à Jour LIVE)", command=self.start_training, style="Blue.TButton") 
        self.train_button.pack(pady=10, fill=tk.X)

        # --- NOUVEAU: Cadre pour l'Analyse WFA ---
        wfa_panel = ttk.LabelFrame(mode_frame, text="2. Analyse WFA (Backtest adaptatif & Modèle LIVE adapté)", padding=5) # Nouveau LabelFrame
        wfa_panel.pack(fill=tk.X, pady=5, padx=5)
        
        wfa_expl = (
            "Lance un backtest par recalibrage glissant (WFA).\n"
            "**ATTENTION :** Le modèle 'ia_model.joblib' sera mis à jour par le DERNIER pas WFA."
        )
        ttk.Label(wfa_panel, text=wfa_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)

        # Configuration WFA
        wfa_config_frame = ttk.Frame(wfa_panel)
        wfa_config_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(wfa_config_frame, text="Fenêtre Entraînement:").pack(side=tk.LEFT)
        ttk.Entry(wfa_config_frame, textvariable=self.wfa_window_var, width=5).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(wfa_config_frame, text="Pas Avancement:").pack(side=tk.LEFT)
        ttk.Entry(wfa_config_frame, textvariable=self.wfa_step_var, width=5).pack(side=tk.LEFT, padx=(2, 10))

        # Date de début/fin WFA (pour un backtest WFA)
        wfa_date_frame = ttk.Frame(wfa_panel)
        wfa_date_frame.pack(fill=tk.X, pady=5)
        ttk.Label(wfa_date_frame, text="WFA Début:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(wfa_date_frame, textvariable=self.wfa_start_var, width=10).pack(side=tk.LEFT)
        ttk.Label(wfa_date_frame, text="WFA Fin:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(wfa_date_frame, textvariable=self.wfa_end_var, width=10).pack(side=tk.LEFT)

        # Bouton WFA (Violet)
        self.wfa_button = ttk.Button(wfa_panel, text="LANCER L'ANALYSE WFA (Mise à Jour LIVE Finale)", command=self.start_walk_forward_analysis, style="Purple.TButton") # Texte modifié
        self.wfa_button.pack(pady=10, fill=tk.X)
        
        # --- NOUVEAU: Cadre pour la Prédiction LIVE ---
        predict_frame = ttk.LabelFrame(mode_frame, text="3. Prédiction LIVE (Utilise le Modèle actuel)", padding=5)
        predict_frame.pack(fill=tk.X, pady=5, padx=5)

        predict_expl = (
            "Lance la prédiction immédiate des signaux sur les données d'aujourd'hui, \n"
            "en utilisant le dernier modèle 'ia_model.joblib' sauvegardé (via Entraînement ou WFA)."
        )
        ttk.Label(predict_frame, text=predict_expl, justify=tk.LEFT).pack(fill=tk.X, pady=5)

        self.clear_output_dir = tk.BooleanVar(value=False)
        clear_dir_check = ttk.Checkbutton(predict_frame, 
                                          text="Nettoyer le répertoire d'export avant la prédiction", 
                                          variable=self.clear_output_dir,
                                          style="Danger.TCheckbutton")
        clear_dir_check.pack(pady=5, anchor='w')
        
        # Bouton Prédiction LIVE (Vert)
        self.predict_button = ttk.Button(predict_frame, text="LANCER LA PRÉDICTION LIVE", command=self.start_prediction, style="Green.TButton")
        self.predict_button.pack(pady=10, fill=tk.X)


        # --- ÉTAPE 3 : Console de Log ---
        log_frame = ttk.LabelFrame(main_frame, text="ÉTAPE 3 : Console de Log & Exports", padding=10) # Titre modifié
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        sys.stdout = TextRedirector(self.log_widget, "stdout", self)
        sys.stderr = TextRedirector(self.log_widget, "stderr", self)
        
        # Cadre pour les boutons d'export
        export_buttons_frame = ttk.Frame(main_frame)
        export_buttons_frame.pack(fill=tk.X, pady=5)
        
        self.export_button = ttk.Button(export_buttons_frame, text="Définir le Répertoire d'Export (Rapports)", command=self.select_output_directory)
        self.export_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        # Bouton Export WFA déplacé ici
        self.export_wfa_button = ttk.Button(export_buttons_frame, text="Exporter les Résultats WFA (CSV)", command=self.export_wfa_results, style="Purple.TButton", state=tk.DISABLED) # Ajout du style
        self.export_wfa_button.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))
        
        
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
        
        self.connect_button = ttk.Button(status_frame, text="Tester la Connexion (Paper)", command=self.connect_to_alpaca, style="Blue.TButton") # Style adapté
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        # NOUVEAU v12.15 : Bouton "Afficher Positions"
        self.show_positions_button = ttk.Button(status_frame, text="Afficher les Positions Actuelles", command=self.start_fetch_positions, style="Blue.TButton") # Style adapté
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
        
        # --- MODIFIÉ v15.0 : Élargissement de la colonne Symbole ---
        self.positions_tree.column("Symbole", width=220, anchor=tk.W) # Élargi pour contenir le Nom
        # -----------------------------------------------------------
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
        """Lance l'entraînement standard dans un thread séparé."""
        assets_to_process, _ = self.get_selected_assets() 
        if not assets_to_process:
            return 

        start_date = self.train_start_date_var.get()
        end_date = self.train_end_date_var.get()
        
        output_directory = self.output_directory
        if not output_directory:
            messagebox.showwarning("Export non défini", "Le log d'entraînement ne sera pas sauvegardé.\n\nVeuillez d'abord 'Définir le Répertoire d'Export' (Étape 3) si vous souhaitez un rapport.")
            
        if not (start_date and end_date):
            messagebox.showerror("Erreur", "Les dates de début et de fin d'entraînement sont requises.")
            return

        self.log_widget.delete('1.0', tk.END) 
        print(f"Initialisation du thread d'entraînement (Dates: {start_date} à {end_date})...") 
        
        self.log_buffer = []
        self.capture_log = True
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED, text="Entraînement en cours...")
        self.predict_button.config(state=tk.DISABLED)
        self.sync_button.config(state=tk.DISABLED) 
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        self.wfa_button.config(state=tk.DISABLED) # NOUVEAU v14.0
        self.export_wfa_button.config(state=tk.DISABLED)
        
        self.train_thread = threading.Thread(target=self.run_training_task, args=(assets_to_process, start_date, end_date, output_directory))
        self.train_thread.daemon = True
        self.train_thread.start()

    # --- MODIFIÉ v12.15 / REQUÊTE UTILISATEUR v13.06 ---
    def run_training_task(self, assets_to_process, start_date, end_date, output_directory): 
        """Tâche exécutée par le thread d'entraînement."""
        captured_log = ""
        
        try:
            self.ai_logic.run_training_mode(assets_to_process, start_date, end_date) 
            
            print("\n[Rapport] Entraînement terminé. Sauvegarde du log...")
            self.capture_log = False
            captured_log = "".join(self.log_buffer)
            self.log_buffer = []
            
            if output_directory:
                self.generate_training_report(captured_log, output_directory)
            else:
                print("[Rapport] Aucun répertoire d'export défini. Log non sauvegardé sur disque.")
            
        except Exception as e:
            print(f"[ERREUR FATALE DU THREAD] : {e}\nTrace: {traceback.format_exc()}")
            self.capture_log = False
            self.log_buffer = []
        finally:
            print("Thread d'entraînement terminé. Réactivation des boutons.")
            
            def re_enable():
                self.train_button.config(state=tk.NORMAL, text="LANCER L'ENTRAÎNEMENT (Mise à Jour LIVE)")
                self.predict_button.config(state=tk.NORMAL)
                self.connect_button.config(state=tk.NORMAL)
                self.wfa_button.config(state=tk.NORMAL) # NOUVEAU v14.0
                if self.api: 
                    self.sync_button.config(state=tk.NORMAL)
                    self.show_positions_button.config(state=tk.NORMAL)
            self.root.after_idle(re_enable)

    # --- NOUVEAU v14.0 : Fonctions WFA (Walk Forward Analysis) ---
    def start_walk_forward_analysis(self):
        """Lance l'analyse WFA dans un thread séparé pour le backtest."""
        assets_to_process, ticker_to_name_map = self.get_selected_assets() 
        if not assets_to_process:
            return 

        try:
            wfa_start_date = pd.to_datetime(self.wfa_start_var.get())
            wfa_end_date = pd.to_datetime(self.wfa_end_var.get())
            window_size = self.wfa_window_var.get()
            step_size = self.wfa_step_var.get()
        except ValueError as e:
            messagebox.showerror("Erreur Date/Format", f"Format de date ou de période invalide: {e}\nExemples valides : 5Y, 6M, 30D.")
            return

        if wfa_start_date >= wfa_end_date:
            messagebox.showerror("Erreur Période", "La date de début WFA doit être antérieure à la date de fin WFA.")
            return
            
        output_directory = self.output_directory
        if not output_directory:
            messagebox.showwarning("Export non défini", "Les résultats WFA ne seront pas sauvegardés.\n\nVeuillez d'abord 'Définir le Répertoire d'Export' (Étape 3).")

        self.log_widget.delete('1.0', tk.END) 
        print(f"Initialisation du thread WFA...")
        print(f"Période WFA : {wfa_start_date.strftime('%Y-%m-%d')} à {wfa_end_date.strftime('%Y-%m-%d')}")
        print(f"Fenêtre d'entraînement : {window_size} | Pas d'avancement : {step_size}") 
        
        self.log_buffer = []
        self.capture_log = True
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)
        self.sync_button.config(state=tk.DISABLED) 
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        self.wfa_button.config(state=tk.DISABLED, text="WFA en cours...")
        self.export_wfa_button.config(state=tk.DISABLED)
        
        self.wfa_thread = threading.Thread(target=self.run_wfa_task, args=(
            assets_to_process, ticker_to_name_map, wfa_start_date, wfa_end_date, window_size, step_size, output_directory
        ))
        self.wfa_thread.daemon = True
        self.wfa_thread.start()

    def run_wfa_task(self, assets_to_process, ticker_to_name_map, wfa_start_date, wfa_end_date, window_size, step_size, output_directory):
        """Logique de la boucle Walk Forward Analysis."""
        
        # Initialiser la première fenêtre (le début de l'entraînement)
        current_train_start = wfa_start_date - pd.DateOffset(years=5) # Décalage initial
        current_train_end = wfa_start_date
        
        all_allocations = {} # Pour stocker {date: allocations}
        wfa_results_list = [] # Liste pour le DataFrame final

        total_start_time = time()
        step_count = 0
        
        # Le début de la fenêtre d'avancement (prediction)
        current_prediction_date = wfa_start_date
        
        captured_log = "" # Variable pour stocker le log
        
        try:
            # Boucle WFA : Avancer tant que la date d'avancement est inférieure à la date de fin
            while current_prediction_date <= wfa_end_date:
                step_count += 1
                
                # 1. Définir les bornes de la fenêtre d'entraînement et de prédiction (recalibrage)
                
                # --- CORRECTION DE L'ERREUR VALUVALUE ---
                train_end_date = current_prediction_date.strftime('%Y-%m-%d')
                
                try:
                    window_offset = pd.tseries.frequencies.to_offset(window_size)
                    step_offset = pd.tseries.frequencies.to_offset(step_size)
                except ValueError as e:
                    # Gérer l'erreur si l'utilisateur a entré une chaîne non valide (ex: '5X')
                    print(f"[ERREUR Période WFA] Format d'offset invalide : {e}")
                    raise TypeError(f"Format d'offset invalide : {e}")
                
                # Appliquer l'offset (soustraction)
                train_start_date = (current_prediction_date - window_offset).strftime('%Y-%m-%d')
                
                print("\n" + "~"*40)
                print(f"--- ÉTAPE WFA #{step_count} : RECALIBRAGE ---")
                print(f"Fenêtre Train : {train_start_date} à {train_end_date}")

                # 2. Entraînement (Recalibrage du modèle)
                # On réutilise run_training_mode, qui surcharge les fichiers joblib
                # Ce modèle sera utilisé pour prédire sur la date actuelle.
                success, model_trained = self.ai_logic.run_training_mode(
                    assets_to_process, train_start_date, train_end_date
                )
                
                if not success:
                    print(f"--- ÉCHEC de l'Entraînement WFA à l'étape #{step_count}. Avancement ignoré. ---")
                    # Avancer la fenêtre même en cas d'échec
                    
                    # Correction 2: Utiliser l'offset pré-calculé pour l'avancement
                    current_prediction_date += step_offset
                    continue

                # 3. Prédiction (Utilise le modèle nouvellement entraîné sur la date de fin de training)
                print(f"Fenêtre Predict: {train_end_date}")
                
                # Le mode prédiction est appelé SANS use_live_data, 
                # en utilisant train_end_date comme date de prédiction
                results_df, alloc_series, prediction_date_str, _, final_allocations = self.ai_logic.run_prediction_mode(
                    assets_to_process, {}, ticker_to_name_map, use_live_data=False,
                    custom_start_date=train_start_date, custom_end_date=train_end_date
                )
                
                if results_df is not None:
                    # Enregistrer les résultats
                    print(f"Prédiction du {prediction_date_str} terminée. Enregistrement des allocations...")
                    
                    # Convertir les allocations en format simple {ticker: poids}
                    for ticker, weight in final_allocations.items():
                        if ticker != 'CASH':
                            # Enregistrement {Date, Ticker, Poids, Probabilité}
                            # --- MODIFICATION: AJOUT DE regex=False ---
                            wfa_results_list.append({
                                'Date': prediction_date_str,
                                'Ticker': ticker,
                                'Allocation': weight,
                                'Probabilite_Hausse': results_df.loc[results_df.index.str.contains(f'({ticker})', regex=False)].iloc[0] if f"{ticker_to_name_map.get(ticker, ticker)} ({ticker})" in results_df.index else np.nan
                            })
                        elif weight > 0:
                             # Enregistrer le cash
                             wfa_results_list.append({
                                'Date': prediction_date_str,
                                'Ticker': 'CASH',
                                'Allocation': weight,
                                'Probabilite_Hausse': 0.0
                            })
                            
                else:
                    print(f"--- ÉCHEC de la Prédiction WFA à l'étape #{step_count}. ---")
                
                # 4. Avancer la fenêtre d'un pas
                # Correction 3: Utiliser l'offset pré-calculé pour l'avancement
                current_prediction_date += step_offset
                
            # 5. Fin de la boucle
            total_time = time() - total_start_time
            print("\n" + "="*50)
            print(f"--- ANALYSE WFA TERMINÉE ---")
            print(f"Nombre de recalibrages : {step_count}")
            print(f"Temps total : {total_time:.2f} secondes.")
            print("="*50 + "\n")
            
            # --- SAUVEGARDE DU LOG COMPLET DE LA WFA ---
            self.capture_log = False
            captured_log = "".join(self.log_buffer)
            self.log_buffer = [] # Vider le buffer après capture
            
            if output_directory:
                self.generate_wfa_report_log(captured_log, output_directory)
            else:
                print("[Rapport] Aucun répertoire d'export défini. Log WFA non sauvegardé sur disque.")
            
            # Stocker les résultats finaux
            self.wfa_results_df = pd.DataFrame(wfa_results_list)
            
            # Sauvegarder immédiatement si le répertoire est défini
            if output_directory and self.wfa_results_df is not None:
                self.export_wfa_results(output_directory=output_directory)
            
        except Exception as e:
            print(f"[ERREUR FATALE DU THREAD WFA] : {e}\nTrace: {traceback.format_exc()}")
            self.capture_log = False
            self.log_buffer = []
        finally:
            print("Thread WFA terminé. Réactivation des boutons.")
            
            def re_enable_wfa():
                self.train_button.config(state=tk.NORMAL)
                self.predict_button.config(state=tk.NORMAL)
                self.connect_button.config(state=tk.NORMAL)
                self.wfa_button.config(state=tk.NORMAL, text="LANCER L'ANALYSE WFA (Mise à Jour LIVE Finale)")
                if self.wfa_results_df is not None and not self.wfa_results_df.empty:
                    self.export_wfa_button.config(state=tk.NORMAL)
                if self.api: 
                    self.sync_button.config(state=tk.NORMAL)
                    self.show_positions_button.config(state=tk.NORMAL)
            self.root.after_idle(re_enable_wfa)

    # --- NOUVEAU v14.3 : Fonction de rapport de log WFA ---
    def generate_wfa_report_log(self, captured_log, output_directory):
        """Génère un simple rapport .txt avec le log complet de la WFA."""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            txt_filename = os.path.join(output_directory, f"report_WFA_LOG_{timestamp}.txt")
            print(f"        -> Création du rapport TXT de Log WFA : {txt_filename}")
            
            report_string = f"--- Journal d'Événements (Log) de l'Analyse WFA ---\n"
            report_string += f"--- Généré le: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            report_string += "="*70 + "\n\n"
            report_string += captured_log 
            report_string += "\n" + "="*70 + "\n"
            report_string += "--- Fin du Journal d'Événements WFA ---\n"
            
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(report_string)
            
            print(f"[Rapport] Fichier log WFA sauvegardé avec succès dans {output_directory}")

        except Exception as e:
            print(f"[ERREUR Rapport WFA] Impossible de générer le rapport log : {e}\nTrace: {traceback.format_exc()}")
            messagebox.showerror("Erreur d'Export (Log WFA)", f"Impossible de sauvegarder le rapport log WFA :\n{e}")
    # --- FIN NOUVEAU v14.3 ---

    # --- NOUVEAU v14.0 : Export WFA ---
    def export_wfa_results(self, output_directory=None):
        """Exporte le DataFrame des allocations WFA en CSV."""
        if self.wfa_results_df is None or self.wfa_results_df.empty:
            messagebox.showwarning("Avertissement Export", "Aucun résultat WFA à exporter. Veuillez lancer l'analyse WFA d'abord.")
            return

        if output_directory is None:
            output_directory = self.output_directory
        
        if not output_directory:
            messagebox.showerror("Erreur Export", "Veuillez définir le répertoire d'export à l'Étape 3.")
            return
            
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = os.path.join(output_directory, f"report_WFA_Allocations_{timestamp}.csv")
            
            self.wfa_results_df.to_csv(csv_filename, index=False)
            print(f"[WFA Export] Succès : Résultats WFA exportés dans {csv_filename}")
            messagebox.showinfo("Export WFA", f"Résultats WFA exportés avec succès dans :\n{csv_filename}")

        except Exception as e:
            print(f"[ERREUR Export WFA] Impossible d'exporter les résultats : {e}\nTrace: {traceback.format_exc()}")
            messagebox.showerror("Erreur d'Export WFA", f"Impossible d'exporter les résultats WFA :\n{e}")
    # --- FIN NOUVEAU v14.0 ---

    # MODIFIÉ v12.15 : Gestion de l'état des boutons
    def start_prediction(self):
        """Lance la prédiction Live dans un un thread séparé."""
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
        
        print("Initialisation du thread de prédiction Live...") 
        
        # Geler tous les boutons
        self.train_button.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED, text="Prédiction en cours...")
        self.sync_button.config(state=tk.DISABLED) 
        self.show_positions_button.config(state=tk.DISABLED) 
        self.connect_button.config(state=tk.DISABLED) 
        self.wfa_button.config(state=tk.DISABLED) # NOUVEAU v14.0
        self.export_wfa_button.config(state=tk.DISABLED)
        
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
            # Mode Live normal (use_live_data=True)
            results_df, alloc_series, last_date, full_indicators_history, final_allocations = self.ai_logic.run_prediction_mode(
                assets_to_process, display_prefs, ticker_to_name_map, use_live_data=True
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
                
                # --- MODIFIÉ v14.8 : Passage de l'aperçu des ordres au rapport ---
                self.generate_report(
                    results_df, alloc_series, last_date, output_directory, 
                    captured_log, full_indicators_history, clear_dir_pref, ticker_to_name_map,
                    trade_preview_list=trade_preview_list # Ajouté ici
                )
                # --- Fin MODIFIÉ v14.8 ---
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
                self.predict_button.config(state=tk.NORMAL, text="LANCER LA PRÉDICTION LIVE")
                self.connect_button.config(state=tk.NORMAL)
                self.wfa_button.config(state=tk.NORMAL) # NOUVEAU v14.0
                if self.wfa_results_df is not None and not self.wfa_results_df.empty:
                    self.export_wfa_button.config(state=tk.NORMAL)
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
            
            # --- MODIFICATION DEMANDÉE : Affichage détaillé ---
            cash = float(account.cash)
            equity = float(account.equity)
            invested = equity - cash # Portefeuille acheté (non vendu)
            initial_cash = 100000.0 # Valeur standard pour le Paper Trading Alpaca
            
            # Calcul du rapport de performance
            performance = equity / initial_cash
            
            status_text = (f"Statut : Connecté | CASH: ${cash:,.2f} | "
                           f"Investi: ${invested:,.2f} | Perf: {performance:.2%}")
            
            self.alpaca_status_label.config(text=status_text, foreground="green")
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
        account_data = None # Pour le status
        try:
            # --- MODIF : Récupérer aussi les infos du compte pour mettre à jour le header ---
            account = self.api.get_account()
            account_data = {
                'cash': float(account.cash),
                'equity': float(account.equity)
            }
            
            positions = self.api.list_positions()
            for p in positions:
                # --- MODIF v15.0 : Récupération du Nom de l'actif ---
                try:
                    asset = self.api.get_asset(p.symbol)
                    symbol_display = f"{asset.name} ({p.symbol})"
                except Exception:
                    symbol_display = p.symbol
                # ----------------------------------------------------

                positions_data.append({
                    "symbol": symbol_display, # Utilisation du nom combiné
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_intraday_pl": float(p.unrealized_intraday_pl),
                    "unrealized_pl": float(p.unrealized_pl)
                })
            
            # Mettre à jour l'IHM dans le thread principal
            self.root.after_idle(self.update_positions_treeview, positions_data, account_data)
            
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

    def update_positions_treeview(self, positions_data, account_data=None):
        """Met à jour le Treeview avec les données (appelé depuis le thread principal)."""
        
        # --- MODIFICATION DEMANDÉE : Mise à jour du label de statut ---
        if account_data:
            cash = account_data['cash']
            equity = account_data['equity']
            invested = equity - cash
            initial_cash = 100000.0 # Standard Paper Trading
            performance = equity / initial_cash
            
            status_text = (f"Statut : Connecté | CASH: ${cash:,.2f} | "
                           f"Investi: ${invested:,.2f} | Perf: {performance:.2%}")
            
            self.alpaca_status_label.config(text=status_text, foreground="green")

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
            target_alpaca_symbols = {s.split('.')[0] for s in target_allocations.keys() if s != 'CASH'}
            
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
            messagebox.showerror("Erreur", "Veuillez d'abord lancer une 'Prédiction Live' pour calculer les allocations cibles.")
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
        self.wfa_button.config(state=tk.DISABLED) # NOUVEAU v14.0
        self.export_wfa_button.config(state=tk.DISABLED)

        
        # Démarrer le thread
        self.sync_thread = threading.Thread(target=self.run_alpaca_sync_task)
        self.sync_thread.daemon = True
        self.sync_thread.start()

  # ... existing code ...
    # --- MODIFIÉ v12.15 / v13.05 (CORRECTIF TYPE & LIQUIDATION) : Tâche de Sync Alpaca ---
    def run_alpaca_sync_task(self):
        """Exécute la logique de rebalancement du portefeuille."""
        print("[Alpaca] Démarrage de la synchronisation du portefeuille...")
        
        # --- Enregistrer l'aperçu avant exécution ---
        try:
            print("[Alpaca] Génération de l'aperçu pour l'enregistrement...")
            trade_preview_for_log = self.get_trade_preview_list()
            self.log_trade_execution(trade_preview_for_log)
        except Exception as e:
            print(f"[ERREUR Log] Impossible de générer ou d'enregistrer l'aperçu des trades : {e}")
        
        try:
            # 1. Obtenir l'état actuel du compte
            account = self.api.get_account()
            equity = float(account.equity)
            print(f"[Alpaca] Equity total du compte : ${equity:,.2f}")
            
            # 2. Obtenir les positions actuelles
            positions = self.api.list_positions()
            # Map pour la valeur (calcul des diffs)
            current_positions_map = {p.symbol: float(p.market_value) for p in positions}
            # Map pour la quantité (pour les liquidations propres)
            current_positions_qty_map = {p.symbol: float(p.qty) for p in positions}
            
            print(f"[Alpaca] Positions actuelles : {current_positions_map}")
            
            # 3. Obtenir les allocations cibles (map de tickers)
            target_allocations = self.last_allocations
            print(f"[Alpaca] Allocations cibles (IA) : {target_allocations}")
            
            # 4. Créer la liste de tous les symboles (actuels + cibles)
            current_alpaca_symbols = set(current_positions_map.keys())
            target_alpaca_symbols = {s.split('.')[0] for s in target_allocations.keys() if s != 'CASH'}
            symbols_to_liquidate_alpaca = current_alpaca_symbols - target_alpaca_symbols
            all_symbols_to_process_yf = set(target_allocations.keys())
            
            # 5. Annuler tous les ordres ouverts pour éviter les conflits
            print("[Alpaca] Annulation de tous les ordres ouverts...")
            self.api.cancel_all_orders()
            
            # 6. Logique de rebalancement
            print("[Alpaca] Calcul et exécution du rebalancement...")
            
            # 6a. Traiter les cibles de l'IA (Achats et Ventes partielles)
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
                    # ACHAT
                    # CORRECTION v16.1 : Formatage strict en string 2 décimales pour éviter l'erreur Alpaca 'notional'
                    amount_to_buy = "{:.2f}".format(diff_value)
                    
                    print(f"[Alpaca] ORDRE ACHAT : {alpaca_symbol}, Montant : ${amount_to_buy} (Cible: ${target_value:.2f})")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            notional=amount_to_buy, # Envoi en string formaté
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                    except Exception as e:
                        print(f"[ERREUR Ordre Achat] {alpaca_symbol}: {e}")
                        
                elif diff_value < -1:
                    # VENTE PARTIELLE (Rebalancement)
                    # CORRECTION v16.1 : Formatage strict en string 2 décimales
                    amount_to_sell = "{:.2f}".format(abs(diff_value))
                    
                    print(f"[Alpaca] ORDRE VENTE : {alpaca_symbol}, Montant : ${amount_to_sell} (Cible: ${target_value:.2f})")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            notional=amount_to_sell, # Envoi en string formaté
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                    except Exception as e:
                        print(f"[ERREUR Ordre Vente] {alpaca_symbol}: {e}")
                else:
                    # Conserver
                    print(f"[Alpaca] CONSERVER : {alpaca_symbol} (Allocation déjà correcte)")
            
            # 6b. Traiter les liquidations (Vente TOTALE des positions non voulues)
            for alpaca_symbol in symbols_to_liquidate_alpaca:
                # Pour une liquidation totale, on utilise la QUANTITÉ (qty) et non le montant ($).
                # Cela évite l'erreur "insufficient qty" si le prix bouge de 1 centime.
                qty_to_sell = current_positions_qty_map.get(alpaca_symbol, 0)
                
                if qty_to_sell > 0: 
                    # CORRECTIF JSON : On force le type float()
                    qty_to_sell = float(qty_to_sell)
                    
                    print(f"[Alpaca] ORDRE LIQUIDATION : {alpaca_symbol}, Qty : {qty_to_sell} (Non dans les cibles IA)")
                    try:
                        self.api.submit_order(
                            symbol=alpaca_symbol,
                            qty=qty_to_sell, # Utilisation de qty au lieu de notional
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
            def final_sync_updates():
                self.train_button.config(state=tk.NORMAL)
                self.predict_button.config(state=tk.NORMAL)
                self.wfa_button.config(state=tk.NORMAL)
                if self.wfa_results_df is not None and not self.wfa_results_df.empty:
                    self.export_wfa_button.config(state=tk.NORMAL)
                self.start_fetch_positions()
            
            self.root.after_idle(final_sync_updates)

    # --- NOUVELLE FONCTION (Demandée) : Enregistrement des prédictions ---
# ... existing code ...

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
    # --- MODIFIÉ V14.8 : Ajout de trade_preview_list ---
  # --- MODIFIÉ V12.18 : Logique de génération des graphiques ---
    # --- MODIFIÉ V14.8 : Ajout de trade_preview_list ---
    # --- MODIFIÉ V16.2 : Export CSV des features par actif ---
    def generate_report(self, results_df, alloc_series, last_date, output_directory, captured_log="", full_indicators_history=None, clear_directory=False, ticker_to_name_map=None, trade_preview_list=None):
        """Génère et sauvegarde un rapport TXT, des graphes PNG et des fichiers CSV de features."""
        
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
            
            # --- NOUVEAU v14.8 : Ajout de l'aperçu des ordres dans le rapport ---
            if trade_preview_list:
                report_string += "\n--- Aperçu des Ordres (Détails Achats/Ventes) ---\n"
                for line in trade_preview_list:
                    report_string += line + "\n"
            # -------------------------------------------------------------------
            
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
                print("        -> Génération des graphiques d'analyse technique et export CSV...")
                
                # --- DÉBUT MODIF v12.17 / v12.18 ---
                plt.style.use('default') # S'assurer que le style est blanc

                for asset, indicator_df in full_indicators_history.items(): 
                    
                    display_name = ticker_to_name_map.get(asset, asset)
                    safe_asset_name = display_name.replace('^', '').replace('=', '_').replace('/', '_').replace('.', '_')
                    
                    # --- NOUVEAU v16.2 : Export CSV des features pour chaque actif ---
                    csv_filename = os.path.join(output_directory, f"features_data_{safe_asset_name}_{timestamp}.csv")
                    try:
                        # Utilisation de utf-8-sig pour compatibilité Excel et ; comme séparateur
                        indicator_df.to_csv(csv_filename, sep=';', encoding='utf-8-sig')
                        print(f"                      -> Export CSV réalisé : {csv_filename}")
                    except Exception as e:
                        print(f"                      -> [!] Erreur export CSV pour {asset}: {e}")
                    # ---------------------------------------------------------------

                    indicator_filename = os.path.join(output_directory, f"graph_Analyse_Technique_{safe_asset_name}_{timestamp}.png")
                    print(f"                      -> Création du graphe pour : {display_name} ({asset})")
                    
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
                        print(f"                   [!] Erreur lors du création du graphe pour {asset}: {e}")
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