# =====================================================
# 🧠 PORTFOLIO OPTIMIZER AI v11.1 - AJOUT SHARPE & SORTINO
#    (Code Annoté et Documenté)
# =====================================================
#
# DESCRIPTION :
# Ce logiciel télécharge des données boursières, calcule des indicateurs techniques,
# entraîne un modèle IA (XGBoost) pour prédire les mouvements (Baisse/Neutre/Hausse),
# et simule la performance d'un portefeuille basé sur ces prédictions.
#
# CHANGELOG v11.1 :
# 1. METRICS : Ajout du calcul et de l'affichage des Ratios de Sharpe et Sortino
#    dans le rapport final (Analyse du Risque Ajusté).
# 2. UX : "Silent Mode" activé (suppression logs Yahoo).
# 3. CORE : Correctifs WFA et PDF.
# =====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
from time import time
import traceback
import os
import csv
import sys
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
import requests
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import contextlib 

# Suppression des avertissements non critiques pour garder la console propre
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# -----------------------------------------------------
# 🌍 Dictionnaire de Conversion des Tickers
# -----------------------------------------------------
# Ce dictionnaire sert de "traducteur" entre les noms communs (ex: "LVMH")
# et les symboles techniques Yahoo Finance (ex: "MC.PA").
TICKER_CONVERSION_MAP = {
    # --- CORRECTION DES ERREURS DE TÉLÉCHARGEMENT ---
    "QLYS.O": "QLYS", "PYPL.O": "PYPL", "GDDY.K": "GDDY", "VSCO.K": "VSCO",
    "ADBE.O": "ADBE", "EXEL.O": "EXEL", "FTDR.O": "FTDR", "AMGN.O": "AMGN",
    "DECK.K": "DECK", "CEG": "CEG",
    
    # --- ACTIONS FR (Euronext Paris ajoute souvent .PA) ---
    "LVMH": "MC.PA", "TOTAL": "TTE.PA", "AIRBUS": "AIR.PA",
    "KERING": "KER.PA", "ENGIE": "ENGI.PA", "PEUGEOT": "STLA",
    "CREDIT AGRICOLE": "ACA.PA", "BNP PARIBAS": "BNP.PA", "SOCIETE GENERALE": "GLE.PA",
    "TOTALENERGIES": "TTE.PA", "SANOFI": "SAN.PA", "L'OREAL": "OR.PA",
    "AIR LIQUIDE": "AI.PA", "AXA": "CS.PA", "ESSILOR": "EL.PA",
    "HERMES": "RMS.PA", "SCHNEIDER": "SU.PA", "SCHNEIDER ELECTRIC": "SU.PA",
    "DANONE": "BN.PA", "MICHELIN": "ML.PA", "VINCI": "DG.PA",
    "TELEPERFORMANCE": "TEP.PA", "DASSAULT SYSTEMES": "DSY.PA",
    "STELLANTIS": "STLA",
    
    # --- ACTIONS US ---
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "AMAZON": "AMZN", "TESLA": "TSLA", "NVIDIA": "NVDA", "META": "META",
    "TEAM": "TEAM",
    
    # --- INDICES & ETF ---
    "CAC40": "^FCHI", "DAX": "^GDAXI", "S&P 500": "^GSPC", "SP500": "^GSPC",
    "NASDAQ": "^IXIC", "NASDAQ 100": "QQQ", "ETF CAC40": "C40.PA",
    "ETF S&P 500": "SPY", "AMUNDI ETF S&P 500": "500.PA",
    
    # --- FOREX & COMMODITIES ---
    "EUR/USD": "EURUSD=X", "EURUSD": "EURUSD=X", 
    "GOLD": "GC=F", "OIL": "CL=F", "BTC-USD": "BTC-USD"
}
watchlist_map = {}
YAHOO_SEARCH_CACHE = {}

# =====================================================
# SECTION 1 : LOGIQUE DE BACKTEST IA (MULTI-CLASSE)
# =====================================================

# --- NOUVELLES FONCTIONS DE MESURE DE PERFORMANCE AJUSTÉE AU RISQUE ---

def calculate_annualized_return(portfolio_values, initial_capital, days):
    """ Calcule le rendement annualisé (CAGR). """
    if days == 0 or initial_capital == 0:
        return 0.0
    years = days / 365.25
    final_value = portfolio_values[-1]
    # Handle negative returns
    if final_value <= 0:
        # If portfolio is destroyed, return a very bad, non-calculable return
        return -1.0 
    return ((final_value / initial_capital) ** (1 / years)) - 1

def calculate_sharpe_ratio(returns_series, risk_free_rate=0.03, annualization_factor=12):
    """
    Calcule le Ratio de Sharpe.
    Ratio de Sharpe = (Rendement Portefeuille - Taux Sans Risque) / Volatilité du Portefeuille
    
    Le taux sans risque de 0.03 (3%) est basé sur un taux standard pour les obligations d'État 
    de la zone Euro/US. L'annualisation est basée sur 12 périodes (rééquilibrage mensuel).
    """
    if returns_series.empty: return 0.0
    
    # Taux sans risque ajusté à la fréquence de rééquilibrage (mensuel ici)
    # R_f_monthly = (1 + R_f_annual)^(1/12) - 1
    risk_free_rate_monthly = (1 + risk_free_rate) ** (1/annualization_factor) - 1
    
    # Rendement excédentaire (excess return)
    excess_return = returns_series - risk_free_rate_monthly
    
    # Volatilité (standard deviation)
    volatility = excess_return.std()
    
    if volatility == 0: return 0.0
    
    # Annualisation (Mean and Volatility)
    annualized_excess_return = excess_return.mean() * annualization_factor
    annualized_volatility = volatility * np.sqrt(annualization_factor)
    
    return annualized_excess_return / annualized_volatility

def calculate_sortino_ratio(returns_series, risk_free_rate=0.03, annualization_factor=12):
    """
    Calcule le Ratio de Sortino (utilise seulement la volatilité négative).
    Ratio de Sortino = (Rendement Portefeuille - Taux Sans Risque) / Volatilité à la Baisse
    """
    if returns_series.empty: return 0.0

    # Taux sans risque ajusté à la fréquence de rééquilibrage (mensuel ici)
    risk_free_rate_monthly = (1 + risk_free_rate) ** (1/annualization_factor) - 1
    
    # Rendement excédentaire (excess return)
    excess_return = returns_series - risk_free_rate_monthly
    
    # Rendements négatifs (downside deviations)
    # On ne considère que les rendements *inférieurs* au taux sans risque
    downside_returns = excess_return[excess_return < 0]
    
    # Volatilité à la baisse (downside deviation)
    # Formule : racine(moyenne(rendements_négatifs ^ 2))
    downside_volatility = np.sqrt(np.mean(downside_returns**2))
    
    if downside_volatility == 0:
        # Si aucune perte par rapport au taux sans risque, le Sortino est très élevé/infini.
        # On peut retourner un très grand nombre ou le Sharpe pour la cohérence.
        return calculate_sharpe_ratio(returns_series, risk_free_rate, annualization_factor)

    # Annualisation du rendement moyen excédentaire
    annualized_excess_return = excess_return.mean() * annualization_factor
    # Annualisation de la volatilité à la baisse
    annualized_downside_volatility = downside_volatility * np.sqrt(annualization_factor)
    
    return annualized_excess_return / annualized_downside_volatility


# --- INDICATEURS TECHNIQUES ---
# Fonctions mathématiques standard pour calculer RSI, MACD, Moyennes Mobiles.

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_sma(series, period=20):
    return series.rolling(window=period).mean()

def compute_bollinger(series, window=20, num_std=2):
    sma = compute_sma(series, window)
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return sma, upper, lower

# --- CRÉATION DES FEATURES (DONNÉES D'ENTRÉE DE L'IA) ---
def create_features(asset_data, regime_df):
    """
    Transforme les prix bruts en un tableau d'indicateurs compréhensibles par l'IA.
    Définit également la 'Target' (ce que l'IA doit prédire).
    """
    df = pd.DataFrame()
    df["Close"] = asset_data["Close"]
    
    # Gestion du Volume (remplace 0 par une valeur infime pour éviter division par zéro)
    if "Volume" in asset_data.columns and not asset_data["Volume"].isnull().all():
        df["Volume"] = asset_data["Volume"]
    else:
        df["Volume"] = 0
    df["Volume"] = df["Volume"].replace(0, 1e-6)

    # Intégration du Régime de Marché (VIX) si disponible
    if not regime_df.empty:
        df = df.join(regime_df, how='inner')
        feature_names = ["RSI", "MACD", "Signal", "Momentum_1D", "SMA_20", "BB_Width", "Volume_Change", "Market_Regime"]
    else:
        feature_names = ["RSI", "MACD", "Signal", "Momentum_1D", "SMA_20", "BB_Width", "Volume_Change"]

    # Calculs Techniques
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    
    bb_sma, bb_upper, bb_lower = compute_bollinger(df["Close"])
    df["SMA_20"] = bb_sma
    df["Bollinger_Haute"] = bb_upper
    df["Bollinger_Basse"] = bb_lower
    df["BB_Width"] = (bb_upper - bb_lower) / bb_sma.replace(0, 1e-9)
    
    df["Volume_Change"] = df["Volume"].pct_change(fill_method=None)
    df["Momentum_1D"] = df["Close"].pct_change(fill_method=None)
    
    # --- LOGIQUE CIBLE MULTI-CLASSE (LE CŒUR DE L'APPRENTISSAGE) ---
    # L'IA essaie de prédire le rendement à 21 jours (1 mois ouvré)
    future_returns = df["Close"].pct_change(21).shift(-21)
    
    # Seuil de neutralité : si ça bouge de moins de 1.5%, on considère ça "Neutre"
    threshold = 0.015 

    conditions = [
        (future_returns < -threshold),       # Classe 0 : Baisse
        (future_returns.abs() <= threshold), # Classe 1 : Neutre
        (future_returns > threshold)         # Classe 2 : Hausse
    ]
    choices = [0, 1, 2]
    
    # Création de la colonne Target
    df["Target"] = np.select(conditions, choices, default=1).astype(int)

    # Nettoyage des NaNs (valeurs manquantes au début du calcul des indicateurs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    valid_feature_names = [f for f in feature_names if f in df.columns]
    X = df[valid_feature_names]
    y = df["Target"]
    
    return X, y, valid_feature_names, df

# --- LOGGER ---
# Permet d'enregistrer les sorties console dans un fichier texte
class Logger(object):
    def __init__(self, filename="debug_log.txt"):
        self.terminal = sys.__stdout__
        try:
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"--- Début du Log de Backtest IA v11.1 ({datetime.now()}) ---\n\n")
            self.logfile = open(filename, 'a', encoding='utf-8')
            print(f"[Logger] Les impressions de la console sont sauvegardées dans : {filename}")
        except Exception as e:
            print(f"[Logger Erreur] Impossible d'écrire dans {filename}: {e}")
            self.logfile = None

    def write(self, message):
        self.terminal.write(message)
        if self.logfile:
            try: self.logfile.write(message)
            except: pass

    def flush(self):
        self.terminal.flush()
        if self.logfile:
            try: self.logfile.flush()
            except: pass

    def close(self):
        if self.logfile:
            try:
                self.logfile.write("\n--- Fin du Log ---\n")
                self.logfile.close()
            except: pass

# --- FONCTIONS DE SYNTHÈSE ET PDF ---

def calculer_performance_par_action(close_prices_full, assets, split_date, end_date, ticker_to_name_map):
    """ Calcule la performance Buy & Hold de chaque actif sur la période de Test. """
    print(f"Calcul performance par action de {split_date} à {end_date}...")
    perf_dict = {}
    try:
        # Gestion de la structure MultiIndex de pandas si plusieurs actifs
        if isinstance(close_prices_full.columns, pd.MultiIndex):
            close_prices = close_prices_full["Close"]
        else:
            close_prices = close_prices_full[["Close"]]
            if len(assets) == 1:
                close_prices.columns = [assets[0]]

        test_prices = close_prices.loc[split_date:end_date]
        if test_prices.empty: return {}

        for asset in assets:
            if asset not in test_prices.columns: continue
            asset_prices = test_prices[asset].dropna()
            if len(asset_prices) < 2:
                perf_dict[asset] = 0.0
                continue
            
            start_price = asset_prices.iloc[0]
            end_price = asset_prices.iloc[-1]
            
            if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                perf_dict[asset] = 0.0
            else:
                perf_dict[asset] = (end_price / start_price) - 1
                
        return perf_dict
    except Exception as e:
        print(f"[!] Erreur dans 'calculer_performance_par_action': {e}")
        return {}

def generer_synthese_analyse(final_value_ai, final_value_bench,
                             portfolio_value_ai, portfolio_value_bench,
                             valid_rebalance_dates, portfolio_tp_sl_signals,
                             all_allocations, importances_series,
                             performance_par_action, ticker_to_name_map,
                             ai_drawdown, bench_drawdown,
                             sharpe_ai, sortino_ai, sharpe_bench, sortino_bench): # Nouveaux arguments
    """ Génère le texte récapitulatif qui sera affiché et mis dans le PDF. """
    synthese_text = []
    synthese_text.append("\n" + "="*60)
    synthese_text.append(" SYNTHÈSE ET ANALYSE AUTOMATIQUE DE LA PERFORMANCE (v1.2 Multi-Class) ")
    synthese_text.append("="*60)
    
    # 1. Comparaison IA vs Benchmark
    synthese_text.append("\n--- 1. L'IA bat-elle le Benchmark ? (Alpha) ---")
    if final_value_ai > final_value_bench:
        try:
            surperformance_pct = (final_value_ai / final_value_bench - 1) * 100
            synthese_text.append(f"[✓ BON] OUI. Surperformance de +{surperformance_pct:.2f}%.")
        except: synthese_text.append(f"[✓ BON] OUI. (Benchmark nul ou négatif).")
    else:
        try:
            sousperformance_pct = (1 - final_value_ai / final_value_bench) * 100
            synthese_text.append(f"[X MAUVAIS] NON. Sous-performance de -{sousperformance_pct:.2f}%.")
        except: synthese_text.append(f"[X MAUVAIS] NON. Sous-performance.")

    # 2. Analyse du Risque (Drawdown)
    synthese_text.append("\n--- 2. Analyse du Risque Ajusté (Sharpe & Sortino) ---")
    synthese_text.append("  a) Max Drawdown :")
    try:
        ai_max_dd = ai_drawdown.max()
        bench_max_dd = bench_drawdown.max()
        synthese_text.append(f"     IA {ai_max_dd:.1%} vs Bench {bench_max_dd:.1%}")
        if ai_max_dd < bench_max_dd:
            synthese_text.append(f"     [✓ BON] L'IA a mieux protégé le capital contre les fortes baisses.")
        else:
            synthese_text.append(f"     [~ MOYEN] L'IA a été aussi/plus volatile lors des corrections.")
    except: pass
    
    synthese_text.append("  b) Ratio de Sharpe (Rendement / Volatilité totale) :")
    synthese_text.append(f"     IA {sharpe_ai:.3f} | Benchmark {sharpe_bench:.3f}")
    synthese_text.append("     * Plus le Sharpe est élevé, plus le rendement est bon par unité de risque.")
    
    synthese_text.append("  c) Ratio de Sortino (Rendement / Volatilité à la baisse) :")
    synthese_text.append(f"     IA {sortino_ai:.3f} | Benchmark {sortino_bench:.3f}")
    synthese_text.append("     * Le Sortino ne pénalise que les mouvements négatifs. Un Sortino > Sharpe est idéal.")
    
    if sortino_ai > sharpe_ai * 1.05 and sortino_ai > sortino_bench:
        synthese_text.append(f"     [✓ BON] L'IA excelle à filtrer la volatilité négative (Sortino > Sharpe).")
    elif sortino_ai > sortino_bench:
        synthese_text.append(f"     [~ MOYEN] L'IA gère mieux le risque de baisse que le Benchmark.")
    else:
        synthese_text.append(f"     [X MAUVAIS] Le Benchmark a un meilleur rendement ajusté au risque de baisse.")

    # 3. Signaux TP/SL
    synthese_text.append("\n--- 3. Logique et Trade Management ---")
    tp_count = sum(1 for s in portfolio_tp_sl_signals if s['type'] == 'buy')
    sl_count = sum(1 for s in portfolio_tp_sl_signals if s['type'] == 'sell')
    synthese_text.append(f"  a) Signaux TP/SL (Ratio Gains/Pertes) :")
    synthese_text.append(f"     TP (Gains) : {tp_count} | SL (Pertes) : {sl_count}")
    
    # 4. Cash Management
    try:
        alloc_df_synth = pd.DataFrame(all_allocations)
        if 'CASH' in alloc_df_synth.columns:
            avg_cash = alloc_df_synth['CASH'].mean()
            max_cash = alloc_df_synth['CASH'].max()
            synthese_text.append(f"  b) Cash Moyen: {avg_cash:.1%} (Max: {max_cash:.1%})")
            if max_cash > 0.5: synthese_text.append("     [✓ BON] L'IA sait se retirer du marché (Cash > 50%).")
    except: pass
        
    # 5. Features Importantes
    synthese_text.append("\n--- 4. Diagnostic du Modèle ---")
    if importances_series is not None and not importances_series.empty:
        synthese_text.append("  - Top 3 Features :")
        for i, (idx, val) in enumerate(importances_series.sort_values(ascending=False).head(3).items()):
            synthese_text.append(f"    {i+1}. {idx} ({val:.3f})")

    # 6. Détail par actif
    synthese_text.append("\n--- 5. Performance par Actif (Test) ---")
    if performance_par_action:
        sorted_perf = sorted(performance_par_action.items(), key=lambda item: item[1], reverse=True)
        for asset_ticker, perf in sorted_perf:
            display_name = ticker_to_name_map.get(asset_ticker, asset_ticker)
            synthese_text.append(f"  - {display_name:<20} : {perf:+.2%}")
    
    return synthese_text

def get_pdf_style(line):
    """ Définit la couleur et le style du texte dans le PDF selon le contenu """
    style = {'fontsize': 10, 'fontfamily': 'monospace', 'va': 'top', 'color': 'black', 'fontweight': 'normal'}
    stripped = line.strip()
    if stripped.startswith("SYNTHÈSE") or stripped.startswith("="):
        style.update({'fontsize': 11, 'fontweight': 'bold', 'color': '#1E293B'})
    elif stripped.startswith("---"):
        style.update({'fontweight': 'bold', 'color': '#334155'})
    elif stripped.startswith("[✓ BON]"):
        style.update({'fontweight': 'bold', 'color': '#16A34A'})
    elif stripped.startswith("[X MAUVAIS]"):
        style.update({'fontweight': 'bold', 'color': '#DC2626'})
    elif stripped.startswith("[~ MOYEN]"):
        style.update({'color': '#3B82F6'})
    return style

def add_text_to_pdf(pdf_object, text_list):
    """ Ajoute le texte ligne par ligne dans le PDF avec pagination automatique """
    try:
        fig, ax = plt.subplots(figsize=(8.27, 11.69)) # Format A4
        ax.axis('off')
        y_pos = 0.95
        y_step = 0.018
        line_count = 0
        max_lines = 50
        
        for line in text_list:
            if line_count >= max_lines:
                pdf_object.savefig(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis('off')
                y_pos = 0.95
                line_count = 0
            
            style_args = get_pdf_style(line)
            x_pos = 0.05
            if line.strip().startswith("SYNTHÈSE") or line.strip().startswith("="):
                style_args['ha'] = 'center'
                x_pos = 0.5
            
            fig.text(x_pos, y_pos, line, **style_args)
            y_pos -= y_step
            line_count += 1
        
        pdf_object.savefig(fig)
        plt.close(fig)
    except: pass

# ==============================================================
# --- CŒUR DU SYSTÈME : FONCTION DE BACKTEST (MULTI-CLASSE) ---
# ==============================================================
def run_backtest_logic(assets_list, start_date_str, end_date_str, split_date_str, capital_float, 
                       ticker_to_name_map, queue, segment_label=""):
    try:
        print("\n" + "="*50)
        print(f" Lancement de la Logique de Backtest Multi-Classe {segment_label}")
        print("="*50 + "\n")
        
        # Configuration
        CURRENT_ASSETS = list(assets_list)
        start_date = start_date_str
        split_date = split_date_str
        end_date = end_date_str
        initial_capital = capital_float
        REGIME_TICKER = "^VIX" # Utilisation du VIX pour mesurer la peur sur les marchés
        rebalance_period = 'ME' # Rééquilibrage Mensuel (Month End)
        TAKE_PROFIT_PCT = 0.15  # Gain de 15% déclenche une vente
        STOP_LOSS_PCT = 0.08    # Perte de 8% déclenche une vente

        print(f"Actifs demandés : {CURRENT_ASSETS}")
        
        if queue: 
            queue.put({'status': 'progress', 'text': f'Téléchargement données {segment_label}...', 'step': 1})
        
        print(f"--- [ÉTAPE 2/7] Téléchargement des données (Silent Mode) ---")
        all_assets_to_download = CURRENT_ASSETS + [REGIME_TICKER]
        
        # Date de fin +1 pour inclure le dernier jour dans Yahoo
        end_date_plus_1 = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # --- BLACK MAGIC : MODE SILENCIEUX ---
        # Cette classe intercepte stdout/stderr pour empêcher yfinance d'afficher 
        # des barres de progression ou des erreurs rouges dans la console.
        class NullWriter(object):
            def write(self, arg): pass
            def flush(self): pass

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
        
        try:
            data_full_ohlc = yf.download(all_assets_to_download, start=start_date, end=end_date_plus_1, auto_adjust=True, progress=False)
        finally:
            # Restauration immédiate de la console
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        data_full_ohlc.ffill(inplace=True)

        # --- FILTRAGE : On garde uniquement les actifs qui ont vraiment renvoyé des données ---
        available_tickers = []
        if isinstance(data_full_ohlc.columns, pd.MultiIndex):
            available_tickers = data_full_ohlc.columns.get_level_values(1).unique().tolist()
        else:
            if not data_full_ohlc.empty:
                available_tickers = all_assets_to_download

        VALID_ASSETS = [a for a in CURRENT_ASSETS if a in available_tickers]
        
        # Double vérification : est-ce que la colonne 'Close' existe et a des données ?
        FINAL_ASSETS = []
        for asset in VALID_ASSETS:
            try:
                if isinstance(data_full_ohlc.columns, pd.MultiIndex):
                    check_data = data_full_ohlc.xs(asset, level=1, axis=1)['Close']
                else:
                    check_data = data_full_ohlc['Close']
                
                if not check_data.dropna().empty:
                    FINAL_ASSETS.append(asset)
            except: pass
        
        ASSETS = FINAL_ASSETS
        print(f"Actifs valides (avec données) : {len(ASSETS)}/{len(CURRENT_ASSETS)}")
        
        if not ASSETS:
            print("[AVERTISSEMENT] Aucun actif valide récupéré pour cette période.")
            return None

        # Préparation des données de Régime (VIX)
        try:
            if REGIME_TICKER in available_tickers:
                if isinstance(data_full_ohlc.columns, pd.MultiIndex):
                    regime_data_raw = data_full_ohlc.xs(REGIME_TICKER, level=1, axis=1)['Close'].to_frame(name=REGIME_TICKER)
                else:
                    regime_data_raw = data_full_ohlc['Close'].to_frame(name=REGIME_TICKER)

                regime_data_raw.ffill(inplace=True)
                # On prend la moyenne sur 20 jours du VIX comme indicateur de régime
                regime_feature = regime_data_raw[REGIME_TICKER].rolling(window=20).mean().to_frame(name='Market_Regime')
                regime_feature.ffill(inplace=True) 
                regime_feature.dropna(inplace=True)
                
                # On retire le VIX des données "tradables"
                if isinstance(data_full_ohlc.columns, pd.MultiIndex):
                    data_full = data_full_ohlc.drop(columns=REGIME_TICKER, level=1, errors='ignore')
                else:
                    data_full = data_full_ohlc 
            else:
                regime_feature = pd.DataFrame()
                data_full = data_full_ohlc
        except Exception as e:
            print(f"[!] Erreur VIX: {e}")
            regime_feature = pd.DataFrame() 
            data_full = data_full_ohlc

        # --- PRÉPARATION TRAIN (APPRENTISSAGE) ---
        print(f"--- [ÉTAPE 3/7] Préparation des données d'APPRENTISSAGE (Train) ---")
        features_train, targets_train = [], []
        all_feature_names = []
        
        train_data = data_full[data_full.index < split_date]
        if not regime_feature.empty:
            regime_feature_train = regime_feature[regime_feature.index < split_date]
        else:
            regime_feature_train = pd.DataFrame()

        for asset in ASSETS:
            if queue: 
                queue.put({'status': 'progress', 'text': f'Train Feat: {asset[:10]} {segment_label}', 'step': 1})
            
            try:
                if isinstance(train_data.columns, pd.MultiIndex):
                    asset_train_data = train_data.xs(asset, level=1, axis=1)
                else:
                    asset_train_data = train_data 
                
                if not asset_train_data['Close'].dropna().empty:
                    X, y, f_names, _ = create_features(asset_train_data, regime_feature_train)
                    if not X.empty:
                        features_train.append(X)
                        targets_train.append(y)
                        all_feature_names = f_names
            except Exception as e: pass 

        if not features_train:
            print("[ERREUR] Aucun échantillon d'apprentissage (Train vide).")
            return None
            
        X_total_train = pd.concat(features_train)
        y_total_train = pd.concat(targets_train)
        
        # --- ENTRAÎNEMENT DU MODÈLE XGBOOST ---
        if queue: 
            queue.put({'status': 'progress', 'text': f'Entraînement IA {segment_label}...', 'step': 1})
        print(f"--- [ÉTAPE 4/7] Entraînement XGBoost (3 Classes) ---")
        
        # Normalisation des données (StandardScaler) pour aider l'IA
        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_total_train) 

        # Configuration de l'IA Multi-Classe
        # objective='multi:softprob' -> renvoie des probabilités pour chaque classe (0, 1, 2)
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=100, # Nombre d'arbres
            learning_rate=0.1,
            max_depth=3,      # Profondeur faible pour éviter le surentraînement
            eval_metric='mlogloss',
            random_state=42,   
            enable_categorical=True 
        )
        model.fit(X_scaled_train, y_total_train) 

        # --- PRÉDICTIONS SUR LE TEST ---
        print(f"--- [ÉTAPE 5/7] Prédictions sur le Test ---")
        predictions = {} 
        indicator_dfs_test = {} 
        test_data = data_full[data_full.index >= split_date]

        if not regime_feature.empty:
            regime_feature_test = regime_feature[regime_feature.index >= split_date]
        else:
            regime_feature_test = pd.DataFrame()

        for asset in ASSETS:
            try:
                if isinstance(test_data.columns, pd.MultiIndex):
                    asset_test_data = test_data.xs(asset, level=1, axis=1)
                else:
                    asset_test_data = test_data 
                    
                if not asset_test_data['Close'].dropna().empty:
                    X_test, _, _, df_test = create_features(asset_test_data, regime_feature_test)
                    
                    if not X_test.empty:
                        # On s'assure que les colonnes sont dans le même ordre que le Train
                        X_test_ordered = X_test[all_feature_names]
                        X_scaled_test = scaler.transform(X_test_ordered)
                        
                        probas = model.predict_proba(X_scaled_test)
                        # On ne s'intéresse qu'à la probabilité de la CLASSE 2 (HAUSSE)
                        prob_achat = probas[:, 2] 
                        
                        predictions[asset] = pd.Series(prob_achat, index=X_test.index, name="IA_Probability")
                        indicator_dfs_test[asset] = df_test
                    else:
                        predictions[asset] = pd.Series(name="IA_Probability")
                else:
                    predictions[asset] = pd.Series(name="IA_Probability")
            except Exception as e:
                predictions[asset] = pd.Series(name="IA_Probability")
                
        # --- SIMULATION DU PORTEFEUILLE ---
        if queue: 
            queue.put({'status': 'progress', 'text': f'Simulation Backtest {segment_label}...', 'step': 1})
        print(f"--- [ÉTAPE 7/7] Simulation du Portefeuille ---")
        
        capital_ai = initial_capital
        portfolio_value_ai = [capital_ai]
        capital_bench = initial_capital
        portfolio_value_bench = [capital_bench]
        all_allocations = []
        portfolio_tp_sl_signals = []
        
        rebalance_period = 'ME' # Rééquilibrage Mensuel (Month End)
        rebalance_dates = test_data.resample(rebalance_period).last().index
        valid_rebalance_dates = rebalance_dates[rebalance_dates.isin(test_data.index)]
        
        if len(valid_rebalance_dates) < 2:
            print("[ERREUR] Pas assez de dates.")
            return None
            
        if isinstance(data_full.columns, pd.MultiIndex):
            close_prices_full = data_full["Close"]
        else:
            close_prices_full = data_full[["Close"]] 
            if len(ASSETS) == 1: close_prices_full.columns = [ASSETS[0]]

        # Liste pour stocker les rendements mensuels pour le calcul de Sharpe/Sortino
        monthly_returns_ai = []
        monthly_returns_bench = []

        # BOUCLE PRINCIPALE DE SIMULATION (Mois par Mois)
        for i in range(len(valid_rebalance_dates) - 1):
            current_date = valid_rebalance_dates[i]
            next_date = valid_rebalance_dates[i+1]
            
            # --- 7a. DÉCISION D'ALLOCATION IA ---
            probs = {}
            for asset in ASSETS:
                if asset in predictions and current_date in predictions[asset].index:
                    probs[asset] = predictions[asset].loc[current_date]
            
            weights_ai = {}
            cash_weight = 1.0 
            
            if probs:
                # Stratégie : On investit proportionnellement à la confiance au-dessus de 50%
                # Si proba = 0.60 -> (0.60 - 0.50) * 2 = 0.20 (Poids brut)
                raw_weights = {asset: max(0, (prob - 0.5) * 2) for asset, prob in probs.items()}
                total_raw_weight = sum(raw_weights.values())
                
                if total_raw_weight > 1.0:
                    # Si la somme dépasse 100%, on normalise
                    weights_ai = {asset: w / total_raw_weight for asset, w in raw_weights.items()}
                    cash_weight = 0.0
                elif total_raw_weight > 0:
                    # Sinon, le reste va en CASH
                    weights_ai = raw_weights
                    cash_weight = 1.0 - total_raw_weight
                else:
                    # Aucune bonne opportunité -> 100% Cash
                    cash_weight = 1.0
            
            current_allocs = weights_ai.copy()
            current_allocs['CASH'] = cash_weight
            all_allocations.append(current_allocs)

            # --- 7b. Allocation Benchmark (Equipondéré) ---
            start_prices_bench_val = close_prices_full.loc[current_date]
            tradable_assets = [a for a in ASSETS if a in start_prices_bench_val.index and pd.notna(start_prices_bench_val[a]) and start_prices_bench_val[a] > 0]
            weights_bench = {}
            if tradable_assets:
                weight = 1.0 / len(tradable_assets)
                weights_bench = {a: weight for a in tradable_assets}

            # --- 7c. SIMULATION QUOTIDIENNE (Gestion TP/SL) ---
            # On parcourt les jours entre deux rebalancements pour vérifier TP/SL
            period_prices = data_full_ohlc.loc[current_date:next_date]
            total_portfolio_return = 0.0
            sl_hit_this_month = False
            tp_hit_this_month = False
            
            for asset, initial_weight in weights_ai.items():
                if initial_weight == 0: continue
                try:
                    if isinstance(period_prices.columns, pd.MultiIndex):
                        asset_ohlc = period_prices.xs(asset, level=1, axis=1)
                    else:
                        asset_ohlc = period_prices
                        
                    entry_price = asset_ohlc['Close'].iloc[0]
                    if pd.isna(entry_price) or entry_price == 0: continue
                        
                    tp_level = entry_price * (1 + TAKE_PROFIT_PCT)
                    sl_level = entry_price * (1 - STOP_LOSS_PCT)
                    
                    asset_return = 0.0
                    position_open = True
                    
                    for j in range(1, len(asset_ohlc)):
                        day_data = asset_ohlc.iloc[j]
                        if day_data['Low'] <= sl_level:
                            asset_return = -STOP_LOSS_PCT
                            position_open = False
                            sl_hit_this_month = True
                            break 
                        elif day_data['High'] >= tp_level:
                            asset_return = TAKE_PROFIT_PCT
                            position_open = False
                            tp_hit_this_month = True
                            break
                
                    if position_open:
                        final_price = asset_ohlc['Close'].iloc[-1]
                        asset_return = (final_price - entry_price) / entry_price
                        
                    total_portfolio_return += (asset_return * initial_weight)
                    
                except Exception as e: pass

            # Gestion du Cash - le cash ne fait pas de rendement mais est là pour la valorisation
            if cash_weight > 0:
                 # Le cash est supposé être rémunéré au taux sans risque (très faible ici, on le néglige ou on l'ajoute plus tard)
                 # Pour simplifier on assume 0% de rendement sur le cash pour cette simulation journalière, il est juste "protégé"
                pass 
            
            total_portfolio_return = total_portfolio_return # + (cash_weight * 0) 

            # Enregistrement des signaux pour le graphique
            if sl_hit_this_month:
                portfolio_tp_sl_signals.append({'date': current_date, 'type': 'sell', 'value': capital_ai})
            elif tp_hit_this_month:
                portfolio_tp_sl_signals.append({'date': current_date, 'type': 'buy', 'value': capital_ai})

            # Calcul du rendement réel du portefeuille AI pour ce mois
            return_ai_period = total_portfolio_return
            capital_ai *= (1 + return_ai_period)
            portfolio_value_ai.append(capital_ai)
            monthly_returns_ai.append(return_ai_period)


            # --- 7d. Performance Benchmark ---
            try:
                start_prices = close_prices_full.loc[current_date].astype(float).replace(0, 1e-9)
                end_prices = close_prices_full.loc[next_date].astype(float)
                period_return = (end_prices - start_prices) / start_prices
                period_return = period_return.fillna(0)
                portfolio_return_bench = sum(period_return[a] * weights_bench.get(a, 0) for a in weights_bench)
                if pd.isna(portfolio_return_bench): portfolio_return_bench = 0
                
                # Calcul du rendement réel du Benchmark pour ce mois
                return_bench_period = portfolio_return_bench
                capital_bench *= (1 + return_bench_period)
            except: 
                return_bench_period = 0
                
            portfolio_value_bench.append(capital_bench)
            monthly_returns_bench.append(return_bench_period)


        # === 8. Assemblage des Résultats ===
        importances_synthese = None
        if hasattr(model, 'feature_importances_'):
            importances_synthese = pd.Series(model.feature_importances_, index=all_feature_names)
        
        performance_par_action = calculer_performance_par_action(
            data_full_ohlc, ASSETS, split_date, end_date, ticker_to_name_map
        )
        
        # Conversion des listes de rendements en Series pour les métriques
        monthly_returns_series_ai = pd.Series(monthly_returns_ai)
        monthly_returns_series_bench = pd.Series(monthly_returns_bench)

        results_dict = {
            "portfolio_value_ai": portfolio_value_ai,
            "portfolio_value_bench": portfolio_value_bench,
            "valid_rebalance_dates": valid_rebalance_dates, 
            "portfolio_tp_sl_signals": portfolio_tp_sl_signals,
            "all_allocations": all_allocations,
            "performance_par_action": performance_par_action,
            "initial_capital": initial_capital,
            "importances_series": importances_synthese,
            "X_total_train": X_total_train,
            "X_scaled_train": X_scaled_train,
            "y_total_train": y_total_train,
            "indicator_dfs_test": indicator_dfs_test,
            "predictions": predictions,
            "ASSETS": ASSETS,
            "ticker_to_name_map": ticker_to_name_map,
            "start_date": start_date_str,
            "split_date": split_date_str,
            "end_date": end_date_str,
            "monthly_returns_ai": monthly_returns_series_ai,   # NOUVEAU
            "monthly_returns_bench": monthly_returns_series_bench # NOUVEAU
        }
        return results_dict

    except Exception as e:
        print(f"\n[ERREUR CRITIQUE LOGIQUE] {e}")
        print(traceback.format_exc())
        if queue: 
            queue.put({'status': 'error', 'title': f'Erreur Critique {segment_label}', 'text': f"Erreur majeure : {e}"})
        return None

def generate_final_report(results_dict, gen_pdf, show_stock_charts, output_directory, queue, report_title="Backtest IA"):
    """ Génère les graphiques Matplotlib et le PDF final. """
    try:
        print(f"\n--- [ÉTAPE 8/13] Génération du Rapport Final pour : {report_title} ---")
        
        # Extraction des données
        portfolio_value_ai = results_dict["portfolio_value_ai"]
        portfolio_value_bench = results_dict["portfolio_value_bench"]
        valid_rebalance_dates = results_dict["valid_rebalance_dates"]
        portfolio_tp_sl_signals = results_dict["portfolio_tp_sl_signals"]
        all_allocations = results_dict["all_allocations"]
        performance_par_action = results_dict["performance_par_action"]
        initial_capital = results_dict["initial_capital"]
        importances_series = results_dict["importances_series"]
        X_scaled_train = results_dict["X_scaled_train"]
        y_total_train = results_dict["y_total_train"]
        indicator_dfs_test = results_dict["indicator_dfs_test"]
        predictions = results_dict["predictions"]
        ticker_to_name_map = results_dict["ticker_to_name_map"]
        monthly_returns_ai = results_dict["monthly_returns_ai"]
        monthly_returns_bench = results_dict["monthly_returns_bench"]
        
        final_value_ai = portfolio_value_ai[-1]
        final_value_bench = portfolio_value_bench[-1]

        # Calcul des Ratios de Sharpe et Sortino
        sharpe_ai = calculate_sharpe_ratio(monthly_returns_ai)
        sortino_ai = calculate_sortino_ratio(monthly_returns_ai)
        sharpe_bench = calculate_sharpe_ratio(monthly_returns_bench)
        sortino_bench = calculate_sortino_ratio(monthly_returns_bench)
        
        # Calcul Drawdowns
        ai_drawdown_series = pd.Series([0.0], index=[valid_rebalance_dates[0]])
        bench_drawdown_series = pd.Series([0.0], index=[valid_rebalance_dates[0]])
        try:
            dates_index = valid_rebalance_dates[:len(portfolio_value_ai)]
            if not isinstance(dates_index, pd.DatetimeIndex): dates_index = pd.to_datetime(dates_index)
            # Supprimer les doublons dans l'index (utile si WFA est mal aligné)
            dates_index = dates_index[~dates_index.duplicated()] 
            
            ai_series = pd.Series(portfolio_value_ai, index=dates_index)
            ai_cum_max = ai_series.cummax()
            ai_drawdown_series = (ai_cum_max - ai_series) / ai_cum_max.replace(0, np.nan)
            
            bench_series = pd.Series(portfolio_value_bench, index=dates_index)
            bench_cum_max = bench_series.cummax()
            bench_drawdown_series = (bench_cum_max - bench_series) / bench_cum_max.replace(0, np.nan)
        except: pass

        synthese_texte = generer_synthese_analyse(
            final_value_ai, final_value_bench,
            portfolio_value_ai, portfolio_value_bench,
            valid_rebalance_dates, portfolio_tp_sl_signals,
            all_allocations, importances_series,
            performance_par_action, ticker_to_name_map,
            ai_drawdown_series, bench_drawdown_series,
            sharpe_ai, sortino_ai, sharpe_bench, sortino_bench # Passage des nouveaux ratios
        )
        print("\n".join(synthese_texte))

        if queue: queue.put({'status': 'progress', 'text': 'Génération Graphiques...', 'step': 1})

        # Graphique 1 : Performance Comparée
        fig_perf = plt.figure(figsize=(14, 7))
        plot_dates = valid_rebalance_dates[:len(portfolio_value_ai)]
        plt.plot(plot_dates, portfolio_value_ai, label=f"IA - ${final_value_ai:,.0f}", marker='o', lw=2, color='blue', markersize=4)
        plt.plot(plot_dates, portfolio_value_bench, label=f"Benchmark - ${final_value_bench:,.0f}", marker='x', linestyle='--', lw=2, color='gray')
        
        # Ajout des marqueurs TP/SL
        if portfolio_tp_sl_signals:
            buy_signals = [s for s in portfolio_tp_sl_signals if s['type'] == 'buy']
            sell_signals = [s for s in portfolio_tp_sl_signals if s['type'] == 'sell']
            if buy_signals:
                plt.scatter([s['date'] for s in buy_signals], [s['value'] for s in buy_signals], marker='^', color='green', s=150, label='TP Touché', zorder=5)
            if sell_signals:
                plt.scatter([s['date'] for s in sell_signals], [s['value'] for s in sell_signals], marker='v', color='red', s=150, label='SL Touché', zorder=5)

        plt.title(f"Performance : {report_title}")
        plt.ylabel("Valeur ($)")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

        # Graphique 2 : Allocations (Bar Stacked)
        fig_alloc = None
        try:
            alloc_dates = valid_rebalance_dates[:-1]
            if not isinstance(alloc_dates, pd.DatetimeIndex): alloc_dates = pd.to_datetime(alloc_dates)
            alloc_df = pd.DataFrame(all_allocations, index=alloc_dates)
            alloc_df = alloc_df[~alloc_df.index.duplicated()]
            alloc_df.fillna(0, inplace=True)
            if 'CASH' in alloc_df.columns:
                cols = [c for c in alloc_df.columns if c != 'CASH'] + ['CASH']
                alloc_df = alloc_df[cols]
            alloc_df_renamed = alloc_df.rename(columns=ticker_to_name_map)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            alloc_df_renamed.plot(kind='bar', stacked=True, title=f"Allocations IA ({report_title})", ax=ax)
            ax.set_xticklabels([d.strftime('%Y-%m') for d in alloc_df_renamed.index], rotation=45)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()
            fig_alloc = fig
            plt.show(block=False)
        except Exception as e: print(f"[!] Erreur Allocations: {e}")

        # Graphique 3 : Drawdown
        fig_drawdown = None
        try:
            fig_drawdown, ax = plt.subplots(figsize=(14, 7))
            (ai_drawdown_series * 100).fillna(0).plot(ax=ax, label="Drawdown IA", color='red', lw=2)
            (bench_drawdown_series * 100).fillna(0).plot(ax=ax, label="Drawdown Benchmark", color='gray', linestyle='--', lw=2)
            ax.set_title("Évolution du Drawdown (%)")
            ax.set_ylabel("Perte (%)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}%'))
            plt.show(block=False)
        except: pass

        # Graphique 4 : PCA 3D (Visualisation des clusters de données)
        fig_pca = None
        try:
            if X_scaled_train is not None and y_total_train is not None:
                y_train_np = y_total_train.values
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled_train)
                
                fig_pca = plt.figure(figsize=(10, 8))
                ax = fig_pca.add_subplot(111, projection='3d')
                # Couleurs selon la classe : Rouge(Vente), Gris(Neutre), Vert(Achat)
                colors = {0: 'red', 1: 'gray', 2: 'green'}
                labels = {0: 'Vente', 1: 'Neutre', 2: 'Achat'}
                
                for cl in [0, 1, 2]:
                    idx = (y_train_np == cl)
                    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2],
                               c=colors[cl], label=labels[cl], alpha=0.3, s=10)
                
                ax.set_title("Analyse PCA 3D (Train)")
                ax.legend()
                plt.show(block=False)
        except: pass

        # --- GÉNÉRATION PDF ---
        pdf = None
        if gen_pdf:
            try:
                safe_title = report_title.replace(" ", "_").replace(":", "").replace("/", "-").replace("\\", "-")
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                pdf_filename = os.path.join(output_directory, f"Rapport_{safe_title}_{datetime.now():%Y%m%d_%H%M%S}.pdf")
                pdf = PdfPages(pdf_filename)
                
                if fig_perf: pdf.savefig(fig_perf)
                if fig_alloc: pdf.savefig(fig_alloc)
                if fig_drawdown: pdf.savefig(fig_drawdown)
                if fig_pca: pdf.savefig(fig_pca)
                
                add_text_to_pdf(pdf, synthese_texte)
            except Exception as e: 
                print(f"Erreur PDF: {e}")
                pdf = None

        # --- GRAPHIQUES INDIVIDUELS ---
        if show_stock_charts or gen_pdf:
            for asset, df_base in indicator_dfs_test.items():
                try:
                    df_plot = df_base.join(predictions.get(asset))
                    df_plot['IA_Probability'].fillna(method='ffill', inplace=True)
                    
                    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
                    display_name = ticker_to_name_map.get(asset, asset)
                    fig.suptitle(f"Analyse : {display_name}", fontsize=16)

                    axes[0].plot(df_plot.index, df_plot['Close'], label='Prix', color='black')
                    axes[0].plot(df_plot.index, df_plot['SMA_20'], label='SMA 20', color='blue', linestyle='--')
                    axes[0].plot(df_plot.index, df_plot['Bollinger_Haute'], color='red', linestyle=':')
                    axes[0].plot(df_plot.index, df_plot['Bollinger_Basse'], color='green', linestyle=':')
                    axes[0].legend()

                    axes[1].plot(df_plot.index, df_plot['RSI'], color='purple', label='RSI')
                    axes[1].axhline(70, color='red', linestyle='--')
                    axes[1].axhline(30, color='green', linestyle='--')

                    axes[2].plot(df_plot.index, df_plot['MACD'], label='MACD', color='blue')
                    axes[2].plot(df_plot.index, df_plot['Signal'], label='Signal', color='orange')

                    axes[3].plot(df_plot.index, df_plot['IA_Probability'], label='Proba HAUSSE (Classe 2)', color='green')
                    axes[3].axhline(0.5, color='grey', linestyle='--')
                    axes[3].set_ylim(0, 1)
                    
                    plt.tight_layout()
                    
                    if pdf: pdf.savefig(fig)
                    if show_stock_charts: plt.show(block=False)
                    plt.close(fig)
                except: pass

        if pdf:
            pdf.close()
            print(f"PDF généré : {pdf_filename}")

        plt.show()
        return {"synthese_texte": synthese_texte}

    except Exception as e:
        print(f"Erreur Rapport: {e}")
        traceback.print_exc()
        return {"synthese_texte": [f"Erreur: {e}"]}

# ==============================================================
# --- LOGIQUE WALK-FORWARD (WFA) ---
# ==============================================================
# Le WFA divise le temps en segments glissants (ex: Train sur 2 ans, Test sur 6 mois, 
# puis on décale de 3 mois et on recommence). C'est plus réaliste qu'un backtest simple.

def assemble_wfa_results(all_segment_results, initial_capital):
    """ Recolles les morceaux des segments WFA pour créer une courbe continue. """
    print("\n--- Assemblage WFA ---")
    if not all_segment_results: return None

    full_portfolio_value_ai = [initial_capital]
    full_portfolio_value_bench = [initial_capital]
    full_valid_rebalance_dates = [all_segment_results[0]['valid_rebalance_dates'][0]]
    full_portfolio_tp_sl_signals = []
    full_all_allocations = []
    all_perf_par_action = []
    full_monthly_returns_ai = [] # NOUVEAU
    full_monthly_returns_bench = [] # NOUVEAU

    last_capital_ai = initial_capital
    last_capital_bench = initial_capital

    for i, segment_results in enumerate(all_segment_results):
        segment_values_ai = segment_results['portfolio_value_ai'][1:]
        segment_values_bench = segment_results['portfolio_value_bench'][1:]
        segment_dates = segment_results['valid_rebalance_dates'][1:]
        segment_allocations = segment_results['all_allocations']
        segment_signals = segment_results['portfolio_tp_sl_signals']
        
        # Ajout des rendements mensuels pour Sharpe/Sortino
        full_monthly_returns_ai.extend(segment_results['monthly_returns_ai'].tolist())
        full_monthly_returns_bench.extend(segment_results['monthly_returns_bench'].tolist())

        # Calcul des rendements du segment pour appliquer au capital cumulé
        segment_returns_ai = pd.Series(segment_values_ai).pct_change().fillna(0).values
        segment_returns_bench = pd.Series(segment_values_bench).pct_change().fillna(0).values
        
        for j, (ret_ai, ret_bench) in enumerate(zip(segment_returns_ai, segment_returns_bench)):
            if j == 0:
                initial_seg_cap = segment_results['initial_capital']
                # On calcule le rendement du premier mois basé sur le capital réel de départ
                current_ret_ai = (segment_values_ai[0] / initial_seg_cap) - 1
                current_ret_bench = (segment_values_bench[0] / initial_seg_cap) - 1
            else:
                current_ret_ai = ret_ai
                current_ret_bench = ret_bench
                
            last_capital_ai *= (1 + current_ret_ai)
            last_capital_bench *= (1 + current_ret_bench)
            
            full_portfolio_value_ai.append(last_capital_ai)
            full_portfolio_value_bench.append(last_capital_bench)
            full_valid_rebalance_dates.append(segment_dates[j])

        full_all_allocations.extend(segment_allocations)
        
        # Ajustement des valeurs des signaux TP/SL
        for signal in segment_signals:
            try:
                # La date de signal est un index dans full_valid_rebalance_dates
                full_date_index = full_valid_rebalance_dates.index(signal['date'])
                signal['value'] = full_portfolio_value_ai[full_date_index]
                full_portfolio_tp_sl_signals.append(signal)
            except: pass

        all_perf_par_action.append(segment_results['performance_par_action'])
        
    combined_perf = pd.DataFrame(all_perf_par_action).mean().to_dict()
    last_segment = all_segment_results[-1]
    
    return {
        "portfolio_value_ai": full_portfolio_value_ai,
        "portfolio_value_bench": full_portfolio_value_bench,
        "valid_rebalance_dates": full_valid_rebalance_dates,
        "portfolio_tp_sl_signals": full_portfolio_tp_sl_signals,
        "all_allocations": full_all_allocations,
        "performance_par_action": combined_perf,
        "initial_capital": initial_capital,
        "importances_series": last_segment["importances_series"],
        "X_total_train": last_segment["X_total_train"],
        "X_scaled_train": last_segment["X_scaled_train"],
        "y_total_train": last_segment["y_total_train"],
        "indicator_dfs_test": last_segment["indicator_dfs_test"],
        "predictions": last_segment["predictions"],
        "ASSETS": last_segment["ASSETS"],
        "ticker_to_name_map": last_segment["ticker_to_name_map"],
        "start_date": all_segment_results[0]["start_date"],
        "split_date": all_segment_results[0]["split_date"],
        "end_date": all_segment_results[-1]["end_date"],
        "monthly_returns_ai": pd.Series(full_monthly_returns_ai), # NOUVEAU
        "monthly_returns_bench": pd.Series(full_monthly_returns_bench) # NOUVEAU
    }

def run_walk_forward_threaded(gui_params, wfa_params, queue):
    """ Fonction exécutée dans un Thread séparé pour gérer la boucle WFA sans figer l'IHM. """
    try:
        print("\n--- WFA START ---")
        global_start_date = pd.to_datetime(gui_params['start_date'])
        global_end_date = pd.to_datetime(gui_params['end_date'])
        opt_days = wfa_params['opt_size']
        test_days = wfa_params['test_size']
        step_days = wfa_params['step_size']

        all_segment_results = []
        current_start_date = global_start_date
        
        while True:
            train_start_date = current_start_date
            train_end_date = train_start_date + timedelta(days=opt_days)
            test_start_date = train_end_date
            test_end_date = test_start_date + timedelta(days=test_days)
            
            if test_start_date >= global_end_date: break
            if test_end_date > global_end_date: test_end_date = global_end_date

            s_train_start = train_start_date.strftime('%Y-%m-%d')
            s_test_start = test_start_date.strftime('%Y-%m-%d')
            s_test_end = test_end_date.strftime('%Y-%m-%d')
            
            segment_results = run_backtest_logic(
                assets_list=gui_params['assets_list'],
                start_date_str=s_train_start,
                end_date_str=s_test_end,
                split_date_str=s_test_start,
                capital_float=gui_params['capital_float'],
                ticker_to_name_map=gui_params['ticker_to_name_map'],
                queue=queue,
                segment_label=f"[{s_test_start}]"
            )
            
            if segment_results: all_segment_results.append(segment_results)
            else: print(f"Echec segment {s_test_start}")
                
            current_start_date = current_start_date + timedelta(days=step_days)
            if test_end_date == global_end_date: break

        if not all_segment_results:
            if queue: queue.put({'status': 'error', 'title': 'Erreur WFA', 'text': 'Aucun segment valide.'})
            return

        if queue: queue.put({'status': 'progress', 'text': 'Assemblage WFA...', 'step': 1})
        assembled_results = assemble_wfa_results(all_segment_results, gui_params['capital_float'])

        if queue: queue.put({'status': 'progress', 'text': 'Rapport WFA...', 'step': 1})
        report_results = generate_final_report(
            results_dict=assembled_results,
            gen_pdf=gui_params['gen_pdf'],
            show_stock_charts=gui_params['show_stock_charts'],
            output_directory=gui_params['output_directory'],
            queue=queue,
            report_title=f"WFA ({opt_days}/{test_days}/{step_days})"
        )
        
        if queue: queue.put({'status': 'complete', 'synthese_list': report_results['synthese_texte']})

    except Exception as e:
        print(f"Erreur WFA: {e}")
        traceback.print_exc()
        if queue: queue.put({'status': 'error', 'title': 'Erreur WFA', 'text': f"{e}"})

# =====================================================
# SECTION 2 : FONCTIONS IHM (TKINTER)
# =====================================================

def search_yahoo_finance(query):
    """ Cherche le ticker correct via l'API Yahoo si l'utilisateur entre un nom """
    if query in YAHOO_SEARCH_CACHE: return YAHOO_SEARCH_CACHE[query]
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            ticker = data['quotes'][0]['symbol']
            YAHOO_SEARCH_CACHE[query] = ticker
            return ticker
    except: pass
    return None

def convert_csv_symbol_to_yahoo(symbol):
    """ Nettoie et convertit les symboles importés depuis un fichier CSV """
    symbol_str = str(symbol).strip().upper()
    if symbol_str in TICKER_CONVERSION_MAP: return TICKER_CONVERSION_MAP[symbol_str]
    
    forex_match = re.match(r'^([A-Z]{3})/([A-Z]{3})', symbol_str)
    if forex_match: return f"{forex_match.group(1)}{forex_match.group(2)}=X"
    
    cleaned = symbol_str
    suffixes = [".PA", ".DE", ".L", ".MI", ".AS", ".SW", ".T", ".HK", ".AX", ".IS", ".O", ".K", ".N"]
    for s in suffixes:
        if symbol_str.endswith(s):
            cleaned = symbol_str[:-len(s)]
            break
            
    if cleaned in TICKER_CONVERSION_MAP: return TICKER_CONVERSION_MAP[cleaned]
    
    found = search_yahoo_finance(cleaned)
    if found: 
        TICKER_CONVERSION_MAP[symbol_str] = found
        return found
        
    return symbol_str

def detect_csv_delimiter(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f: line = f.readline()
        if line.count(';') > line.count(','): return ';'
        return ','
    except: return ','

def load_watchlist():
    global watchlist_map
    filepath = filedialog.askopenfilename(filetypes=(("CSV/Excel", "*.csv *.xlsx"),))
    if not filepath: return
    
    messagebox.showinfo("Conversion", "L'agent va convertir les tickers inconnus. Patientez.")
    try:
        watchlist_map = {}
        ticker_listbox.delete(0, tk.END)
        status_var.set("Chargement...")
        root.update_idletasks()

        if filepath.endswith('.csv'):
            delim = detect_csv_delimiter(filepath)
            df = pd.read_csv(filepath, sep=delim, encoding='utf-8-sig')
        else:
            df = pd.read_excel(filepath)
            
        df.columns = [str(c).strip().lower() for c in df.columns]
        name_col = next((c for c in df.columns if 'name' in c or 'nom' in c), df.columns[0])
        sym_col = next((c for c in df.columns if 'symbol' in c or 'ticker' in c), df.columns[1] if len(df.columns)>1 else df.columns[0])
        
        for _, row in df.iterrows():
            name = str(row[name_col])
            sym = str(row[sym_col])
            disp = name if name else sym
            if not disp or not sym: continue
            
            status_var.set(f"Conversion... {disp[:20]}")
            root.update_idletasks()
            
            yf_ticker = convert_csv_symbol_to_yahoo(sym)
            if yf_ticker and disp not in watchlist_map:
                watchlist_map[disp] = yf_ticker
                ticker_listbox.insert(tk.END, disp)
        
        status_var.set(f"Terminé. {ticker_listbox.size()} chargés.")
        
    except Exception as e:
        messagebox.showerror("Erreur", f"{e}")

def show_synthesis_popup(synthese_list):
    try:
        popup = tk.Toplevel(root)
        popup.title("Synthèse")
        popup.geometry("800x600")
        
        popup.configure(bg=STYLE_CONFIG["BG_APP"])
        
        txt = tk.Text(popup, font=("Consolas", 10), bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"], bd=0)
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        for line in synthese_list: txt.insert(tk.END, line + "\n")
    except: pass

def check_queue(queue, btn_run, status_var, progress_bar):
    """ Vérifie régulièrement si le Thread de calcul a envoyé des messages """
    try:
        msg = queue.get_nowait()
        if msg.get('max'): progress_bar['maximum'] = msg['max']
        if msg.get('step'): progress_bar.step(msg['step'])
        if msg.get('text'): status_var.set(msg['text'])
        
        if msg.get('status') == 'error':
            messagebox.showerror("Erreur", msg.get('text'))
            btn_run.config(state="normal")
            return
        if msg.get('status') == 'complete':
            status_var.set("Terminé.")
            btn_run.config(state="normal")
            if msg.get('synthese_list'): show_synthesis_popup(msg['synthese_list'])
            messagebox.showinfo("Fin", "Backtest terminé.")
            return
    except Empty: pass
    root.after(100, check_queue, queue, btn_run, status_var, progress_bar)

def run_threaded_task(gui_params, wfa_params, queue):
    """ Point d'entrée du Thread secondaire """
    original_stdout = sys.stdout
    try:
        if gui_params['save_log']:
            sys.stdout = Logger(os.path.join(gui_params['output_directory'], "log.txt"))
            
        if gui_params['mode'] == 'Simple':
            res = run_backtest_logic(
                assets_list=gui_params['assets_list'],
                start_date_str=gui_params['start_date'],
                end_date_str=gui_params['end_date'],
                split_date_str=gui_params['split_date'],
                capital_float=gui_params['capital_float'],
                ticker_to_name_map=gui_params['ticker_to_name_map'],
                queue=queue, segment_label="[Simple]"
            )
            if res:
                rep = generate_final_report(res, gui_params['gen_pdf'], gui_params['show_stock_charts'], gui_params['output_directory'], queue, "Backtest Simple")
                queue.put({'status': 'complete', 'synthese_list': rep['synthese_texte']})
        else:
            run_walk_forward_threaded(gui_params, wfa_params, queue)
            
    except Exception as e:
        print(f"Erreur Thread: {e}")
        queue.put({'status': 'error', 'text': f"{e}"})
    finally:
        if gui_params['save_log']: sys.stdout = original_stdout

def start_ai_backtest_from_gui():
    """ Callback du bouton 'Lancer' """
    try:
        selection = [ticker_listbox.get(i) for i in ticker_listbox.curselection()]
        t_map = {watchlist_map[n]: n for n in selection if n in watchlist_map}
        
        manual = [t.strip() for t in manual_ticker_entry.get().split(",") if t.strip()]
        for m in manual:
            conv = convert_csv_symbol_to_yahoo(m)
            if conv: t_map[conv] = m
            
        assets = sorted(list(set(t_map.keys())))
        if not assets: return messagebox.showerror("Erreur", "Aucun actif.")
        
        params = {
            'mode': backtest_mode_var.get(),
            'start_date': start_entry.get(),
            'end_date': end_entry.get(),
            'split_date': split_entry.get(),
            'capital_float': float(capital_entry.get()),
            'gen_pdf': gen_pdf_var.get(),
            'show_stock_charts': show_stock_charts_var.get(),
            'save_log': save_log_var.get(),
            'assets_list': assets,
            'ticker_to_name_map': t_map,
            'output_directory': None
        }
        
        if params['gen_pdf'] or params['save_log']:
            d = filedialog.askdirectory()
            if not d: return
            params['output_directory'] = d
            
        wfa = {}
        if params['mode'] == 'WFA':
            wfa = {
                'opt_size': int(wfa_opt_entry.get()),
                'test_size': int(wfa_test_entry.get()),
                'step_size': int(wfa_step_entry.get())
            }
            
        btn_run.config(state="disabled")
        q = Queue()
        threading.Thread(target=run_threaded_task, args=(params, wfa, q), daemon=True).start()
        root.after(100, check_queue, q, btn_run, status_var, progress_bar)
        
    except Exception as e:
        messagebox.showerror("Erreur", f"{e}")

# =====================================================
# INTERFACE GRAPHIQUE MODERN DARK (Tkinter)
# =====================================================

STYLE_CONFIG = {
    "BG_APP": "#121212",
    "BG_FRAME": "#1E1E1E",
    "TEXT_COLOR": "#E0E0E0",
    "TITLE_COLOR": "#BB86FC",
    "PRIMARY_COLOR": "#03DAC6",
    "GREEN_BTN": "#00C853",
    "GREEN_BTN_HOVER": "#00E676",
    "ENTRY_BG": "#2D2D2D",
    "ENTRY_FG": "#FFFFFF",
    "FONT_MAIN": ("Segoe UI", 10),
    "FONT_BOLD": ("Segoe UI", 10, "bold"),
    "FONT_TITLE": ("Segoe UI", 16, "bold"),
}

def center_window(window, width=650, height=850):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width/2) - (width/2))
    y = int((screen_height/2) - (height/2))
    window.geometry(f'{width}x{height}+{x}+{y}')

def toggle_wfa_params(*args):
    if backtest_mode_var.get() == "WFA":
        wfa_frame.pack(pady=10, padx=20, fill="x")
        split_label.grid_remove()
        split_entry.grid_remove()
        start_label.config(text="Début Global:")
        end_label.config(text="Fin Globale:")
    else:
        wfa_frame.pack_forget()
        split_label.grid()
        split_entry.grid()
        start_label.config(text="Début (Train):")
        end_label.config(text="Fin (Test):")

root = tk.Tk()
root.title("Portfolio Optimizer AI v11.1 - AJOUT SHARPE & SORTINO")
root.configure(bg=STYLE_CONFIG["BG_APP"])
center_window(root)

style = ttk.Style()
style.theme_use('clam')
style.configure("TLabel", background=STYLE_CONFIG["BG_FRAME"], foreground=STYLE_CONFIG["TEXT_COLOR"], font=STYLE_CONFIG["FONT_MAIN"])
style.configure("TCheckbutton", background=STYLE_CONFIG["BG_FRAME"], foreground=STYLE_CONFIG["TEXT_COLOR"], font=STYLE_CONFIG["FONT_MAIN"])
style.map("TCheckbutton", background=[('active', STYLE_CONFIG["BG_FRAME"])], indicatorcolor=[('selected', STYLE_CONFIG["PRIMARY_COLOR"])])
style.configure("TRadiobutton", background=STYLE_CONFIG["BG_FRAME"], foreground=STYLE_CONFIG["TEXT_COLOR"], font=STYLE_CONFIG["FONT_MAIN"])
style.map("TRadiobutton", background=[('active', STYLE_CONFIG["BG_FRAME"])], indicatorcolor=[('selected', STYLE_CONFIG["PRIMARY_COLOR"])])
style.configure("Horizontal.TProgressbar", background=STYLE_CONFIG["PRIMARY_COLOR"], troughcolor=STYLE_CONFIG["BG_FRAME"], bordercolor=STYLE_CONFIG["BG_FRAME"], lightcolor=STYLE_CONFIG["PRIMARY_COLOR"], darkcolor=STYLE_CONFIG["PRIMARY_COLOR"])

main_canvas = tk.Canvas(root, bg=STYLE_CONFIG["BG_APP"], highlightthickness=0)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
scroll_frame = tk.Frame(main_canvas, bg=STYLE_CONFIG["BG_APP"])

scroll_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
main_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=630)
main_canvas.configure(yscrollcommand=scrollbar.set)

main_canvas.pack(side="left", fill="both", expand=True, padx=10)
scrollbar.pack(side="right", fill="y")

# --- Header ---
tk.Label(scroll_frame, text="🚀 AI BACKTEST MULTI-CLASS", font=STYLE_CONFIG["FONT_TITLE"], bg=STYLE_CONFIG["BG_APP"], fg=STYLE_CONFIG["TITLE_COLOR"]).pack(pady=(20, 10))
tk.Label(scroll_frame, text="v11.1 - AJOUT SHARPE & SORTINO", font=("Segoe UI", 9), bg=STYLE_CONFIG["BG_APP"], fg="#888").pack(pady=(0, 20))

# --- 1. ACTIFS ---
f1 = tk.LabelFrame(scroll_frame, text="1. Sélection des Actifs", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["PRIMARY_COLOR"], font=STYLE_CONFIG["FONT_BOLD"], bd=0, highlightthickness=1, highlightbackground="#333")
f1.pack(fill="x", padx=10, pady=10, ipady=5)
tk.Button(f1, text="📂 Charger CSV/Excel", command=load_watchlist, bg="#333", fg="white", relief="flat", activebackground="#444", activeforeground="white").pack(pady=10, padx=10, fill="x")
ticker_listbox = tk.Listbox(f1, selectmode=tk.MULTIPLE, height=6, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], bd=0, highlightthickness=0, selectbackground=STYLE_CONFIG["PRIMARY_COLOR"])
ticker_listbox.pack(fill="x", padx=10, pady=5)
tk.Label(f1, text="Ajout manuel (séparé par virgule):", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"]).pack(anchor="w", padx=10)
manual_ticker_entry = tk.Entry(f1, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0, highlightthickness=1, highlightcolor=STYLE_CONFIG["PRIMARY_COLOR"])
manual_ticker_entry.pack(fill="x", padx=10, pady=5, ipady=3)

# --- 2. MODE ---
f2 = tk.LabelFrame(scroll_frame, text="2. Mode de Backtest", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["PRIMARY_COLOR"], font=STYLE_CONFIG["FONT_BOLD"], bd=0, highlightthickness=1, highlightbackground="#333")
f2.pack(fill="x", padx=10, pady=10, ipady=5)
backtest_mode_var = tk.StringVar(value="Simple")
backtest_mode_var.trace("w", toggle_wfa_params)
rb_frame = tk.Frame(f2, bg=STYLE_CONFIG["BG_FRAME"])
rb_frame.pack(pady=5)
ttk.Radiobutton(rb_frame, text="Backtest Simple", value="Simple", variable=backtest_mode_var).pack(side=tk.LEFT, padx=20)
ttk.Radiobutton(rb_frame, text="Walk-Forward (WFA)", value="WFA", variable=backtest_mode_var).pack(side=tk.LEFT, padx=20)

# --- 3. CONFIG ---
f3 = tk.LabelFrame(scroll_frame, text="3. Configuration", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["PRIMARY_COLOR"], font=STYLE_CONFIG["FONT_BOLD"], bd=0, highlightthickness=1, highlightbackground="#333")
f3.pack(fill="x", padx=10, pady=10, ipady=5)
grid_f3 = tk.Frame(f3, bg=STYLE_CONFIG["BG_FRAME"])
grid_f3.pack(fill="x", padx=10, pady=5)

start_label = tk.Label(grid_f3, text="Début:", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"])
start_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
start_entry = tk.Entry(grid_f3, width=12, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); start_entry.insert(0, "2018-01-01")
start_entry.grid(row=0, column=1, padx=5, pady=5)

split_label = tk.Label(grid_f3, text="Split:", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"])
split_label.grid(row=0, column=2, padx=5, pady=5, sticky="e")
split_entry = tk.Entry(grid_f3, width=12, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); split_entry.insert(0, "2023-01-01")
split_entry.grid(row=0, column=3, padx=5, pady=5)

end_label = tk.Label(grid_f3, text="Fin:", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"])
end_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
end_entry = tk.Entry(grid_f3, width=12, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); end_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
end_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(grid_f3, text="Capital:", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"]).grid(row=1, column=2, padx=5, pady=5, sticky="e")
capital_entry = tk.Entry(grid_f3, width=10, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); capital_entry.insert(0, "10000")
capital_entry.grid(row=1, column=3, padx=5, pady=5)

# --- 4. WFA SETTINGS ---
wfa_frame = tk.LabelFrame(scroll_frame, text="4. WFA Settings", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["PRIMARY_COLOR"], font=STYLE_CONFIG["FONT_BOLD"], bd=0, highlightthickness=1, highlightbackground="#333")
wr = tk.Frame(wfa_frame, bg=STYLE_CONFIG["BG_FRAME"])
wr.pack(pady=5)
tk.Label(wr, text="Opt(j):", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"]).pack(side=tk.LEFT)
wfa_opt_entry = tk.Entry(wr, width=6, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); wfa_opt_entry.insert(0, "730")
wfa_opt_entry.pack(side=tk.LEFT, padx=5)
tk.Label(wr, text="Test(j):", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"]).pack(side=tk.LEFT, padx=(10,0))
wfa_test_entry = tk.Entry(wr, width=6, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); wfa_test_entry.insert(0, "180")
wfa_test_entry.pack(side=tk.LEFT, padx=5)
tk.Label(wr, text="Step(j):", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["TEXT_COLOR"]).pack(side=tk.LEFT, padx=(10,0))
wfa_step_entry = tk.Entry(wr, width=6, bg=STYLE_CONFIG["ENTRY_BG"], fg=STYLE_CONFIG["ENTRY_FG"], insertbackground="white", bd=0); wfa_step_entry.insert(0, "90")
wfa_step_entry.pack(side=tk.LEFT, padx=5)

# --- 5. OPTIONS ---
f5 = tk.LabelFrame(scroll_frame, text="5. Options de Sortie", bg=STYLE_CONFIG["BG_FRAME"], fg=STYLE_CONFIG["PRIMARY_COLOR"], font=STYLE_CONFIG["FONT_BOLD"], bd=0, highlightthickness=1, highlightbackground="#333")
f5.pack(fill="x", padx=10, pady=10, ipady=5)
gen_pdf_var = tk.BooleanVar(value=True)
show_stock_charts_var = tk.BooleanVar(value=False)
save_log_var = tk.BooleanVar(value=True)
chk_frame = tk.Frame(f5, bg=STYLE_CONFIG["BG_FRAME"])
chk_frame.pack(pady=5)
ttk.Checkbutton(chk_frame, text="Rapport PDF", variable=gen_pdf_var).pack(side=tk.LEFT, padx=15)
ttk.Checkbutton(chk_frame, text="Graphiques Live", variable=show_stock_charts_var).pack(side=tk.LEFT, padx=15)
ttk.Checkbutton(chk_frame, text="Log Debug", variable=save_log_var).pack(side=tk.LEFT, padx=15)

# --- RUN ---
run_frame = tk.Frame(scroll_frame, bg=STYLE_CONFIG["BG_APP"])
run_frame.pack(fill="x", padx=20, pady=20)
status_var = tk.StringVar(value="Prêt.")
status_lbl = tk.Label(run_frame, textvariable=status_var, fg=STYLE_CONFIG["PRIMARY_COLOR"], bg=STYLE_CONFIG["BG_APP"], font=("Segoe UI", 9, "italic"))
status_lbl.pack(pady=5)
progress_bar = ttk.Progressbar(run_frame, length=200, mode='determinate')
progress_bar.pack(fill="x", pady=5)
btn_run = tk.Button(run_frame, text="LANCER L'ANALYSE IA", bg=STYLE_CONFIG["GREEN_BTN"], fg="white", font=("Segoe UI", 12, "bold"), relief="flat", activebackground=STYLE_CONFIG["GREEN_BTN_HOVER"], activeforeground="white", command=start_ai_backtest_from_gui, cursor="hand2")
btn_run.pack(fill="x", pady=10, ipady=5)

toggle_wfa_params()
root.mainloop()