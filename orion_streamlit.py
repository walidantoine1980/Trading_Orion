import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import contextlib
import io
import configparser

# Ajouter le chemin courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer la logique principale
from Orion_Live_AI import TradingAI, TICKER_CONVERSION_MAP

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Orion Live AI (Web)", layout="wide", page_icon="🌌")

# --- Initialisation de l'état de session ---
if 'ai_instance' not in st.session_state:
    st.session_state.ai_instance = TradingAI()

def load_ini_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "API_Alpaca.ini")
    key, secret = "", ""
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        if 'alpaca' in config:
            key = config['alpaca'].get('key_id', '').strip()
            secret = config['alpaca'].get('secret_key', '').strip()
    return key, secret

if 'alpaca_config' not in st.session_state:
    default_key, default_secret = load_ini_config()
    st.session_state.alpaca_config = {"api_key": default_key, "api_secret": default_secret, "base_url": "https://paper-api.alpaca.markets"}
if 'alpaca_connected' not in st.session_state:
    st.session_state.alpaca_connected = False
if 'api_instance' not in st.session_state:
    st.session_state.api_instance = None
if 'alpaca_account' not in st.session_state:
    st.session_state.alpaca_account = None
if 'alpaca_positions' not in st.session_state:
    st.session_state.alpaca_positions = pd.DataFrame()
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ["AAPL"]
if 'trade_preview_list' not in st.session_state:
    st.session_state.trade_preview_list = []

# --- Styles CSS personnalisés ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .btn-train { background-color: #2196F3; color: white; }
    .btn-predict { background-color: #4CAF50; color: white; }
    .btn-sync { background-color: #f44336; color: white; }
    .metric-card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# === BARRE LATÉRALE (SIDEBAR) ===
# ==========================================
st.sidebar.title("🌌 ORION LIVE AI v24.0")
st.sidebar.markdown("---")

st.sidebar.subheader("🔑 Connexion Alpaca")
st.session_state.alpaca_config["api_key"] = st.sidebar.text_input("API Key ID", value=st.session_state.alpaca_config["api_key"], type="password")
st.session_state.alpaca_config["api_secret"] = st.sidebar.text_input("Secret Key", value=st.session_state.alpaca_config["api_secret"], type="password")
st.session_state.alpaca_config["base_url"] = st.sidebar.selectbox("Environnement", ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"], index=0)

if st.sidebar.button("🔌 Se connecter à Alpaca"):
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            st.session_state.alpaca_config["api_key"], 
            st.session_state.alpaca_config["api_secret"], 
            st.session_state.alpaca_config["base_url"], 
            api_version='v2'
        )
        account = api.get_account()
        st.session_state.api_instance = api
        st.session_state.alpaca_account = account
        st.session_state.alpaca_connected = True
        st.sidebar.success(f"✅ Connecté! Capital: ${float(account.equity):.2f}")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de connexion: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Univers d'Investissement")

@st.cache_data(ttl=86400)
def fetch_wiki_tickers_and_names(url, match_str, col_ticker, col_name):
    import requests
    import io
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        tables = pd.read_html(io.StringIO(r.text), match=match_str)
        df = tables[0]
        tickers = df[col_ticker].tolist()
        names = df[col_name].tolist()
        return [f"{str(t).replace('.', '-')} - {str(n)}" for t, n in zip(tickers, names)]
    except Exception:
        return []

def load_watchlists():
    import glob
    watchlists = {}
    watchlist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WatchList")
    if os.path.exists(watchlist_dir):
        for file in glob.glob(os.path.join(watchlist_dir, "*.csv")):
            name_file = os.path.basename(file).replace(".csv", "")
            try:
                df = pd.read_csv(file, sep=";")
                if 'Symbol' in df.columns and 'Name' in df.columns:
                    # Nettoyage et formatage "TICKER - NOM"
                    formatted_list = []
                    for _, row in df.dropna(subset=['Symbol']).iterrows():
                        ticker = str(row['Symbol']).split('.')[0]
                        name = str(row['Name']) if not pd.isna(row['Name']) else "Inconnu"
                        formatted_list.append(f"{ticker} - {name}")
                    watchlists[f"📂 {name_file}"] = formatted_list
            except Exception:
                pass
    return watchlists

portfolio_presets = {
    "Sélection Manuelle": [],
    "US Tech Giants": ["AAPL - Apple Inc.", "MSFT - Microsoft Corp.", "GOOGL - Alphabet Inc.", "AMZN - Amazon.com Inc.", "NVDA - NVIDIA Corp."],
    "Dow Jones 30 (Scraping)": fetch_wiki_tickers_and_names('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 'Symbol', 'Symbol', 'Company') or ["AAPL - Apple", "MSFT - Microsoft", "V - Visa", "JPM - JPMorgan Chase"],
    "S&P 500 (Scraping)": fetch_wiki_tickers_and_names('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'Symbol', 'Symbol', 'Security') or ["AAPL - Apple", "MSFT - Microsoft"],
    "Nasdaq 100 (Scraping)": fetch_wiki_tickers_and_names('https://en.wikipedia.org/wiki/Nasdaq-100', 'Ticker', 'Ticker', 'Company') or ["AAPL - Apple", "MSFT - Microsoft"],
    "Russell 2000 (ETF)": ["IWM - iShares Russell 2000 ETF"],
    "Nasdaq Composite (ETF)": ["ONEQ - Fidelity Nasdaq Composite Index ETF"],
    "Indices Futures": ["ES=F - S&P 500 Futures", "NQ=F - Nasdaq 100 Futures", "YM=F - Dow Jones Futures", "RTY=F - Russell 2000 Futures", "CL=F - Crude Oil", "GC=F - Gold"],
    "Indices Sectoriels (US)": ["XLK - Technology Select Sector SPDR", "XLV - Health Care Select Sector SPDR", "XLF - Financial Select Sector SPDR", "XLE - Energy Select Sector SPDR", "XLY - Consumer Discretionary Select Sector SPDR", "XLI - Industrial Select Sector SPDR", "XLP - Consumer Staples Select Sector SPDR", "XLU - Utilities Select Sector SPDR", "XLB - Materials Select Sector SPDR", "XLRE - Real Estate Select Sector SPDR"],
    "Forex & Crypto": ["EURUSD=X - EUR/USD", "EURGBP=X - EUR/GBP", "GBPEUR=X - GBP/EUR", "BTC-USD - Bitcoin", "ETH-USD - Ethereum"]
}

# Ajouter les watchlists dynamiques
portfolio_presets.update(load_watchlists())

preset = st.sidebar.selectbox("Portefeuilles Prédéfinis", list(portfolio_presets.keys()))

if preset != "Sélection Manuelle":
    default_assets = portfolio_presets[preset]
else:
    default_assets = ["AAPL - Apple Inc."]

# Construire la liste exhaustive de TOUS les actifs américains, matières et devises
all_us_stocks = set()
for key, items in portfolio_presets.items():
    if "Sélection Manuelle" not in key:
        all_us_stocks.update(items)

# Assurer que default_assets est dans la liste
all_us_stocks.update(default_assets)
all_options = sorted(list(all_us_stocks))

selected_assets = st.sidebar.multiselect(
    "Actifs Sélectionnés", 
    options=all_options,
    default=default_assets
)

if len(selected_assets) > 50:
    st.sidebar.error(f"⚠️ AVERTISSEMENT DE PERFORMANCE: Vous avez sélectionné {len(selected_assets)} actifs. L'exécution prendra beaucoup de temps.")

# Fonction utilitaire pour extraire le ticker pur pour YFinance
def get_tickers(assets):
    tickers = []
    for asset in assets:
        # Isoler le ticker avant le " - "
        ticker_part = asset.split(" - ")[0].strip()
        ticker = TICKER_CONVERSION_MAP.get(ticker_part, ticker_part)
        tickers.append(ticker)
    return tickers

st.session_state.selected_tickers = get_tickers(selected_assets)

# ==========================================
# === ONGLET PRINCIPAL ===
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tableau de Bord", "🔄 Walk-Forward Analysis", "🔮 Prédictions du Jour", "💼 Positions Alpaca"])

# --- TAB 1: TABLEAU DE BORD ---
with tab1:
    st.header("Vue d'ensemble du Portefeuille")
    if st.session_state.alpaca_connected and st.session_state.alpaca_account:
        acc = st.session_state.alpaca_account
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Capital Total", f"${float(acc.equity):.2f}")
        col2.metric("Cash Disponible", f"${float(acc.cash):.2f}")
        col3.metric("Pouvoir d'Achat", f"${float(acc.buying_power):.2f}")
        
        pnl = float(acc.equity) - float(acc.last_equity)
        col4.metric("PnL Jour", f"${pnl:.2f}", f"{(pnl / float(acc.last_equity) * 100):.2f}%")
        
        st.subheader("Bilan des Signaux en attente")
        if not st.session_state.trade_preview_list:
            st.info("Aucune prédiction récente n'a été générée. Allez dans l'onglet 'Prédictions du Jour'.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.trade_preview_list), use_container_width=True)
    else:
        st.warning("⚠️ Veuillez vous connecter à Alpaca dans la barre latérale pour afficher votre tableau de bord.")

# --- TAB 2: WALK-FORWARD ANALYSIS ---
with tab2:
    st.header("Analyse Walk-Forward (WFA)")
    st.markdown("Testez la robustesse de l'algorithme sur des périodes historiques avec un recalibrage progressif.")
    
    col1, col2 = st.columns(2)
    with col1:
        wfa_start_date = st.date_input("Date de début WFA", value=datetime(2022, 1, 1))
        wfa_end_date = st.date_input("Date de fin WFA", value=datetime.today())
    with col2:
        window_size = st.selectbox("Fenêtre d'entraînement (Window Size)", ["1YE", "2YE", "3YE", "5YE", "6ME"], index=0)
        step_size = st.selectbox("Pas d'avancement (Step Size)", ["1ME", "3ME", "6ME", "1YE"], index=0)
        
    if st.button("🔄 Lancer l'Analyse WFA", use_container_width=True, type="primary"):
        if not st.session_state.selected_tickers:
            st.error("Sélectionnez au moins un actif dans la barre latérale.")
        else:
            with st.status("Exécution WFA en cours...", expanded=True) as status:
                st.write(f"Analyse du {wfa_start_date.strftime('%Y-%m-%d')} au {wfa_end_date.strftime('%Y-%m-%d')}...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_text = st.empty()
                
                # Conversion des dates
                current_prediction_date = pd.to_datetime(wfa_start_date)
                end_date_pd = pd.to_datetime(wfa_end_date)
                
                try:
                    window_offset = pd.tseries.frequencies.to_offset(window_size)
                    step_offset = pd.tseries.frequencies.to_offset(step_size)
                except ValueError as e:
                    st.error(f"Format d'offset invalide : {e}")
                    st.stop()
                
                total_steps = 0
                temp_date = current_prediction_date
                while temp_date <= end_date_pd:
                    total_steps += 1
                    temp_date += step_offset
                
                step_count = 0
                wfa_results_list = []
                f = io.StringIO()
                
                reverse_map = {v: k for k, v in TICKER_CONVERSION_MAP.items()}
                import time
                total_start_time = time.time()
                
                with contextlib.redirect_stdout(f):
                    try:
                        while current_prediction_date <= end_date_pd:
                            step_count += 1
                            progress = min(step_count / max(1, total_steps), 1.0)
                            progress_bar.progress(progress)
                            
                            train_end_date = current_prediction_date.strftime('%Y-%m-%d')
                            train_start_date = (current_prediction_date - window_offset).strftime('%Y-%m-%d')
                            
                            status_text.text(f"Étape {step_count}/{total_steps} | Entraînement: {train_start_date} -> {train_end_date}")
                            print(f"\n--- ÉTAPE WFA #{step_count} : RECALIBRAGE ---")
                            
                            # 1. Entraînement
                            success, model_trained = st.session_state.ai_instance.run_training_mode(
                                st.session_state.selected_tickers, train_start_date, train_end_date
                            )
                            
                            if not success:
                                print(f"Échec entraînement à l'étape #{step_count}.")
                                current_prediction_date += step_offset
                                continue
                                
                            # 2. Prédiction Historique
                            results_df, alloc_series, prediction_date_str, _, final_allocations = st.session_state.ai_instance.run_prediction_mode(
                                st.session_state.selected_tickers, {}, reverse_map, use_live_data=False,
                                custom_start_date=train_start_date, custom_end_date=train_end_date
                            )
                            
                            if results_df is not None:
                                for ticker, weight in final_allocations.items():
                                    if ticker != 'CASH':
                                        prob = results_df.loc[results_df.index.str.contains(f'({ticker})', regex=False)].iloc[0] if f"{reverse_map.get(ticker, ticker)} ({ticker})" in results_df.index else np.nan
                                        wfa_results_list.append({'Date': prediction_date_str, 'Ticker': ticker, 'Allocation': weight, 'Probabilite_Hausse': prob})
                                    elif weight > 0:
                                        wfa_results_list.append({'Date': prediction_date_str, 'Ticker': 'CASH', 'Allocation': weight, 'Probabilite_Hausse': 0.0})
                            
                            current_prediction_date += step_offset
                            log_text.text_area("Logs WFA (Live)", f.getvalue(), height=300)
                            
                        wfa_success = True
                    except Exception as e:
                        st.error(f"Erreur critique WFA: {e}")
                        wfa_success = False
                
                total_time = time.time() - total_start_time
                if wfa_success:
                    status.update(label=f"✅ Analyse WFA terminée en {total_time:.2f}s ({step_count} recalibrages)", state="complete", expanded=False)
                    st.success("Le modèle a été mis à jour via WFA. Allez dans l'onglet 'Prédictions du Jour' pour exécuter les modèles `.joblib`.")
                    
                    if wfa_results_list:
                        wfa_df = pd.DataFrame(wfa_results_list)
                        st.subheader("Résultats Finaux (Historique des Allocations)")
                        st.dataframe(wfa_df, use_container_width=True)
                        
                        csv = wfa_df.to_csv(index=False).encode('utf-8')
                        st.download_button("💾 Télécharger les résultats en CSV", data=csv, file_name="WFA_Results.csv", mime="text/csv")


# --- TAB 3: PRÉDICTIONS DU JOUR ---
with tab3:
    st.header("🔮 Prédictions du Jour (Live)")
    st.markdown("Ce module charge le modèle XGBoost fraîchement entraîné par la WFA (fichiers `.joblib`), télécharge les 56 indicateurs financiers des dernières 24h, et génère le plan de trade de demain.")

    if st.button("🚀 GÉNÉRER LES PRÉDICTIONS & ALLOCATION", type="primary", use_container_width=True):
        if not st.session_state.selected_tickers:
            st.error("Sélectionnez au moins un actif dans la barre latérale.")
        else:
            with st.status("Traitement des données de marché...", expanded=True) as status:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        reverse_map = {v: k for k, v in TICKER_CONVERSION_MAP.items()}
                        # On appelle run_prediction_mode avec use_live_data=True
                        results_df, alloc_series, last_date, full_indicators_history, final_allocations = st.session_state.ai_instance.run_prediction_mode(
                            st.session_state.selected_tickers, {}, reverse_map, use_live_data=True
                        )
                        success = True
                    except Exception as e:
                        st.error(f"Erreur critique de prédiction: {e}")
                        success = False
                
                logs = f.getvalue()
                
                if success and results_df is not None:
                    status.update(label="✅ Modèles générés !", state="complete", expanded=False)
                    
                    # 1. Formatage Explicite des Probabilités
                    st.subheader("1. Probabilités de Hausse (56 Features Analysées)")
                    df_probs = pd.DataFrame(results_df)
                    # Convertissons la probabilité en pourcentage et créons le signal
                    def generate_signal(prob):
                        if pd.isna(prob): return "N/A"
                        val = float(prob)
                        if val > 0.6: return "ACHAT FORT"
                        elif val > 0.5: return "ACHAT"
                        return "ATTENTE"
                        
                    df_probs['Signal'] = df_probs['Probabilité_Hausse_IA_21J'].apply(generate_signal)
                    df_probs['Probabilité'] = df_probs['Probabilité_Hausse_IA_21J'].apply(lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A")
                    
                    # Utiliser map_func sans erreur pandas
                    def color_signal(val):
                        if 'ACHAT' in str(val) or (isinstance(val, str) and '%' in val and float(val.strip('%')) > 50):
                            return 'background-color: #2e7d32; color: white;'
                        return 'background-color: #c62828; color: white;'
                        
                    st.dataframe(df_probs[['Probabilité', 'Signal']].style.map(
                        color_signal,
                        subset=['Signal', 'Probabilité']
                    ), use_container_width=True)
                    
                    # 2. Formatage Explicite de l'Allocation
                    st.subheader("2. Allocation du Portefeuille Recommandée")
                    st.info("L'algorithme filtre rigoureusement et alloue du Cash uniquement aux actifs ayant > 50% de probabilité de hausse.")
                    
                    alloc_data = []
                    for ticker, weight in final_allocations.items():
                        if weight > 0:
                            alloc_data.append({"Actif": ticker, "Allocation Capital": f"{weight*100:.1f}%"})
                    
                    st.table(pd.DataFrame(alloc_data))
                    
                    # 3. Préparation pour Alpaca
                    sorted_trade_list = []
                    for ticker, weight in final_allocations.items():
                        if ticker != 'CASH' and weight > 0:
                            sorted_trade_list.append({
                                "Ticker": ticker, 
                                "Action": "ACHETER", 
                                "Allocation": f"{weight*100:.1f}%"
                            })
                    st.session_state.trade_preview_list = sorted_trade_list
                else:
                    status.update(label="❌ Erreur de génération", state="error", expanded=False)
                    st.text_area("Logs d'erreur", logs)
                    
    # --- Exécution Alpaca Post-Prédiction ---
    if st.session_state.trade_preview_list:
        st.markdown("---")
        st.subheader("3. Exécution en Direct (Alpaca)")
        st.markdown("L'IA propose de transmettre ces signaux au broker Alpaca via API.")
        
        if st.button("🔥 SYNCHRONISER AVEC ALPACA", type="primary", use_container_width=True):
            if not st.session_state.alpaca_connected:
                st.error("Vous devez être connecté à Alpaca (Barre latérale) !")
            else:
                with st.spinner("Transmission des ordres à Alpaca..."):
                    api = st.session_state.api_instance
                    for trade in st.session_state.trade_preview_list:
                        action = trade.get('Action')
                        ticker = trade.get('Ticker')
                        qty = 1 # Quantité par défaut
                        try:
                            if action == "ACHETER":
                                api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='day')
                                st.success(f"Ordre ACHAT envoyé pour {ticker}")
                            elif action == "VENDRE":
                                try:
                                    position = api.get_position(ticker)
                                    if int(position.qty) > 0:
                                        api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='day')
                                        st.success(f"Ordre VENTE envoyé pour {ticker}")
                                except:
                                    st.info(f"Pas de position sur {ticker} pour vendre.")
                        except Exception as e:
                            st.error(f"Échec de l'ordre sur {ticker}: {e}")
                    st.success("✅ Synchronisation terminée !")

# --- TAB 4: POSITIONS ALPACA ---
with tab4:
    st.header("Positions Ouvertes (Live)")
    
    # Bouton principal de rafraîchissement
    if st.button("🔄 Rafraîchir les Positions", use_container_width=True):
        if st.session_state.alpaca_connected:
            try:
                positions = st.session_state.api_instance.list_positions()
                if not positions:
                    st.info("Aucune position ouverte sur Alpaca.")
                    st.session_state.alpaca_positions_list = []
                else:
                    pos_data = []
                    for p in positions:
                        pnl_pct = float(p.unrealized_plpc) * 100
                        # --- LOGIQUE EXPERT (Money Management) ---
                        if pnl_pct > 5.0:
                            advice = "🎯 TAKE PROFIT"
                            color = "#4CAF50" # Vert
                        elif pnl_pct < -3.0:
                            advice = "🛑 STOP LOSS"
                            color = "#F44336" # Rouge
                        else:
                            advice = "⏳ HOLD"
                            color = "#FF9800" # Orange
                            
                        pos_data.append({
                            "symbol": p.symbol,
                            "qty": p.qty,
                            "avg_entry": float(p.avg_entry_price),
                            "current": float(p.current_price),
                            "market_value": float(p.market_value),
                            "pnl_usd": float(p.unrealized_pl),
                            "pnl_pct": pnl_pct,
                            "advice": advice,
                            "color": color
                        })
                    st.session_state.alpaca_positions_list = pos_data
            except Exception as e:
                st.error(f"Erreur de récupération: {e}")
        else:
            st.error("Veuillez vous connecter à Alpaca d'abord (Barre latérale).")
            
    # Affichage des cartes interactives
    if 'alpaca_positions_list' in st.session_state and st.session_state.alpaca_positions_list:
        for pos in st.session_state.alpaca_positions_list:
            with st.container():
                st.markdown(f"### {pos['symbol']} <span style='font-size:16px;color:#888;'>(Qté: {pos['qty']})</span>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Valeur", f"${pos['market_value']:.2f}", f"${pos['current']:.2f}/u")
                col2.metric("P&L ($)", f"${pos['pnl_usd']:.2f}")
                col3.metric("P&L (%)", f"{pos['pnl_pct']:.2f}%")
                col4.markdown(f"**Avis Expert:** <br><span style='color:{pos['color']}; font-weight:bold; font-size:18px;'>{pos['advice']}</span>", unsafe_allow_html=True)
                
                if st.button(f"🔴 VENDRE {pos['symbol']}", key=f"sell_btn_{pos['symbol']}"):
                    if st.session_state.alpaca_connected:
                        try:
                            st.session_state.api_instance.submit_order(
                                symbol=pos['symbol'],
                                qty=pos['qty'],
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            st.success(f"✅ Ordre de Vente (Market) envoyé pour {pos['symbol']} ! Cliquez sur Rafraîchir pour mettre à jour.")
                        except Exception as e:
                            st.error(f"Échec de l'ordre de vente: {e}")
                    else:
                        st.error("Non connecté à Alpaca.")
                st.markdown("---")
