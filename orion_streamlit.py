import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import contextlib
import io

# Ajouter le chemin courant pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer la logique principale
from Orion_Live_AI import TradingAI, TICKER_CONVERSION_MAP

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Orion Live AI (Web)", layout="wide", page_icon="🌌")

# --- Initialisation de l'état de session ---
if 'ai_instance' not in st.session_state:
    st.session_state.ai_instance = TradingAI()
import configparser

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
st.sidebar.title("🌌 ORION LIVE AI v16.0")
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
st.sidebar.subheader("⚙️ Paramètres IA")
train_start = st.sidebar.date_input("Début Entraînement", value=datetime(2020, 1, 1))
train_end = st.sidebar.date_input("Fin Entraînement", value=datetime.today())
pred_start = st.sidebar.date_input("Début Prédiction", value=datetime(2023, 1, 1))
pred_end = st.sidebar.date_input("Fin Prédiction", value=datetime.today())

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Univers d'Investissement")
portfolio_presets = {
    "Sélection Manuelle": [],
    "CAC 40 Leaders": ["LVMH", "TOTAL", "AIRBUS", "SANOFI", "L'OREAL"],
    "US Tech Giants": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "NVIDIA"]
}
preset = st.sidebar.selectbox("Portefeuilles Prédéfinis", list(portfolio_presets.keys()))

if preset != "Sélection Manuelle":
    default_assets = portfolio_presets[preset]
else:
    default_assets = ["AAPL"]

selected_assets = st.sidebar.multiselect(
    "Actifs Sélectionnés", 
    options=list(TICKER_CONVERSION_MAP.keys()) + ["BTC-USD", "GC=F", "EURUSD=X"],
    default=default_assets
)

# Fonction utilitaire pour convertir les actifs sélectionnés en tickers YFinance
def get_tickers(assets):
    tickers = []
    for asset in assets:
        ticker = TICKER_CONVERSION_MAP.get(asset, asset)
        tickers.append(ticker)
    return tickers

st.session_state.selected_tickers = get_tickers(selected_assets)

# ==========================================
# === ONGLET PRINCIPAL ===
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tableau de Bord", "🤖 Entraînement IA", "📈 Prédictions & Live Trading", "💼 Positions Alpaca"])

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
            st.info("Aucune prédiction récente n'a été générée. Allez dans l'onglet 'Prédictions & Live Trading'.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.trade_preview_list), use_container_width=True)
    else:
        st.warning("⚠️ Veuillez vous connecter à Alpaca dans la barre latérale pour afficher votre tableau de bord.")

# --- TAB 2: ENTRAÎNEMENT IA ---
with tab2:
    st.header("Laboratoire d'Entraînement de l'IA (XGBoost Hybrid)")
    st.markdown("Lancez l'entraînement du modèle sur les actions sélectionnées. Le modèle combinera les indicateurs techniques et les fondamentaux financiers (PER, Croissance CA).")
    
    if st.button("🚀 Démarrer l'Entraînement", use_container_width=True, type="primary"):
        if not st.session_state.selected_tickers:
            st.error("Sélectionnez au moins un actif dans la barre latérale.")
        else:
            with st.status("Entraînement du modèle en cours...", expanded=True) as status:
                st.write(f"Téléchargement des données de {train_start.strftime('%Y-%m-%d')} à {train_end.strftime('%Y-%m-%d')}...")
                
                # Redirection du stdout pour capturer les logs de TradingAI
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        st.session_state.ai_instance.run_training_mode(
                            st.session_state.selected_tickers, 
                            train_start.strftime("%Y-%m-%d"), 
                            train_end.strftime("%Y-%m-%d")
                        )
                        success = True
                    except Exception as e:
                        st.error(f"Erreur d'entraînement: {e}")
                        success = False
                
                # Affichage des logs
                logs = f.getvalue()
                st.text_area("Logs de l'IA", logs, height=300)
                
                if success:
                    status.update(label="✅ Entraînement terminé avec succès !", state="complete", expanded=False)
                    st.success("Le modèle a été sauvegardé localement (`ia_model_v16.joblib`).")

# --- TAB 3: PRÉDICTIONS & TRADING ---
with tab3:
    st.header("Moteur de Prédiction & Exécution")
    st.markdown("Utilisez le modèle entraîné pour générer les signaux de trading du jour sur votre Watchlist.")
    
    if st.button("🔮 Générer les Prédictions du Jour", use_container_width=True):
        if not st.session_state.selected_tickers:
            st.error("Veuillez sélectionner des actifs.")
        else:
            with st.status("Génération des prédictions...", expanded=True) as status:
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        # Mappage pour l'affichage (Ticker -> Nom)
                        reverse_map = {v: k for k, v in TICKER_CONVERSION_MAP.items()}
                        
                        # run_prediction_mode retourne 3 éléments
                        results_df, alloc_series, sorted_trade_list = st.session_state.ai_instance.run_prediction_mode(
                            st.session_state.selected_tickers,
                            display_prefs={}, # Pas de prefs spécifiques pour l'instant
                            ticker_to_name_map=reverse_map,
                            use_live_data=True,
                            custom_start_date=pred_start.strftime("%Y-%m-%d"),
                            custom_end_date=pred_end.strftime("%Y-%m-%d")
                        )
                        
                        st.session_state.trade_preview_list = sorted_trade_list
                        success = True
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction: {e}")
                        success = False
                
                logs = f.getvalue()
                if success:
                    status.update(label="✅ Prédictions générées !", state="complete", expanded=False)
                    
                    st.subheader("Résultats des Prédictions")
                    st.dataframe(results_df.style.applymap(
                        lambda val: 'background-color: #2e7d32; color: white;' if val == 1 else 'background-color: #c62828; color: white;',
                        subset=['Prédiction (1=Achat)']
                    ), use_container_width=True)
                    
                    st.subheader("Signaux Confirmés & Allocation Recommandée")
                    st.table(pd.DataFrame(st.session_state.trade_preview_list))
                else:
                    st.text_area("Logs d'erreur", logs)

    st.markdown("---")
    st.subheader("🚀 Exécution en Direct (Alpaca)")
    if st.button("🔥 SYNCHRONISER AVEC ALPACA", type="primary", use_container_width=True):
        if not st.session_state.alpaca_connected:
            st.error("Vous devez être connecté à Alpaca (Barre latérale) !")
        elif not st.session_state.trade_preview_list:
            st.warning("Générez d'abord des prédictions avant d'exécuter des ordres.")
        else:
            with st.spinner("Transmission des ordres à Alpaca..."):
                api = st.session_state.api_instance
                for trade in st.session_state.trade_preview_list:
                    action = trade.get('Action')
                    ticker = trade.get('Ticker')
                    qty = 1 # Quantité par défaut
                    
                    try:
                        # Logique simplifiée de l'ordre
                        # Remarque: Dans l'app Tkinter, la logique était plus complexe (gestion des positions existantes).
                        # Ceci est une version simplifiée pour le Streamlit.
                        if action == "ACHETER":
                            api.submit_order(
                                symbol=ticker,
                                qty=qty,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            st.success(f"Ordre ACHAT envoyé pour {ticker}")
                        elif action == "VENDRE":
                            # Vérifier si on a la position avant de vendre
                            try:
                                position = api.get_position(ticker)
                                if int(position.qty) > 0:
                                    api.submit_order(
                                        symbol=ticker,
                                        qty=qty,
                                        side='sell',
                                        type='market',
                                        time_in_force='day'
                                    )
                                    st.success(f"Ordre VENTE envoyé pour {ticker}")
                            except:
                                st.info(f"Pas de position sur {ticker} pour vendre.")
                    except Exception as e:
                        st.error(f"Échec de l'ordre sur {ticker}: {e}")
                st.success("✅ Synchronisation terminée !")

# --- TAB 4: POSITIONS ALPACA ---
with tab4:
    st.header("Positions Ouvertes (Live)")
    if st.button("🔄 Rafraîchir les Positions"):
        if st.session_state.alpaca_connected:
            try:
                positions = st.session_state.api_instance.list_positions()
                if not positions:
                    st.info("Aucune position ouverte sur Alpaca.")
                else:
                    pos_data = []
                    for p in positions:
                        pos_data.append({
                            "Actif": p.symbol,
                            "Quantité": p.qty,
                            "Prix Moyen": f"${float(p.avg_entry_price):.2f}",
                            "Prix Actuel": f"${float(p.current_price):.2f}",
                            "Valeur Marché": f"${float(p.market_value):.2f}",
                            "PnL Total ($)": f"${float(p.unrealized_pl):.2f}",
                            "PnL Total (%)": f"{float(p.unrealized_plpc)*100:.2f}%"
                        })
                    df_positions = pd.DataFrame(pos_data)
                    st.session_state.alpaca_positions = df_positions
            except Exception as e:
                st.error(f"Erreur de récupération: {e}")
        else:
            st.error("Veuillez vous connecter à Alpaca d'abord.")
            
    if not st.session_state.alpaca_positions.empty:
        st.dataframe(
            st.session_state.alpaca_positions.style.applymap(
                lambda val: 'color: green;' if float(val.strip('%$')) > 0 else 'color: red;',
                subset=['PnL Total ($)', 'PnL Total (%)']
            ),
            use_container_width=True
        )
