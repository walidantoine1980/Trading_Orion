import os
import sys
import glob
import time
import threading
import datetime
import random
import tempfile
import shutil
import pandas as pd
import numpy as np
import matplotlib
# Force Matplotlib en mode 'Agg' (sans interface graphique) pour éviter les crashs avec Tkinter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from fpdf import FPDF

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ==========================================
# CONFIGURATION & SIMULATION (ORION TRADER)
# ==========================================

# Format attendu des fichiers Orion_Live (WFA Allocations)
EXPECTED_COLUMNS = ['Date', 'Ticker', 'Allocation', 'Probabilite_Hausse']

def generate_dummy_data(directory, count=5, source_images_dir=None):
    """
    Génère des fichiers de trading factices : Allocations, Features, Rapport IA ET Logs Alpaca.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print(f"Génération de données test dans {directory}...")
    
    start_date = datetime.datetime.now() - datetime.timedelta(days=60)
    tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'BTC-USD', 'EURUSD=X', 'CASH']
    
    # 1. Génération des CSV (Allocations) et Logs Alpaca correspondants
    for i in range(count):
        # --- Allocations ---
        data = []
        timestamp_str = (start_date + datetime.timedelta(days=i*2)).strftime('%Y%m%d_%H%M%S')
        filename = f"report_WFA_Allocations_{timestamp_str}.csv"
        current_date_str = (start_date + datetime.timedelta(days=i*2)).strftime('%Y-%m-%d')
        
        selected_tickers = random.sample(tickers_list, k=random.randint(3, 6))
        remaining_alloc = 1.0
        
        active_tickers = [] # Pour les logs Alpaca
        
        for ticker in selected_tickers:
            if ticker == 'CASH':
                alloc = 0.0
            else:
                alloc = random.uniform(0.05, 0.3)
                if remaining_alloc - alloc < 0:
                    alloc = remaining_alloc
                remaining_alloc -= alloc
                active_tickers.append(ticker)
            
            prob = random.uniform(0.45, 0.85)
            
            data.append({
                'Date': current_date_str,
                'Ticker': ticker,
                'Allocation': round(alloc, 4),
                'Probabilite_Hausse': round(prob, 4)
            })
        
        if remaining_alloc > 0.01:
             data.append({'Date': current_date_str, 'Ticker': 'CASH', 'Allocation': round(remaining_alloc, 4), 'Probabilite_Hausse': 0.0})
        
        pd.DataFrame(data).to_csv(os.path.join(directory, filename), index=False)
        print(f"  -> {filename} généré.")

        # --- Logs Transactions Alpaca (Simulation) ---
        alpaca_data = []
        for t in active_tickers:
            # On simule un achat ou une vente aléatoire
            side = 'BUY' if random.random() > 0.3 else 'SELL'
            qty = random.randint(1, 100)
            price = round(random.uniform(100, 1000), 2)
            alpaca_data.append({
                'Timestamp': f"{current_date_str} {random.randint(9,16)}:{random.randint(10,59)}:00",
                'Symbol': t,
                'Side': side,
                'Qty': qty,
                'Price': price,
                'Status': 'FILLED'
            })
        
        if alpaca_data:
            alpaca_filename = f"Alpaca_Orders_{timestamp_str}.csv"
            pd.DataFrame(alpaca_data).to_csv(os.path.join(directory, alpaca_filename), index=False)
            print(f"  -> {alpaca_filename} généré.")

    # 2. Génération des fichiers FEATURES (Simulation)
    print("Génération des fichiers Features simulés...")
    feature_tickers = ['AAPL', 'TSLA', 'BTC-USD']
    dates = pd.date_range(start=start_date, periods=30, freq='D')
    
    for ticker in feature_tickers:
        feat_data = {
            'Date': dates,
            'Ticker': ticker,
            'RSI': np.random.uniform(30, 70, size=len(dates)),
            'MACD': np.random.randn(len(dates)).cumsum(),
            'Volatility': np.random.uniform(0.01, 0.05, size=len(dates)),
            'Signal_Strength': np.random.uniform(0, 1, size=len(dates))
        }
        df_feat = pd.DataFrame(feat_data)
        feat_filename = f"Features_{ticker}_20251212.csv"
        df_feat.to_csv(os.path.join(directory, feat_filename), index=False)

    # 3. Génération du fichier texte Prediction IA
    print("Génération du rapport texte IA...")
    ts_ia = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ai_report_content = (
        "=== RAPPORT DE PRÉDICTION IA ===\n"
        f"ID Rapport : {ts_ia}\n"
        f"Date du rapport : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "ANALYSE DE SENTIMENT :\n"
        "Le sentiment global du marché est actuellement HAUSSIER (Bullish).\n"
        "Les indicateurs macro-économiques suggèrent une stabilisation des taux.\n\n"
        "SECTEURS CIBLES :\n"
        "- Technologie (AI & Semi-conducteurs) : Fort potentiel\n"
        "- Énergie : Neutre\n\n"
        "AVERTISSEMENT RISQUE :\n"
        "Volatilité accrue attendue sur les crypto-actifs (BTC-USD) dans les 48h.\n"
        "Recommandation : Maintenir les positions longues sur AAPL et NVDA.\n"
        "---------------------------------------------------\n"
        "Fin du rapport automatique.\n"
    )
    ai_filename = f"report_AI_Prediction_{ts_ia}.txt"
    with open(os.path.join(directory, ai_filename), "w", encoding="utf-8") as f:
        f.write(ai_report_content)

    # 4. Copie Graphiques existants
    existing_images = []
    if source_images_dir and os.path.exists(source_images_dir):
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        for ext in extensions:
            existing_images.extend(glob.glob(os.path.join(source_images_dir, "**", ext), recursive=True))
    
    if existing_images:
        print(f"Copie de {len(existing_images)} graphiques existants...")
        for img_path in existing_images:
            try: shutil.copy(img_path, directory)
            except: pass
    else:
        for ticker in ['BTC-USD', 'ETH-USD']:
            plt.figure(figsize=(10, 6))
            plt.plot(np.random.randn(50).cumsum(), label=f'Cours {ticker}')
            plt.title(f"Analyse Technique (Simulée) - {ticker}")
            plt.legend()
            img_name = f"graph_Analyse_Technique_{ticker}.png"
            plt.savefig(os.path.join(directory, img_name))
            plt.close()

# ==========================================
# MOTEUR D'ANALYSE (BACKEND)
# ==========================================

class OrionAnalyzer:
    def __init__(self):
        self.allocation_frames = []
        self.feature_frames = []
        self.alpaca_frames = [] # Pour stocker les logs Alpaca
        self.summary_stats = {}
        self.files_processed = 0
        self.errors_log = []
        self.technical_analysis_images = []
        self.ai_prediction_text = None

    def scan_and_parse(self, directory, progress_callback=None):
        """Parcourt et parse les fichiers Allocations, Features, Alpaca et le rapport texte."""
        
        # Reset des données
        self.ai_prediction_text = None
        self.allocation_frames = []
        self.feature_frames = []
        self.alpaca_frames = []
        self.technical_analysis_images = []

        all_files = glob.glob(os.path.join(directory, "**", "*.*"), recursive=True)
        
        # Images
        img_exts = ['.png', '.jpg', '.jpeg']
        self.technical_analysis_images = sorted([
            f for f in all_files if any(f.lower().endswith(ext) for ext in img_exts)
        ])
        
        total_files = len(all_files)
        
        for idx, filepath in enumerate(all_files):
            try:
                filename = os.path.basename(filepath)
                ext = filename.split('.')[-1].lower()
                
                # A. Rapport IA
                if filename.startswith("report_AI_Prediction") and filename.endswith(".txt"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if self.ai_prediction_text:
                            self.ai_prediction_text += f"\n\n{'='*40}\nFICHIER : {filename}\n{'='*40}\n\n{content}"
                        else:
                            self.ai_prediction_text = content
                    except UnicodeDecodeError:
                        with open(filepath, 'r', encoding='latin-1') as f:
                            content = f.read()
                        if self.ai_prediction_text:
                            self.ai_prediction_text += f"\n\n{'='*40}\nFICHIER : {filename}\n{'='*40}\n\n{content}"
                        else:
                            self.ai_prediction_text = content
                    continue

                # B. Fichiers CSV/Excel
                if ext in ['csv', 'xlsx']:
                    if ext == 'xlsx':
                        df = pd.read_excel(filepath)
                    else:
                        try: df = pd.read_csv(filepath)
                        except: df = pd.read_csv(filepath, sep=';')

                    df.columns = [c.strip() for c in df.columns]
                    
                    # B1. Alpaca Orders (Nouveau)
                    if "Alpaca_Orders" in filename or ("Symbol" in df.columns and "Side" in df.columns and "Status" in df.columns):
                        df['_SourceFile'] = filename
                        self.alpaca_frames.append(df)

                    # B2. Features
                    elif "Features" in filename:
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        
                        if 'Ticker' not in df.columns:
                            parts = filename.split('_')
                            if len(parts) > 1:
                                guessed_ticker = parts[1]
                                df['Ticker'] = guessed_ticker
                        
                        df['_SourceFile'] = filename
                        self.feature_frames.append(df)
                    
                    # B3. Allocations
                    elif 'Allocation' in df.columns and 'Ticker' in df.columns:
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                        df['_SourceFile'] = filename
                        self.allocation_frames.append(df)
                    
            except Exception as e:
                if filepath.endswith(('.csv', '.xlsx', '.txt')):
                    self.errors_log.append(f"Erreur lecture {os.path.basename(filepath)}: {str(e)}")
            
            if progress_callback:
                progress_callback(idx + 1, total_files)

        self.files_processed = len(self.allocation_frames) + len(self.feature_frames) + len(self.alpaca_frames)
        return self.files_processed

    def aggregate_allocations(self):
        if not self.allocation_frames:
            return pd.DataFrame()
        return pd.concat(self.allocation_frames, ignore_index=True)

    def aggregate_alpaca(self):
        if not self.alpaca_frames:
            return pd.DataFrame()
        return pd.concat(self.alpaca_frames, ignore_index=True)

    def generate_stats(self, df):
        if df.empty: return {}
        stats = {
            'total_rows': len(df),
            'date_min': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns else 'N/A',
            'date_max': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else 'N/A',
            'unique_tickers': df['Ticker'].nunique() if 'Ticker' in df.columns else 0,
            'top_asset': 'N/A',
            'avg_ai_confidence': 0.0
        }
        if 'Ticker' in df.columns:
            non_cash = df[df['Ticker'] != 'CASH']
            if not non_cash.empty: stats['top_asset'] = non_cash['Ticker'].mode()[0]
        if 'Probabilite_Hausse' in df.columns:
            valid = df[df['Probabilite_Hausse'] > 0]
            if not valid.empty: stats['avg_ai_confidence'] = valid['Probabilite_Hausse'].mean()
        return stats

# ==========================================
# GÉNÉRATEUR GRAPHIQUES & PDF
# ==========================================

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport Analyse Robot Trader (Orion)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(5)

    def section_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()
    
    # Helper pour créer un tableau simple
    def draw_table(self, df, columns, col_widths, title=""):
        if title:
            self.set_font('Arial', 'B', 10)
            self.cell(0, 8, title, 0, 1)
        
        # Header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(220, 220, 220)
        for col, w in zip(columns, col_widths):
            self.cell(w, 7, col, 1, 0, 'C', 1)
        self.ln()
        
        # Rows
        self.set_font('Arial', '', 8)
        # On limite le nombre de lignes pour ne pas exploser le PDF si gros log
        limit = 40 
        rows = df.tail(limit) # On prend les plus récents
        if len(df) > limit:
            self.cell(0, 7, f"... (Affichage des {limit} dernières lignes sur {len(df)}) ...", 0, 1, 'C')

        for _, row in rows.iterrows():
            for col, w in zip(columns, col_widths):
                val = str(row.get(col, ''))
                # Petit nettoyage si trop long
                if len(val) > 20: val = val[:17] + "..."
                self.cell(w, 6, val, 1, 0, 'C')
            self.ln()
        self.ln(5)

def create_allocation_charts(df):
    """Génère les camemberts et histogrammes de base."""
    image_paths = []
    if df.empty: return image_paths
    
    # Pie Chart
    if 'Ticker' in df.columns and 'Allocation' in df.columns:
        plt.figure(figsize=(7, 5))
        avg = df.groupby('Ticker')['Allocation'].mean()
        avg = avg[avg > 0.01]
        plt.pie(avg, labels=avg.index, autopct='%1.1f%%', startangle=90)
        plt.title('Allocation Moyenne')
        path = tempfile.mktemp(suffix='.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        image_paths.append(('Répartition du Portefeuille', path))

    # Histogramme Confiance
    if 'Probabilite_Hausse' in df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(df[df['Ticker']!='CASH']['Probabilite_Hausse'], bins=20, color='purple', alpha=0.7)
        plt.title('Distribution Confiance IA')
        path = tempfile.mktemp(suffix='.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        image_paths.append(('Histogramme Confiance IA', path))

    return image_paths

def create_feature_charts(feature_frames):
    image_paths = []
    if not feature_frames:
        return image_paths

    full_feat = pd.concat(feature_frames, ignore_index=True)
    if 'Ticker' not in full_feat.columns:
        return image_paths

    tickers = full_feat['Ticker'].unique()
    
    for ticker in tickers:
        df_t = full_feat[full_feat['Ticker'] == ticker].sort_values('Date')
        numeric_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_plot = [c for c in numeric_cols if c not in ['Ticker', '_SourceFile']]
        
        if not cols_to_plot: continue
            
        num_plots = len(cols_to_plot)
        if num_plots > 6: num_plots = 6 
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes]
        
        fig.suptitle(f"Analyse des Features : {ticker}", fontsize=14, fontweight='bold')
        
        for i in range(num_plots):
            col = cols_to_plot[i]
            ax = axes[i]
            ax.plot(df_t['Date'], df_t[col], label=col, color=plt.cm.tab10(i))
            ax.set_ylabel(col)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        path = tempfile.mktemp(suffix=f'_feat_{ticker}.png')
        plt.savefig(path, dpi=100)
        plt.close()
        image_paths.append(path)
        
    return image_paths

def generate_pdf(stats, alloc_df, alpaca_df, alloc_imgs, feat_imgs, ext_imgs, ai_text_content, output_path, errors):
    pdf = PDFReport()
    pdf.add_page()
    
    # 1. Résumé
    pdf.section_title("Résumé de l'Activité")
    txt = (f"Période: {stats.get('date_min', 'N/A')} au {stats.get('date_max', 'N/A')}\n"
           f"Tickers uniques: {stats.get('unique_tickers', 0)}\n"
           f"Top Actif: {stats.get('top_asset', 'N/A')}\n"
           f"Confiance IA Moyenne: {stats.get('avg_ai_confidence', 0):.2%}")
    pdf.section_body(txt)
    
    # 2. Prédictions IA (Texte)
    if ai_text_content:
        pdf.section_title("Prédictions IA / Analyse de Marché")
        pdf.set_font('Courier', '', 10)
        try:
            clean_text = ai_text_content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 5, clean_text)
        except:
             pdf.multi_cell(0, 5, ai_text_content)
        pdf.ln(5)

    # 3. Tableau Détaillé Allocations & Probabilités (NOUVEAU)
    if not alloc_df.empty:
        pdf.add_page()
        pdf.section_title("Détail des Allocations & Probabilités (Dernier Snapshot)")
        
        # On essaie de prendre le snapshot le plus récent
        if 'Date' in alloc_df.columns:
            last_date = alloc_df['Date'].max()
            current_alloc = alloc_df[alloc_df['Date'] == last_date].copy()
            # Formatage Date string
            current_alloc['Date'] = current_alloc['Date'].dt.strftime('%Y-%m-%d')
        else:
            current_alloc = alloc_df.copy()
        
        # Sélection et renommer pour affichage propre
        cols_to_show = ['Date', 'Ticker', 'Allocation', 'Probabilite_Hausse']
        # Vérif existence colonnes
        cols_final = [c for c in cols_to_show if c in current_alloc.columns]
        
        pdf.draw_table(current_alloc, cols_final, [40, 40, 40, 50], f"Positions au {last_date}")

    # 4. Communications Alpaca (NOUVEAU)
    if not alpaca_df.empty:
        pdf.add_page()
        pdf.section_title("Communications Alpaca (Achats / Ventes)")
        
        # Tri par timestamp si possible
        if 'Timestamp' in alpaca_df.columns:
            alpaca_df = alpaca_df.sort_values('Timestamp', ascending=False)
        
        cols_alpaca = ['Timestamp', 'Symbol', 'Side', 'Qty', 'Price', 'Status']
        # On s'assure que les colonnes existent
        cols_final_alpaca = [c for c in cols_alpaca if c in alpaca_df.columns]
        widths = [45, 30, 20, 20, 30, 30] # Largeurs ajustées
        
        pdf.draw_table(alpaca_df, cols_final_alpaca, widths, "Journal d'Exécution")

    # 5. Graphiques Allocation
    if alloc_imgs:
        pdf.add_page()
        pdf.section_title("Vue d'Ensemble du Portefeuille")
        for title, path in alloc_imgs:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, title, 0, 1)
            pdf.image(path, x=15, w=180)
            pdf.ln(5)
            try: os.remove(path)
            except: pass

    # 6. Features par Action
    if feat_imgs:
        pdf.add_page()
        pdf.section_title("Analyse Détaillée des Features")
        for path in feat_imgs:
            if pdf.get_y() > 200: pdf.add_page()
            pdf.image(path, x=10, w=190)
            pdf.ln(5)
            try: os.remove(path)
            except: pass

    # 7. Graphiques Externes
    if ext_imgs:
        pdf.add_page()
        pdf.section_title("Graphiques Importés")
        
        img_height = 105
        y_margin = 15
        y_start_page = 40
        
        for i, img_path in enumerate(ext_imgs):
            if i > 0 and i % 2 == 0:
                pdf.add_page()
                current_y = y_start_page
            else:
                current_y = y_start_page + (i % 2) * (img_height + y_margin)

            clean_name = os.path.basename(img_path)
            pdf.set_xy(15, current_y - 8)
            pdf.set_font('Arial', 'B', 9)
            pdf.cell(0, 8, f"Fichier : {clean_name}", 0, 1)

            try:
                pdf.image(img_path, x=15, y=current_y, w=180, h=img_height) 
            except Exception as e:
                print(f"Erreur image {img_path}: {e}")

    # 8. Logs
    if errors:
        pdf.add_page()
        pdf.section_title("Logs / Erreurs")
        pdf.set_font('Courier', '', 8)
        pdf.multi_cell(0, 4, "\n".join(errors[:30]))

    try:
        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"Erreur écriture PDF: {e}")
        return False

# ==========================================
# GUI
# ==========================================

class OrionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Orion Reporter - Full Edition (Alloc, Features, Alpaca)")
        self.root.geometry("600x500")
        self.analyzer = OrionAnalyzer()
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Orion Reporter", font=('Helvetica', 16, 'bold')).pack(pady=10)
        ttk.Label(frame, text="Scanne: Allocations, Features, Alpaca Logs, IA Report, Images").pack(pady=5)
        
        f_dir = ttk.LabelFrame(frame, text="Dossier Source", padding=10)
        f_dir.pack(fill=tk.X, pady=10)
        self.entry_dir = ttk.Entry(f_dir)
        self.entry_dir.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(f_dir, text="...", width=3, command=self.browse).pack(side=tk.RIGHT, padx=5)
        
        f_act = ttk.Frame(frame, padding=10)
        f_act.pack(fill=tk.X, pady=10)
        self.btn_gen = ttk.Button(f_act, text="ANALYSER & PDF", command=self.start_analysis, state=tk.DISABLED)
        self.btn_gen.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.btn_test = ttk.Button(f_act, text="Créer Données Test (Tout inclus)", command=self.make_test_data)
        self.btn_test.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.pbar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.pbar.pack(fill=tk.X, pady=10)
        
        self.lbl_stat = ttk.Label(frame, text="Prêt.", foreground="gray")
        self.lbl_stat.pack(anchor=tk.W)
        
    def browse(self):
        d = filedialog.askdirectory()
        if d:
            self.entry_dir.delete(0, tk.END)
            self.entry_dir.insert(0, d)
            self.btn_gen.config(state=tk.NORMAL)
            
    def make_test_data(self):
        target = os.path.join(os.getcwd(), "orion_test_data")
        src = self.entry_dir.get() if os.path.exists(self.entry_dir.get()) else None
        
        # Désactivation boutons pour éviter les clics multiples
        self.btn_gen.config(state=tk.DISABLED)
        self.btn_test.config(state=tk.DISABLED)
        
        self.root.config(cursor="watch")
        self.lbl_stat.config(text="Génération données en cours (Thread)...")
        
        # Exécution dans un thread séparé pour ne pas figer l'interface
        threading.Thread(target=self._run_create_test_data_thread, args=(target, src)).start()

    def _run_create_test_data_thread(self, target, src):
        try:
            generate_dummy_data(target, source_images_dir=src)
            # Succès : mise à jour UI sur le thread principal
            self.root.after(0, lambda: self._on_test_data_success(target))
        except Exception as e:
            # Erreur : mise à jour UI sur le thread principal
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
        finally:
            self.root.after(0, self._reset_ui_state)

    def _on_test_data_success(self, target):
        self.entry_dir.delete(0, tk.END)
        self.entry_dir.insert(0, target)
        messagebox.showinfo("OK", f"Données générées dans:\n{target}")

    def _reset_ui_state(self):
        self.root.config(cursor="")
        self.lbl_stat.config(text="Prêt.")
        self.btn_gen.config(state=tk.NORMAL)
        self.btn_test.config(state=tk.NORMAL)

    def start_analysis(self):
        d_in = self.entry_dir.get()
        f_out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF","*.pdf")], initialfile="Rapport_Orion_Full.pdf")
        if not f_out: return
        
        threading.Thread(target=self.run_thread, args=(d_in, f_out)).start()
        
    def run_thread(self, d_in, f_out):
        self.btn_gen.config(state=tk.DISABLED)
        self.btn_test.config(state=tk.DISABLED)
        
        try:
            # 1. Scan
            self.update_gui("Lecture des fichiers...", 10)
            def cb(c, t): self.update_gui(f"Scan {c}/{t}...", 10 + (c/t)*40)
            
            self.analyzer.scan_and_parse(d_in, cb)
            
            # 2. Agrégation
            self.update_gui("Consolidation Données...", 60)
            df_alloc = self.analyzer.aggregate_allocations()
            df_alpaca = self.analyzer.aggregate_alpaca()
            stats = self.analyzer.generate_stats(df_alloc)
            alloc_imgs = create_allocation_charts(df_alloc)
            
            # 3. Features
            self.update_gui("Génération graphiques Features...", 75)
            feat_imgs = create_feature_charts(self.analyzer.feature_frames)
            
            # 4. PDF
            self.update_gui("Assemblage PDF...", 90)
            ok = generate_pdf(stats, df_alloc, df_alpaca, alloc_imgs, feat_imgs, 
                              self.analyzer.technical_analysis_images, 
                              self.analyzer.ai_prediction_text, f_out, self.analyzer.errors_log)
            
            if ok:
                self.root.after(0, lambda: messagebox.showinfo("Succès", f"Rapport généré:\n{f_out}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Erreur", "Echec création PDF"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
        finally:
            self.update_gui("Terminé.", 0)
            self.root.after(0, lambda: [self.btn_gen.config(state=tk.NORMAL), self.btn_test.config(state=tk.NORMAL)])

    def update_gui(self, txt, val):
        self.root.after(0, lambda: [self.lbl_stat.config(text=txt), self.progress_var.set(val)])

if __name__ == "__main__":
    root = tk.Tk()
    app = OrionApp(root)
    root.mainloop()