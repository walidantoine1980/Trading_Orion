# =====================================================
# 🧠 APPLICATION DE BACKTEST TECHNIQUE — DASHBOARD PRO
# =====================================================
# Fonctionnalités :
# ✅ RSI / MACD / SMA / Bandes de Bollinger
# ✅ Comparaison avec S&P500
# ✅ Interface Tkinter (choix des tickers, dates, indicateurs)
# ✅ Affichage et sauvegarde des graphiques
# ✅ Export complet en PDF avec :
#     - Synthèse du backtest
#     - Graphiques par actif
#     - Tableau de statistiques
#     - Interprétation automatique (optionnelle)
# =====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import os

# -----------------------------------------------------
# 🧮 FONCTIONS TECHNIQUES
# -----------------------------------------------------
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

# -----------------------------------------------------
# 💼 FONCTION DE BACKTEST
# -----------------------------------------------------
def run_backtest():
    try:
        tickers = [t.strip().upper() for t in ticker_entry.get().split(",") if t.strip()]
        start_date = start_entry.get()
        end_date = end_entry.get()
        initial_capital = float(capital_entry.get())
        frequency = freq_var.get()
        indicators = [i for i, v in indicator_vars.items() if v.get()]
        benchmark_symbol = "^GSPC"

        if not tickers:
            messagebox.showerror("Erreur", "Veuillez entrer au moins un ticker.")
            return

        messagebox.showinfo("Téléchargement", f"Téléchargement des données pour {tickers}...")
        data = yf.download(tickers + [benchmark_symbol], start=start_date, end=end_date, interval="1d")["Close"]
        data.dropna(inplace=True)
        benchmark = data[benchmark_symbol]
        data = data.drop(columns=benchmark_symbol)

        # --- Calcul des indicateurs ---
        results = {}
        for asset in tickers:
            df = pd.DataFrame({"Close": data[asset]})
            if "RSI" in indicators:
                df["RSI"] = compute_rsi(df["Close"])
            if "MACD" in indicators:
                macd, signal = compute_macd(df["Close"])
                df["MACD"], df["Signal_MACD"] = macd, signal
            if "SMA" in indicators:
                df["SMA_20"] = compute_sma(df["Close"])
            if "Bollinger" in indicators:
                sma, upper, lower = compute_bollinger(df["Close"])
                df["Bollinger_Moyenne"], df["Bollinger_Haute"], df["Bollinger_Basse"] = sma, upper, lower
            results[asset] = df

        # --- Simulation portefeuille ---
        rebalance_dates = data.resample(frequency).last().index
        capital = initial_capital
        portfolio_value = [capital]
        for i in range(1, len(rebalance_dates)):
            start = data.index.asof(rebalance_dates[i - 1])
            end = data.index.asof(rebalance_dates[i])
            if pd.isna(start) or pd.isna(end):
                continue
            returns = (data.loc[end] - data.loc[start]) / data.loc[start]
            portfolio_return = returns.mean()
            capital *= (1 + portfolio_return)
            portfolio_value.append(capital)

        total_return = (capital - initial_capital) / initial_capital * 100
        benchmark_return = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0] * 100
        alpha = total_return - benchmark_return

        portfolio_df = pd.DataFrame({
            "Date": [data.index[0]] + [data.index.asof(d) for d in rebalance_dates[:len(portfolio_value)-1]],
            "Valeur_Portefeuille": portfolio_value
        }).set_index("Date")
        portfolio_df["Rendement"] = portfolio_df["Valeur_Portefeuille"].pct_change()
        portfolio_df["Année"] = portfolio_df.index.year

        annual_stats = portfolio_df.groupby("Année").agg({"Rendement": ["mean", "std"]})
        annual_stats.columns = ["Rendement_Moyen", "Volatilité"]
        annual_stats["Sharpe"] = annual_stats["Rendement_Moyen"] / annual_stats["Volatilité"]

        portfolio_df["Sommet"] = portfolio_df["Valeur_Portefeuille"].cummax()
        portfolio_df["Drawdown"] = (portfolio_df["Valeur_Portefeuille"] - portfolio_df["Sommet"]) / portfolio_df["Sommet"]
        max_drawdown = portfolio_df["Drawdown"].min() * 100
        drawdown_end = portfolio_df["Drawdown"].idxmin()
        drawdown_start = portfolio_df.loc[:drawdown_end]["Valeur_Portefeuille"].idxmax()
        drawdown_duration = (drawdown_end - drawdown_start).days

        result_text.set(f"""
        Résultats du Backtest :
        --------------------------
        Capital initial   : ${initial_capital:,.2f}
        Capital final     : ${capital:,.2f}
        Performance totale : {total_return:.2f}%
        S&P500 sur période : {benchmark_return:.2f}%
        Surperformance (α) : {alpha:.2f}%
        Max Drawdown       : {max_drawdown:.2f}%
        Durée Drawdown     : {drawdown_duration} jours
        """)

        for row in tree.get_children():
            tree.delete(row)
        for year, row in annual_stats.iterrows():
            tree.insert("", "end", values=(year, f"{row['Rendement_Moyen']*100:.2f}%",
                                           f"{row['Volatilité']*100:.2f}%", f"{row['Sharpe']:.2f}"))

        # --- Graphique portefeuille ---
        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_df.index, portfolio_df["Valeur_Portefeuille"], label="Portefeuille", linewidth=2)
        plt.plot(benchmark.index, benchmark / benchmark.iloc[0] * initial_capital, linestyle="--", label="S&P500")
        plt.legend()
        plt.title("💼 Évolution du Portefeuille vs S&P500")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # =====================================================
        # 📊 GRAPHIQUES TECHNIQUES + EXPORT PDF
        # =====================================================
        if show_graphs_var.get() or save_graphs_var.get() or export_pdf_var.get():
            output_folder = None
            pdf_path = None
            pdf = None

            if save_graphs_var.get() or export_pdf_var.get():
                output_folder = filedialog.askdirectory(title="Choisir un dossier pour enregistrer les graphiques et PDF")
                if not output_folder:
                    messagebox.showinfo("Annulé", "Export des graphiques annulé.")
                    output_folder = None
                elif export_pdf_var.get():
                    pdf_path = os.path.join(output_folder, "Rapport_Backtest.pdf")
                    pdf = PdfPages(pdf_path)

            # --- Graphiques par actif ---
            for asset, df in results.items():
                fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
                fig.suptitle(f"📈 Analyse Technique : {asset}", fontsize=14, fontweight='bold')

                axes[0].plot(df["Close"], color='black', label="Cours de clôture")
                if "SMA" in indicators and "SMA_20" in df:
                    axes[0].plot(df["SMA_20"], color='orange', label="SMA (20)")
                if "Bollinger" in indicators:
                    axes[0].plot(df["Bollinger_Haute"], color='red', linestyle='--', label='Bollinger Haute')
                    axes[0].plot(df["Bollinger_Basse"], color='green', linestyle='--', label='Bollinger Basse')
                axes[0].legend()
                axes[0].set_title("Cours et tendances (SMA / Bollinger)")
                axes[0].grid(True, linestyle='--', alpha=0.5)

                if "RSI" in indicators:
                    axes[1].plot(df["RSI"], color='purple', label="RSI (14 jours)")
                    axes[1].axhline(70, color='red', linestyle='--', alpha=0.6, label='Surachat (70)')
                    axes[1].axhline(30, color='green', linestyle='--', alpha=0.6, label='Survente (30)')
                    axes[1].set_title("RSI (Relative Strength Index)")
                    axes[1].legend()
                    axes[1].grid(True, linestyle='--', alpha=0.5)

                if "MACD" in indicators:
                    axes[2].plot(df["MACD"], label="MACD", color='blue')
                    axes[2].plot(df["Signal_MACD"], label="Signal", color='orange')
                    axes[2].axhline(0, color='black', linestyle='--', alpha=0.7)
                    axes[2].set_title("MACD et Ligne de Signal")
                    axes[2].legend()
                    axes[2].grid(True, linestyle='--', alpha=0.5)

                plt.tight_layout(rect=[0, 0, 1, 0.96])

                if output_folder and save_graphs_var.get():
                    plt.savefig(os.path.join(output_folder, f"{asset}_Analyse_Technique.png"), dpi=300)

                if pdf:
                    pdf.savefig(fig)

                if show_graphs_var.get():
                    plt.show()
                else:
                    plt.close(fig)

            # --- Page de synthèse PDF ---
            if pdf:
                fig_summary = plt.figure(figsize=(8.5, 11))
                plt.axis("off")
                summary = f"""
                RAPPORT DE BACKTEST TECHNIQUE
                
                Période : {start_date} → {end_date}
                Actifs analysés : {', '.join(tickers)}
                
                -----
                Résumé :
                - Capital initial : ${initial_capital:,.2f}
                - Capital final : ${capital:,.2f}
                - Performance totale : {total_return:.2f}%
                - Surperformance (α) : {alpha:.2f}%
                - Max Drawdown : {max_drawdown:.2f}%
                - Durée du drawdown : {drawdown_duration} jours
                """
                plt.text(0.05, 0.8, summary, fontsize=12, va='top')
                pdf.savefig(fig_summary)
                plt.close(fig_summary)

                # --- 📊 Statistiques par actif ---
                stats_data = []
                for asset, df in results.items():
                    perf = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
                    vol = df["Close"].pct_change().std() * np.sqrt(252) * 100
                    rsi_mean = df["RSI"].mean() if "RSI" in df else np.nan
                    macd_mean = df["MACD"].mean() if "MACD" in df else np.nan
                    stats_data.append([asset, f"{perf:.2f}%", f"{vol:.2f}%", f"{rsi_mean:.2f}", f"{macd_mean:.2f}", f'{df["Close"].iloc[-1]:.2f}'])

                stats_df = pd.DataFrame(stats_data, columns=["Actif", "Rendement", "Volatilité", "RSI Moyen", "MACD Moyen", "Cours Final"])

                fig_table = plt.figure(figsize=(8.5, 11))
                plt.axis("off")
                plt.title("📊 Statistiques par Actif", fontsize=14, fontweight='bold', y=0.95)
                plt.table(cellText=stats_df.values, colLabels=stats_df.columns, loc="center", cellLoc='center')
                pdf.savefig(fig_table)
                plt.close(fig_table)

                # --- 🧠 Interprétation automatique (optionnelle) ---
                if include_analysis_var.get():
                    fig_comment = plt.figure(figsize=(8.5, 11))
                    plt.axis("off")
                    plt.title("🧠 Interprétation Automatique du Backtest", fontsize=14, fontweight='bold', y=0.95)
                    insights = []
                    for _, row in stats_df.iterrows():
                        asset = row["Actif"]
                        perf = float(row["Rendement"].replace("%", ""))
                        vol = float(row["Volatilité"].replace("%", ""))
                        try:
                            rsi_mean = float(row["RSI Moyen"])
                        except:
                            rsi_mean = np.nan
                        try:
                            macd_mean = float(row["MACD Moyen"])
                        except:
                            macd_mean = np.nan
                        interpretation = f"🔹 {asset} : "
                        if perf > 0:
                            interpretation += f"Performance positive de {perf:.1f}%. "
                        else:
                            interpretation += f"Baisse de {abs(perf):.1f}%. "
                        if vol > 25:
                            interpretation += f"Volatilité élevée ({vol:.1f}%). "
                        else:
                            interpretation += f"Volatilité modérée ({vol:.1f}%). "
                        if not np.isnan(rsi_mean):
                            if rsi_mean > 70:
                                interpretation += f"RSI moyen {rsi_mean:.1f} → actif souvent en surachat. "
                            elif rsi_mean < 30:
                                interpretation += f"RSI moyen {rsi_mean:.1f} → actif souvent sous-évalué. "
                            else:
                                interpretation += f"RSI moyen {rsi_mean:.1f} → dynamique équilibrée. "
                        if not np.isnan(macd_mean):
                            if macd_mean > 0:
                                interpretation += f"MACD positif → momentum haussier."
                            else:
                                interpretation += f"MACD négatif → pression vendeuse dominante."
                        insights.append(interpretation)
                    y = 0.9
                    for text in insights:
                        plt.text(0.05, y, text, fontsize=11, va='top', wrap=True)
                        y -= 0.07
                    plt.text(0.05, y - 0.05,
                             "⚠️ Ces interprétations sont indicatives et ne constituent pas un conseil en investissement.",
                             fontsize=10, color='gray')
                    pdf.savefig(fig_comment)
                    plt.close(fig_comment)

                pdf.close()
                messagebox.showinfo("PDF généré", f"✅ Rapport complet créé :\n{pdf_path}")

    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

# -----------------------------------------------------
# 🖥️ INTERFACE UTILISATEUR (Tkinter)
# -----------------------------------------------------
root = tk.Tk()
root.title("Dashboard Backtest Technique - RSI / MACD / SMA / Bollinger")
root.geometry("700x950")

tk.Label(root, text="📊 BACKTEST DASHBOARD TECHNIQUE", font=("Arial", 14, "bold")).pack(pady=10)
tk.Label(root,text="Designed by Antoine Aoun 💙",font=("Arial", 11, "bold")).pack(pady=10)
tk.Label(root, text="Tickers Yahoo Finance (ex: AAPL, TSLA, BTC-USD)").pack()
ticker_entry = tk.Entry(root, width=50)
ticker_entry.insert(0, "AAPL, TSLA")
ticker_entry.pack(pady=5)

tk.Label(root, text="Date de début (YYYY-MM-DD)").pack()
start_entry = tk.Entry(root, width=25)
start_entry.insert(0, "2020-01-01")
start_entry.pack(pady=5)

tk.Label(root, text="Date de fin (YYYY-MM-DD)").pack()
end_entry = tk.Entry(root, width=25)
end_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
end_entry.pack(pady=5)

tk.Label(root, text="Capital initial ($)").pack()
capital_entry = tk.Entry(root, width=25)
capital_entry.insert(0, "10000")
capital_entry.pack(pady=5)

tk.Label(root, text="Fréquence d’analyse").pack()
freq_var = tk.StringVar(value="1M")
ttk.Combobox(root, textvariable=freq_var, values=["1D", "1W", "1M"]).pack(pady=5)

tk.Label(root, text="Indicateurs à afficher :").pack(pady=5)
indicator_vars = {name: tk.BooleanVar() for name in ["RSI", "MACD", "SMA", "Bollinger"]}
for name, var in indicator_vars.items():
    tk.Checkbutton(root, text=name, variable=var).pack(anchor="w", padx=100)

# ✅ Options
show_graphs_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Afficher les graphiques techniques", variable=show_graphs_var).pack(pady=5)

save_graphs_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Enregistrer les graphiques au format PNG", variable=save_graphs_var).pack(pady=5)

export_pdf_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Exporter le rapport complet en PDF (avec statistiques par actif)", variable=export_pdf_var).pack(pady=5)

include_analysis_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Inclure l’analyse automatique dans le PDF", variable=include_analysis_var).pack(pady=5)

# --- Lancer ---
tk.Button(root, text="🚀 Lancer le Backtest", command=run_backtest,
          bg="#4CAF50", fg="white", font=("Arial", 12, "bold")).pack(pady=15)

# --- Résultats et tableau ---
result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, font=("Courier", 10), justify="left").pack(pady=10)

tk.Label(root, text="📅 Performance Annuelle du Portefeuille", font=("Arial", 12, "bold")).pack(pady=5)
cols = ("Année", "Rendement (%)", "Volatilité (%)", "Sharpe")
tree = ttk.Treeview(root, columns=cols, show="headings", height=6)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, anchor="center", width=150)
tree.pack(pady=5)

tk.Label(root, text="Formation Pédagogique — Analyse Technique & Backtesting 🧠",
         fg="gray", font=("Arial", 9)).pack(side="bottom", pady=5)

root.mainloop()
