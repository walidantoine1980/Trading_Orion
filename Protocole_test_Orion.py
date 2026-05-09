import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import textwrap
from matplotlib.figure import Figure
from scipy import stats
from datetime import datetime

# =====================================================
# PROTOCOLE TEST ORION - COMPARATEUR WFA (V14 STATS COMPLETES)
# =====================================================
# Objectif : Comparer deux fichiers de résultats WFA (5Y vs 8Y).
#
# Nouveauté V14 : 
# - Module Histogrammes enrichi avec Verdict Expert.
# - Analyse de la forme des distributions (Volatilité, Moyenne).
# - Commentaires éducatifs sur le profil "Tout ou Rien" vs "Nuancé".
# =====================================================

class ScrollableFrame(ttk.Frame):
    """
    Un conteneur qui ajoute automatiquement des barres de défilement
    autour d'un cadre interne (scrollable_frame).
    """
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Canvas et Scrollbars
        self.canvas = tk.Canvas(self, bg="#f0f0f0")
        self.scrollbar_v = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_h = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configuration du scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_v.set, xscrollcommand=self.scrollbar_h.set)

        # Layout (Grid)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_v.grid(row=0, column=1, sticky="ns")
        self.scrollbar_h.grid(row=1, column=0, sticky="ew")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Mousewheel scrolling (Optionnel mais pratique)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except:
            pass

class OrionComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protocole Test Orion - Arbitrage Expert 5Y vs 8Y (2018-2025)")
        self.root.geometry("1300x950")
        
        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#003366")
        style.configure("Result.TLabel", font=("Segoe UI", 10))
        style.configure("Bold.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=5)
        
        # Variables de fichiers
        self.file_path_5y = tk.StringVar()
        self.file_path_8y = tk.StringVar()
        
        # --- LISTE DES ÉVÉNEMENTS MARKET AVEC COMMENTAIRES (RISQUES) ---
        self.events = [
            {
                "name": "Volmageddon (Crash Volatilité)", "date": "2018-02-05", "type": "safety",
                "comment": "CONTEXTE : Explosion des produits VIX inverses. Le S&P 500 perd 10% en quelques jours.\nRISQUE FLASH : Panique algorithmique pure. Le Cash protège d'un risque systémique soudain."
            },
            {
                "name": "Correction Noël 2018 (Guerre Commerciale)", "date": "2018-12-24", "type": "safety",
                "comment": "CONTEXTE : Tensions US-Chine + La Fed monte les taux trop vite. Le marché frôle le Bear Market (-20%).\nPRUDENCE : Il faut être défensif avant le pivot de Jerome Powell."
            },
            {
                "name": "Rebond Post-2018 (Pivot Fed)", "date": "2019-01-15", "type": "reactivity",
                "comment": "CONTEXTE : Powell capitule et arrête de monter les taux. Le marché repart en V.\nOPPORTUNITÉ : Fin de la correction. Le risque est d'être trop prudent et de rater le rallye."
            },
            {
                "name": "Crash COVID (Confinement)", "date": "2020-03-20", "type": "safety",
                "comment": "CONTEXTE : Arrêt mondial de l'économie. Volatilité historique (VIX > 80).\nRISQUE EXTRÊME : Le Cash est roi. Tout actif risqué chute de -30% à -50%. Il faut être liquide."
            },
            {
                "name": "Annonce Vaccin (Pfizer/Moderna)", "date": "2020-11-15", "type": "reactivity",
                "comment": "CONTEXTE : La nouvelle qui sauve l'économie. Rotation sectorielle massive.\nOPPORTUNITÉ : Gap haussier historique. Le risque est d'être absent (Coût d'opportunité)."
            },
            {
                "name": "Sommet Tech (Top du marché)", "date": "2021-11-20", "type": "safety",
                "comment": "CONTEXTE : Euphorie post-covid, valorisations absurdes sur la Tech non-rentable.\nRISQUE : Les taux commencent à monter. Les premiers signes de faiblesse apparaissent. Prudence requise."
            },
            {
                "name": "Invasion Ukraine (Choc Géopolitique)", "date": "2022-02-28", "type": "safety",
                "comment": "CONTEXTE : Guerre en Europe + Choc Pétrolier.\nRISQUE GÉOPOLITIQUE : Incertitude maximale. Flight to quality (Or, Dollar, Cash). Éviter les actions européennes."
            },
            {
                "name": "Pic Inflation CPI (Peur Max)", "date": "2022-06-15", "type": "safety",
                "comment": "CONTEXTE : Inflation US à 9.1%. La Fed doit frapper fort.\nRISQUE MACRO : Repricing violent des actifs à risque (Nasdaq) dû à la hausse des taux. Le Cash protège."
            },
            {
                "name": "Creux du Bear Market (Oct 2022)", "date": "2022-10-15", "type": "safety",
                "comment": "CONTEXTE : Pessimisme maximum, mais l'inflation commence à ralentir.\nCAPITULATION : Moment critique. Le risque de baisse diminue, mais la peur reste paralysante."
            },
            {
                "name": "Faillite SVB (Crise Bancaire)", "date": "2023-03-13", "type": "safety",
                "comment": "CONTEXTE : Faillites bancaires US. Peur de contagion systémique.\nRISQUE SYSTÉMIQUE : Volatilité Flash. Le Cash offre une sécurité temporaire face à l'inconnu bancaire."
            },
            {
                "name": "Boom IA (Nvidia Earnings)", "date": "2023-05-26", "type": "reactivity",
                "comment": "CONTEXTE : ChatGPT et Nvidia lancent une nouvelle tendance séculaire.\nRÉVOLUTION TECH : Changement de paradigme. Le risque est de rester sur la touche face au nouveau cycle."
            },
            {
                "name": "Rallye Fin d'Année 2023", "date": "2023-12-29", "type": "reactivity",
                "comment": "CONTEXTE : Le marché anticipe le pivot de la Fed (baisse des taux).\nFOMO INSTITUTIONNEL : Chasse à la performance. Être trop prudent coûte cher en performance relative."
            },
            {
                "name": "Correction Avril 2024 (Taux)", "date": "2024-04-15", "type": "safety",
                "comment": "CONTEXTE : L'inflation est collante, les taux remontent (10 ans US).\nRISQUE DE TAUX : Prises de bénéfices saines mais brutales sur les leaders IA."
            },
            {
                "name": "Volatilité Été 2024 (Yen Carry Trade)", "date": "2024-08-05", "type": "safety",
                "comment": "CONTEXTE : Hausse des taux Japonais, débouclage violent du Yen Carry Trade.\nRISQUE DE LIQUIDITÉ : Ventes forcées algorithmiques. Le Cash permet de ramasser les morceaux plus bas."
            },
            {
                "name": "Pivot Fed (Baisse Taux 50bps)", "date": "2024-09-18", "type": "reactivity",
                "comment": "CONTEXTE : La Fed baisse enfin ses taux de 0.50%. Le cycle de resserrement est fini.\nSIGNAL HAUSSIER : Le marché célèbre la liquidité retrouvée. Il faut être exposé aux actifs risqués."
            },
            {
                "name": "Élections US (Trump Trade)", "date": "2024-11-06", "type": "reactivity",
                "comment": "CONTEXTE : Victoire de Trump. Anticipation de dérégulation et baisses d'impôts.\nEUPHORIE : Les Small Caps, la Tech et la Crypto explosent. Le Cash est un frein à la performance."
            }
        ]
        
        self.create_widgets()

    def create_widgets(self):
        # --- En-tête ---
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="PROTOCOLE TEST ORION (V14 STATS COMPLETES)", style="Title.TLabel").pack()
        ttk.Label(header_frame, text="Analyse Chronologique, Stats & Rapports A4", style="Result.TLabel").pack()

        # --- Section Sélection de Fichiers ---
        files_frame = ttk.LabelFrame(self.root, text="Chargement des Rapports WFA (.csv)", padding=15)
        files_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Fichier 5Y
        f5_frame = ttk.Frame(files_frame)
        f5_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f5_frame, text="Fichier 'Sprinteur' (5Y) :", width=25, anchor='e').pack(side=tk.LEFT, padx=5)
        ttk.Entry(f5_frame, textvariable=self.file_path_5y, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(f5_frame, text="Parcourir...", command=lambda: self.browse_file(self.file_path_5y)).pack(side=tk.LEFT)
        
        # Fichier 8Y
        f8_frame = ttk.Frame(files_frame)
        f8_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f8_frame, text="Fichier 'Marathonien' (8Y) :", width=25, anchor='e').pack(side=tk.LEFT, padx=5)
        ttk.Entry(f8_frame, textvariable=self.file_path_8y, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(f8_frame, text="Parcourir...", command=lambda: self.browse_file(self.file_path_8y)).pack(side=tk.LEFT)

        # --- Boutons d'Action ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill=tk.X)
        
        # Bouton 1 : Analyse Texte
        self.btn_compare = ttk.Button(action_frame, text="1. LANCER L'ANALYSE EXPERTE (16 Événements)", command=self.run_expert_comparison, style="Action.TButton")
        self.btn_compare.pack(fill=tk.X, ipady=5, pady=5)
        
        # Bouton 2 : Visualisation Graphique
        self.btn_visualize = ttk.Button(action_frame, text="2. VISUALISATION GRAPHIQUE SMART", command=self.open_visualizer, style="Action.TButton")
        self.btn_visualize.pack(fill=tk.X, ipady=5, pady=5)

        # Bouton 3 : Statistiques Avancées
        self.btn_stats = ttk.Button(action_frame, text="3. STATISTIQUES AVANCÉES & AUDIT DE RISQUE", command=self.open_stats_module, style="Action.TButton")
        self.btn_stats.pack(fill=tk.X, ipady=5, pady=5)

        # --- Zone de Résultats ---
        result_frame = ttk.LabelFrame(self.root, text="Rapport d'Arbitrage Détaillé", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, font=("Consolas", 9), state='disabled')
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Config des tags de couleur
        self.result_text.tag_config("title", font=("Consolas", 11, "bold"), foreground="blue")
        self.result_text.tag_config("event_header", font=("Consolas", 10, "bold"), foreground="#444444", background="#eeeeee")
        self.result_text.tag_config("comment", font=("Consolas", 9, "italic"), foreground="#555555")
        self.result_text.tag_config("winner", foreground="green", font=("Consolas", 9, "bold"))
        self.result_text.tag_config("loser", foreground="#b03a2e") # Rouge foncé
        self.result_text.tag_config("info", foreground="#333333")
        self.result_text.tag_config("final", font=("Consolas", 14, "bold"), foreground="#6f42c1", justify='center')

    def browse_file(self, string_var):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            string_var.set(filename)

    # --- LOGIQUE ANALYSE TEXTUELLE EXPERTE ---
    def get_cash_allocation_at_date(self, df, target_date_str):
        df['Date'] = pd.to_datetime(df['Date'])
        target_date = pd.to_datetime(target_date_str)
        df_cash = df[df['Ticker'] == 'CASH'].sort_values('Date')
        
        if df_cash.empty:
            df_dates = df['Date'].sort_values().unique()
            if len(df_dates) == 0: return 0.0, "N/A"
            nearest_idx = abs(df_dates - target_date).argmin()
            nearest_date = df_dates[nearest_idx]
            return 0.0, nearest_date.strftime('%Y-%m-%d')

        nearest_idx = abs(df_cash['Date'] - target_date).argmin()
        nearest_row = df_cash.iloc[nearest_idx]
        return float(nearest_row['Allocation']), nearest_row['Date'].strftime('%Y-%m-%d')

    def run_expert_comparison(self):
        f5 = self.file_path_5y.get()
        f8 = self.file_path_8y.get()
        if not f5 or not f8:
            messagebox.showerror("Erreur", "Veuillez sélectionner les deux fichiers CSV.")
            return
        
        try:
            df5 = pd.read_csv(f5)
            df8 = pd.read_csv(f8)
            
            self.result_text.configure(state='normal')
            self.result_text.delete('1.0', tk.END)
            self.result_text.configure(state='disabled')
            
            self.log_result("--- DÉBUT DE L'ANALYSE COMPARATIVE ÉTENDUE (2018-2025) ---\n", "title")
            
            score_5y, score_8y = 0, 0
            
            for i, evt in enumerate(self.events):
                idx = i + 1
                name = evt['name']
                target_date = evt['date']
                e_type = evt['type']
                comment = evt['comment']
                
                # Récupération données
                c5, d5 = self.get_cash_allocation_at_date(df5, target_date)
                c8, d8 = self.get_cash_allocation_at_date(df8, target_date)
                
                type_lbl = "TEST SÉCURITÉ (Cherche Cash Élevé)" if e_type == 'safety' else "TEST RÉACTIVITÉ (Cherche Investissement Max)"
                self.log_result(f"\n{idx}. {name} [{target_date}] - {type_lbl}", "event_header")
                
                # Ajout du commentaire expert
                comment_formatted = "\n".join([f"   > {line}" for line in comment.split('\n')])
                self.log_result(comment_formatted, "comment")
                
                self.log_result(f"   - 5Y ({d5}) : {c5:.1%} CASH", "info")
                self.log_result(f"   - 8Y ({d8}) : {c8:.1%} CASH", "info")
                
                winner = None
                
                # Logique de victoire
                if e_type == 'safety':
                    if c8 > c5 + 0.05: winner = '8Y'
                    elif c5 > c8 + 0.05: winner = '5Y'
                else:
                    if c8 < c5 - 0.05: winner = '8Y'
                    elif c5 < c8 - 0.05: winner = '5Y'
                
                if winner == '8Y':
                    self.log_result("   => VICTOIRE 8Y", "winner")
                    score_8y += 1
                elif winner == '5Y':
                    self.log_result("   => VICTOIRE 5Y", "winner")
                    score_5y += 1
                else:
                    self.log_result("   => ÉGALITÉ (Allocation similaire)", "info")

            # --- CONCLUSION ---
            self.log_result("\n" + "="*40, "info")
            self.log_result(f"SCORE FINAL SUR {len(self.events)} ROUNDS : 5Y [{score_5y}] - 8Y [{score_8y}]", "title")
            
            if score_8y > score_5y:
                pronostic = "\nPRONOSTIC : LE 8Y (MARATHONIEN) EST PLUS ROBUSTE.\nIl a mieux géré la majorité des événements critiques."
            elif score_5y > score_8y:
                pronostic = "\nPRONOSTIC : LE 5Y (SPRINTEUR) EST PLUS ADAPTÉ.\nIl capture mieux la dynamique actuelle malgré les risques."
            else:
                pronostic = "\nPRONOSTIC : ÉGALITÉ PARFAITE.\nConseil Expert : Privilégiez le 8Y pour la sécurité du capital."
                
            self.log_result(pronostic, "final")

        except Exception as e:
            messagebox.showerror("Erreur Analyse", f"Erreur lors de l'analyse : {str(e)}")

    def log_result(self, text, tag=None):
        self.result_text.configure(state='normal')
        self.result_text.insert(tk.END, text + "\n", tag)
        self.result_text.see(tk.END)
        self.result_text.configure(state='disabled')

    # --- LOGIQUE VISUALISATION GRAPH ---
    def open_visualizer(self):
        self._launch_visualizer_window()

    def _launch_visualizer_window(self):
        f5 = self.file_path_5y.get()
        f8 = self.file_path_8y.get()
        if not f5 or not f8:
            messagebox.showerror("Erreur", "Veuillez d'abord sélectionner les deux fichiers.")
            return

        try:
            df5 = pd.read_csv(f5)
            df8 = pd.read_csv(f8)
            df5['Date'] = pd.to_datetime(df5['Date'])
            df8['Date'] = pd.to_datetime(df8['Date'])
            dates5 = set(df5['Date'].dt.strftime('%Y-%m-%d').unique())
            dates8 = set(df8['Date'].dt.strftime('%Y-%m-%d').unique())
            common_dates = sorted(list(dates5.intersection(dates8)))
            
            if not common_dates:
                messagebox.showerror("Erreur Données", "Aucune date commune trouvée.")
                return

            date_labels, date_to_event, evt_dates_map = [], {}, {}
            
            for e in self.events:
                target_dt = pd.to_datetime(e['date'])
                best_match = None
                min_diff = pd.Timedelta(days=45) 
                
                for d_str in common_dates:
                    d_dt = pd.to_datetime(d_str)
                    diff = abs(d_dt - target_dt)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = d_str
                
                if best_match:
                    if best_match not in evt_dates_map: evt_dates_map[best_match] = []
                    evt_dates_map[best_match].append(e)

            for d in common_dates:
                if d in evt_dates_map:
                    for evt_obj in evt_dates_map[d]:
                        label = f"{d} [★ {evt_obj['name']}]"
                        date_to_event[label] = {"date": d, "event_data": evt_obj}
                        date_labels.append(label)
                else:
                    label = d
                    date_to_event[label] = {"date": d, "event_data": None}
                    date_labels.append(label)

            vis_window = tk.Toplevel(self.root)
            vis_window.title("Comparatif Graphique Smart (A4 Print Ready)")
            vis_window.geometry("1000x950")
            
            # 1. Barre de Contrôle (Fixe)
            ctrl_frame = ttk.Frame(vis_window, padding=10)
            ctrl_frame.pack(fill=tk.X, side=tk.TOP)
            
            ttk.Label(ctrl_frame, text="Sélecteur d'Événements (★) / Dates :", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            
            default_idx = len(date_labels) - 1
            date_var = tk.StringVar()
            date_combo = ttk.Combobox(ctrl_frame, textvariable=date_var, values=date_labels, state="readonly", width=60)
            date_combo.current(default_idx)
            date_combo.pack(side=tk.LEFT, padx=10)
            
            # 2. Zone Scrollable pour les Graphiques
            scroll_container = ScrollableFrame(vis_window)
            scroll_container.pack(fill=tk.BOTH, expand=True)
            
            def update_graph(event=None):
                label = date_var.get()
                info = date_to_event.get(label)
                if info:
                    self.plot_comparison(fig, ax1, ax2, ax3, df5, df8, info['date'], info['event_data'])
                    canvas.draw()
            
            date_combo.bind("<<ComboboxSelected>>", update_graph)
            
            # --- CONFIGURATION A4 ---
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.27, 11.69), dpi=100, gridspec_kw={'height_ratios': [3, 2, 1]})
            plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.4, left=0.12, right=0.95)
            
            canvas = FigureCanvasTkAgg(fig, master=scroll_container.scrollable_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            update_graph()
            
            # 3. Barre de Navigation avec SAVE
            nav_frame = ttk.Frame(vis_window, padding=10)
            nav_frame.pack(fill=tk.X, side=tk.BOTTOM)
            
            def save_graph():
                file_path = filedialog.asksaveasfilename(defaultextension=".pdf", 
                                                         filetypes=[("Fichier PDF", "*.pdf"), ("Image PNG", "*.png")])
                if file_path:
                    try:
                        fig.savefig(file_path, papersize='A4', format='pdf' if file_path.endswith('.pdf') else 'png', bbox_inches='tight')
                        messagebox.showinfo("Succès", f"Graphique sauvegardé :\n{file_path}")
                    except Exception as e:
                        messagebox.showerror("Erreur", f"Erreur sauvegarde : {str(e)}")

            ttk.Button(nav_frame, text="💾 Sauvegarder (PDF/PNG) pour Impression", command=save_graph).pack(side=tk.LEFT)
            ttk.Button(nav_frame, text="Fermer", command=vis_window.destroy).pack(side=tk.RIGHT)

        except Exception as e:
             messagebox.showerror("Erreur Visu", str(e))

    def plot_comparison(self, fig, ax1, ax2, ax3, df5, df8, date_str, event_data=None):
        mask5 = df5['Date'] == date_str
        mask8 = df8['Date'] == date_str
        sub5 = df5[mask5].set_index('Ticker')
        sub8 = df8[mask8].set_index('Ticker')
        all_tickers = sorted(list(set(sub5.index) | set(sub8.index)))
        
        alloc5 = [sub5.loc[t, 'Allocation'] if t in sub5.index else 0 for t in all_tickers]
        alloc8 = [sub8.loc[t, 'Allocation'] if t in sub8.index else 0 for t in all_tickers]
        
        prob5 = [sub5.loc[t, 'Probabilite_Hausse'] if (t in sub5.index and 'Probabilite_Hausse' in sub5.columns and t != 'CASH') else 0 for t in all_tickers]
        prob8 = [sub8.loc[t, 'Probabilite_Hausse'] if (t in sub8.index and 'Probabilite_Hausse' in sub8.columns and t != 'CASH') else 0 for t in all_tickers]

        title_main = f"ANALYSE DU {date_str}"
        if event_data: title_main += f"\n>>> {event_data['name']} <<<"
        fig.suptitle(title_main, fontsize=14, fontweight='bold', color='#2c3e50')

        x = np.arange(len(all_tickers))
        width = 0.35

        ax1.clear()
        rects1 = ax1.bar(x - width/2, alloc5, width, label='5Y (Sprinteur)', color='#3498db', alpha=0.85)
        rects2 = ax1.bar(x + width/2, alloc8, width, label='8Y (Marathonien)', color='#e67e22', alpha=0.85)
        ax1.set_ylabel('Allocation (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_tickers, rotation=45, ha='right', fontsize=9) 
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

        for i, t in enumerate(all_tickers):
            if t == 'CASH':
                ax1.get_xticklabels()[i].set_fontweight('bold')
                ax1.get_xticklabels()[i].set_color('green')
                ax1.get_xticklabels()[i].set_fontsize(10)

        def autolabel(ax, rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0.01:
                    ax.annotate(f'{height:.0%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, rotation=90)
        autolabel(ax1, rects1)
        autolabel(ax1, rects2)

        ax2.clear()
        valid_indices = [i for i, t in enumerate(all_tickers) if t != 'CASH']
        if valid_indices:
            x_prob = np.array(valid_indices)
            p5_cl = [prob5[i] for i in valid_indices]
            p8_cl = [prob8[i] for i in valid_indices]
            tick_cl = [all_tickers[i] for i in valid_indices]
            ax2.bar(x_prob - width/2, p5_cl, width, label='Prob 5Y', color='#5dade2', alpha=0.6)
            ax2.bar(x_prob + width/2, p8_cl, width, label='Prob 8Y', color='#f5b041', alpha=0.6)
            ax2.set_ylabel('Probabilité Hausse')
            ax2.set_title('Confiance des Modèles (Probabilité > 0.5 = Signal Achat)', pad=10, fontsize=10, fontweight='bold') 
            ax2.set_xticks(x_prob)
            ax2.set_xticklabels(tick_cl, rotation=45, ha='right', fontsize=9)
            ax2.axhline(0.5, color='red', linestyle='--', linewidth=1)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, "Pas de données de probabilité", ha='center')

        ax3.clear()
        ax3.axis('off')
        if event_data:
            comment_text = event_data['comment']
            wrapper = textwrap.TextWrapper(width=85)
            wrapped_comment = "\n".join(wrapper.wrap(comment_text))
            props = dict(boxstyle='round,pad=1', facecolor='#fff9c4', edgecolor='#fbc02d', alpha=0.9)
            ax3.text(0.5, 0.5, f"ANALYSE STRATÉGIQUE (CONTEXTE & RISQUES) :\n\n{wrapped_comment}", 
                     transform=ax3.transAxes, fontsize=11, verticalalignment='center', horizontalalignment='center', 
                     bbox=props, color='#333333', fontweight='medium')
        else:
            ax3.text(0.5, 0.5, "Sélectionnez un événement marqué d'une étoile [★] pour voir l'analyse.", 
                     transform=ax3.transAxes, ha='center', va='center', color='gray', style='italic')

    # --- NOUVEAU MODULE STATISTIQUES (V13 - EXPERT EDUC) ---
    def open_stats_module(self):
        f5 = self.file_path_5y.get()
        f8 = self.file_path_8y.get()
        if not f5 or not f8:
            messagebox.showerror("Erreur", "Veuillez d'abord sélectionner les deux fichiers CSV.")
            return

        name5 = os.path.basename(f5)
        name8 = os.path.basename(f8)

        try:
            df5 = pd.read_csv(f5)
            df8 = pd.read_csv(f8)
            df5 = df5.dropna(subset=['Allocation'])
            df8 = df8.dropna(subset=['Allocation'])
            
            # --- FENÊTRE GRAPHIQUE ---
            stats_window = tk.Toplevel(self.root)
            stats_window.title(f"Statistiques Avancées : {name5} vs {name8}")
            stats_window.geometry("1400x1000")
            
            notebook = ttk.Notebook(stats_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # ONGLET 1 : Audit KPIs (Enrichi)
            tab1 = ttk.Frame(notebook)
            notebook.add(tab1, text="1. Audit des Risques (KPIs)")
            
            # Pré-calcul des KPIs
            cash5 = df5[df5['Ticker'] == 'CASH']['Allocation']
            cash8 = df8[df8['Ticker'] == 'CASH']['Allocation']
            avg_c5 = cash5.mean() if not cash5.empty else 0
            avg_c8 = cash8.mean() if not cash8.empty else 0
            fc5 = (cash5 > 0.95).sum() / len(cash5) if not cash5.empty else 0
            fc8 = (cash8 > 0.95).sum() / len(cash8) if not cash8.empty else 0
            
            asset5 = df5[df5['Ticker'] != 'CASH']
            asset8 = df8[df8['Ticker'] != 'CASH']
            corr5 = asset5['Probabilite_Hausse'].corr(asset5['Allocation']) if 'Probabilite_Hausse' in asset5 else 0
            corr8 = asset8['Probabilite_Hausse'].corr(asset8['Allocation']) if 'Probabilite_Hausse' in asset8 else 0
            
            self._create_kpi_dashboard(tab1, avg_c5, avg_c8, fc5, fc8, corr5, corr8, name5, name8)
            
            # ONGLET 2 : Distributions (Enrichi)
            tab2 = ttk.Frame(notebook)
            notebook.add(tab2, text="2. Histogrammes (Distribution)")
            self._create_dynamic_dist_tab(tab2, df5, df8, name5, name8)
            
            # ONGLET 3 : Corrélations (EXPERT)
            tab3 = ttk.Frame(notebook)
            notebook.add(tab3, text="3. Corrélations & Cohérence Expert")
            self._create_dynamic_scatter_tab(tab3, df5, df8, name5, name8)
            
            ttk.Button(stats_window, text="Fermer", command=stats_window.destroy).pack(pady=5)

        except Exception as e:
            messagebox.showerror("Erreur Stats", f"Erreur de calcul : {str(e)}")

    def _create_kpi_dashboard(self, parent, avg_c5, avg_c8, fc5, fc8, corr5, corr8, name5, name8):
        # Utilisation d'un cadre scrollable car le contenu pédagogique est long
        scroll_container = ScrollableFrame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        frame = scroll_container.scrollable_frame
        
        # Titre
        ttk.Label(frame, text="AUDIT DE PERFORMANCE & RISQUE (DÉTAILLÉ)", font=("Segoe UI", 16, "bold", "underline"), foreground="#2c3e50").pack(pady=(20, 20))
        
        # Tableau Comparatif
        columns = ("Métrique", f"Fichier 1 ({name5})", f"Fichier 2 ({name8})", "Interprétation Rapide")
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=5)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=250, anchor='center')
        
        tree.pack(fill=tk.X, padx=20)
        
        tree.insert("", tk.END, values=("Position Cash Moyenne", f"{avg_c5:.1%}", f"{avg_c8:.1%}", "Plus c'est haut, plus c'est défensif."))
        tree.insert("", tk.END, values=("Fréquence 'Panic Mode' (>95% Cash)", f"{fc5:.1%}", f"{fc8:.1%}", "Fréquence des sorties totales du marché."))
        tree.insert("", tk.END, values=("Cohérence (Corrélation)", f"{corr5:.2f}", f"{corr8:.2f}", "Proche de 1.0 = IA cohérente."))
        
        # Section Pédagogique (Documentation)
        lbl_info = ttk.Label(frame, text="GUIDE D'INTERPRÉTATION DES RÉSULTATS :", font=("Segoe UI", 12, "bold"), foreground="#003366")
        lbl_info.pack(anchor="w", padx=20, pady=(20, 5))
        
        explanation_text = (
            "1. Position Cash Moyenne :\n"
            "   - Ce chiffre indique le pourcentage moyen de votre portefeuille resté en liquide sur toute la période.\n"
            "   - Si > 50% : Le modèle est très prudent (Défensif). Il rate probablement des hausses mais protège bien.\n"
            "   - Si < 20% : Le modèle est agressif (Offensif). Il est presque toujours investi.\n\n"
            
            "2. Fréquence 'Panic Mode' (>95% Cash) :\n"
            "   - Indique combien de fois l'IA a décidé de TOUT vendre pour passer 100% liquide.\n"
            "   - Un chiffre élevé (> 20%) signifie que l'IA est 'nerveuse' et change souvent d'avis radicalement.\n"
            "   - Un chiffre faible (< 5%) signifie que l'IA préfère ajuster les positions doucement plutôt que de fuir.\n\n"
            
            "3. Cohérence (Corrélation Probabilité/Allocation) :\n"
            "   - Mesure si l'IA 'joint le geste à la parole'.\n"
            "   - Une corrélation proche de 1.00 signifie que plus l'IA est sûre d'elle (haute probabilité), plus elle investit d'argent.\n"
            "   - Une corrélation faible (< 0.20) indique un modèle bruité ou aléatoire : l'IA investit parfois massivement même quand elle n'est pas sûre.\n"
            "   - C'est le juge de paix de la qualité du cerveau de l'IA.\n"
        )
        
        txt_guide = tk.Text(frame, height=16, font=("Consolas", 10), wrap=tk.WORD, bg="#f9f9f9", relief="flat")
        txt_guide.insert(tk.END, explanation_text)
        txt_guide.config(state="disabled")
        txt_guide.pack(fill=tk.X, padx=20, pady=5)
        
        # Fonction d'Export Texte
        def export_kpi_report():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            default_name = f"Rapport_Audit_KPIs_{timestamp}.txt"
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name,
                                                     filetypes=[("Fichier Texte", "*.txt")])
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("=== RAPPORT D'AUDIT DE PERFORMANCE & RISQUE ===\n")
                        f.write(f"Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
                        f.write(f"Comparaison : {name5} (F1) vs {name8} (F2)\n")
                        f.write("="*60 + "\n\n")
                        
                        f.write("--- RÉSULTATS CHIFFRÉS ---\n")
                        f.write(f"1. Position Cash Moyenne      : F1={avg_c5:.1%} | F2={avg_c8:.1%}\n")
                        f.write(f"2. Fréquence 'Panic Mode'     : F1={fc5:.1%} | F2={fc8:.1%}\n")
                        f.write(f"3. Cohérence (Corrélation)    : F1={corr5:.2f} | F2={corr8:.2f}\n\n")
                        
                        f.write("--- GUIDE D'INTERPRÉTATION (RAPPEL) ---\n")
                        f.write(explanation_text)
                        
                        f.write("\n\n=== FIN DU RAPPORT ===")
                        
                    messagebox.showinfo("Succès", f"Rapport d'audit exporté :\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Erreur Export", f"Impossible de sauvegarder le rapport : {str(e)}")

        # Bouton Imprimer
        btn_print = ttk.Button(frame, text="🖨️ IMPRIMER / EXPORTER LE RAPPORT D'AUDIT (TXT)", command=export_kpi_report)
        btn_print.pack(pady=20)

    def _create_dynamic_dist_tab(self, parent, df5, df8, name5, name8):
        # Utilisation d'un cadre scrollable pour inclure les commentaires experts en bas
        scroll_container = ScrollableFrame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        content_frame = scroll_container.scrollable_frame

        ctrl = ttk.Frame(content_frame, padding=10)
        ctrl.pack(fill=tk.X)
        
        num_cols = list(df5.select_dtypes(include=np.number).columns)
        default_var = 'Allocation' if 'Allocation' in num_cols else (num_cols[0] if num_cols else "")
        
        ttk.Label(ctrl, text="Variable à analyser :").pack(side=tk.LEFT)
        var_combo = ttk.Combobox(ctrl, values=num_cols, state="readonly")
        var_combo.set(default_var)
        var_combo.pack(side=tk.LEFT, padx=5)
        
        # Zone Graphique
        graph_frame = ttk.Frame(content_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone Texte Expert (Initialement vide)
        expert_frame = ttk.LabelFrame(content_frame, text="ANALYSE EXPERTE DE LA DISTRIBUTION", padding=15)
        expert_frame.pack(fill=tk.X, padx=10, pady=10)
        lbl_expert = ttk.Label(expert_frame, text="", font=("Segoe UI", 10), justify="left")
        lbl_expert.pack(anchor="w")
        
        def update_dist(event=None):
            col = var_combo.get()
            if not col: return
            for widget in graph_frame.winfo_children(): widget.destroy()
            
            fig = Figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            
            data5, data8, title_suffix = None, None, ""
            
            if col == 'Allocation':
                data5 = df5[df5['Ticker'] == 'CASH'][col]
                data8 = df8[df8['Ticker'] == 'CASH'][col]
                title_suffix = "(Focus CASH)"
            else:
                data5 = df5[col]
                data8 = df8[col]
                title_suffix = "(Global)"
                
            ax.hist(data5, bins=20, alpha=0.5, label=f'{name5}', color='#3498db', density=True)
            ax.hist(data8, bins=20, alpha=0.5, label=f'{name8}', color='#e67e22', density=True)
            
            ax.set_title(f"Distribution : {col} {title_suffix}", fontsize=12, fontweight='bold')
            ax.set_xlabel(f"Valeurs : {col}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # --- GÉNÉRATION DU VERDICT EXPERT (DISTRIBUTION) ---
            if not data5.empty and not data8.empty:
                mean5, std5 = data5.mean(), data5.std()
                mean8, std8 = data8.mean(), data8.std()
                
                verdict_text = "COMPRENDRE CE GRAPHIQUE :\n"
                verdict_text += "- Un pic à gauche signifie des valeurs faibles fréquentes. Un pic à droite signifie des valeurs élevées fréquentes.\n"
                verdict_text += "- Une distribution étalée signifie que l'IA utilise toute la palette de 0% à 100% (Nuancée).\n\n"
                
                verdict_text += f"ANALYSE COMPARATIVE ({col}) :\n"
                verdict_text += f"1. Modèle 5Y : Moyenne = {mean5:.2f} | Écart-Type = {std5:.2f}\n"
                verdict_text += f"2. Modèle 8Y : Moyenne = {mean8:.2f} | Écart-Type = {std8:.2f}\n\n"
                
                if col == 'Allocation': # Analyse spécifique CASH
                    diff_mean = mean8 - mean5
                    if diff_mean > 0.1:
                        verdict_text += ">>> CONCLUSION : Le modèle 8Y est structurellement plus DÉFENSIF (Plus de Cash en moyenne).\n"
                    elif diff_mean < -0.1:
                        verdict_text += ">>> CONCLUSION : Le modèle 5Y est structurellement plus DÉFENSIF (Plus de Cash en moyenne).\n"
                    else:
                        verdict_text += ">>> CONCLUSION : Les deux modèles ont un profil de risque moyen similaire.\n"
                        
                    if std8 > std5 + 0.05:
                        verdict_text += ">>> STYLE : Le 8Y est plus 'Nuancé' (utilise plus de niveaux de cash intermédiaires)."
                    elif std5 > std8 + 0.05:
                        verdict_text += ">>> STYLE : Le 5Y est plus 'Nuancé' (utilise plus de niveaux de cash intermédiaires)."
                    else:
                        verdict_text += ">>> STYLE : Les deux modèles ont une flexibilité d'adaptation comparable."
                
                lbl_expert.config(text=verdict_text, foreground="#004d40")
            
        var_combo.bind("<<ComboboxSelected>>", update_dist)
        update_dist()

    def _create_dynamic_scatter_tab(self, parent, df5, df8, name5, name8):
        # Utilisation d'un cadre scrollable pour inclure les commentaires experts en bas
        scroll_container = ScrollableFrame(parent)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # Le contenu sera dans le cadre interne scrollable
        content_frame = scroll_container.scrollable_frame
        
        ctrl = ttk.Frame(content_frame, padding=10)
        ctrl.pack(fill=tk.X)
        num_cols = list(df5.select_dtypes(include=np.number).columns)
        
        ttk.Label(ctrl, text="Axe X :").pack(side=tk.LEFT)
        x_combo = ttk.Combobox(ctrl, values=num_cols, state="readonly", width=15)
        x_combo.set('Probabilite_Hausse' if 'Probabilite_Hausse' in num_cols else num_cols[0])
        x_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl, text="Axe Y :").pack(side=tk.LEFT, padx=(10,0))
        y_combo = ttk.Combobox(ctrl, values=num_cols, state="readonly", width=15)
        y_combo.set('Allocation')
        y_combo.pack(side=tk.LEFT, padx=5)
        
        # Zone Graphique
        graph_frame = ttk.Frame(content_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Zone Texte Expert (Initialement vide, remplie par update_scatter)
        expert_frame = ttk.LabelFrame(content_frame, text="ANALYSE EXPERTE DE LA RATIONALITÉ", padding=15)
        expert_frame.pack(fill=tk.X, padx=10, pady=10)
        lbl_expert = ttk.Label(expert_frame, text="", font=("Segoe UI", 10), justify="left")
        lbl_expert.pack(anchor="w")
        
        def update_scatter():
            cx = x_combo.get()
            cy = y_combo.get()
            
            # Nettoyage graphique
            for widget in graph_frame.winfo_children(): widget.destroy()
            
            fig = Figure(figsize=(10, 8), dpi=100) # Taille réduite pour laisser place au texte
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            
            r2_score5, r2_score8 = 0, 0
            
            def plot_with_reg(ax, df, color, label, fname):
                nonlocal r2_score5, r2_score8
                clean = df[[cx, cy]].dropna()
                if clean.empty:
                    ax.text(0.5, 0.5, "Données insuffisantes", ha='center')
                    return 0
                x = clean[cx]
                y = clean[cy]
                ax.scatter(x, y, alpha=0.4, color=color, label=label, s=15)
                r_val = 0
                if len(x) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    r_val = r_value**2
                    line = slope * x + intercept
                    ax.plot(x, line, color='red', linestyle='--', linewidth=1, label=f'R²={r_val:.2f}')
                
                ax.set_title(f"{fname} : {cx} vs {cy}")
                ax.set_xlabel(cx)
                ax.set_ylabel(cy)
                ax.legend()
                ax.grid(True, alpha=0.3)
                return r_val

            d5_sub = df5[df5['Ticker'] != 'CASH'] if 'Ticker' in df5.columns else df5
            d8_sub = df8[df8['Ticker'] != 'CASH'] if 'Ticker' in df8.columns else df8
            
            r2_score5 = plot_with_reg(ax1, d5_sub, '#3498db', 'Modèle 5Y', name5)
            r2_score8 = plot_with_reg(ax2, d8_sub, '#e67e22', 'Modèle 8Y', name8)
            
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # --- GÉNÉRATION DU VERDICT EXPERT ---
            verdict_text = "COMPRENDRE CE GRAPHIQUE :\n"
            verdict_text += "- Un R² proche de 1.0 signifie que l'IA est 'Logique' : plus elle est sûre, plus elle investit.\n"
            verdict_text += "- Un R² proche de 0.0 signifie que l'IA est 'Aléatoire' : ses mises n'ont aucun rapport avec sa confiance (Dangereux).\n\n"
            
            verdict_text += f"VERDICT COMPARATIF :\n"
            verdict_text += f"1. Modèle 5Y : Score R² = {r2_score5:.2f} -> "
            if r2_score5 > 0.5: verdict_text += "EXCELLENT. Modèle très rationnel.\n"
            elif r2_score5 > 0.2: verdict_text += "MOYEN. Cohérence acceptable mais perfectible.\n"
            else: verdict_text += "FAIBLE. Modèle bruité (Attention !).\n"
            
            verdict_text += f"2. Modèle 8Y : Score R² = {r2_score8:.2f} -> "
            if r2_score8 > 0.5: verdict_text += "EXCELLENT. Modèle très rationnel.\n"
            elif r2_score8 > 0.2: verdict_text += "MOYEN. Cohérence acceptable.\n"
            else: verdict_text += "FAIBLE. Modèle bruité.\n\n"
            
            if r2_score8 > r2_score5 + 0.1:
                verdict_text += ">>> CONCLUSION : Le modèle 8Y est structurellement plus sain et prévisible."
            elif r2_score5 > r2_score8 + 0.1:
                verdict_text += ">>> CONCLUSION : Le modèle 5Y est structurellement plus sain et prévisible."
            else:
                verdict_text += ">>> CONCLUSION : Les deux modèles ont une cohérence intellectuelle similaire."
                
            lbl_expert.config(text=verdict_text, foreground="#004d40")

        btn_update = ttk.Button(ctrl, text="Mettre à jour & Analyser", command=update_scatter)
        btn_update.pack(side=tk.LEFT, padx=10)
        
        # Lancer l'analyse au démarrage
        update_scatter()

if __name__ == "__main__":
    root = tk.Tk()
    app = OrionComparatorApp(root)
    root.mainloop()