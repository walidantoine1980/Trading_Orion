# =====================================================
# === LANCEUR ORION AI (v1.4 - Extended) ===
# =====================================================
#
# OBJECTIF DE CE SCRIPT :
#
# 1. Fournir une interface de sélection simple pour tous les modules.
# 2. Lancer le module "LIVE / PRÉDICTION" (Orion_Live_AI.py).
# 3. Lancer le module "BACKTESTING / ANALYSE" (Orion_Backtest_AI.py).
# 4. Lancer le module "PROTOCOLE" (Protocole_test_Orion.py).
# 5. Lancer le module "REPORT" (Orion_Reporter.py).
#
# Gestion automatique des dossiers d'archives pour Live et Backtest.
#
# v1.1 : Suppression de `root.destroy()` pour que le lanceur RESTE OUVERT.
# v1.2 : Ajout de la création de dossiers DATE&HEURE pour les archives Backtest.
# v1.3 : Ajout de la création de dossiers LIVE_DATE&HEURE pour les archives Live.
# v1.4 : Ajout des boutons "5Y or 8Y" et "Report".
# =====================================================

# --- Importation des modules nécessaires ---

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import datetime

# --- Noms des fichiers scripts à lancer ---
# (Ces fichiers DOIVENT être dans le même dossier)

LIVE_SCRIPT_NAME = "Orion_Live_AI.py"
BACKTEST_SCRIPT_NAME = "Orion_Backtest_AI.py"
PROTOCOL_SCRIPT_NAME = "Protocole_test_Orion.py"
REPORT_SCRIPT_NAME = "Orion_Reporter.py"


def get_python_executable():
    """
    Trouve le chemin de l'exécutable Python qui exécute
    actuellement ce script (important pour les environnements virtuels).
    """
    return sys.executable


def check_files_exist(script_dir):
    """Vérifie si les scripts cibles existent."""
    live_path = os.path.join(script_dir, LIVE_SCRIPT_NAME)
    backtest_path = os.path.join(script_dir, BACKTEST_SCRIPT_NAME)
    protocol_path = os.path.join(script_dir, PROTOCOL_SCRIPT_NAME)
    report_path = os.path.join(script_dir, REPORT_SCRIPT_NAME)
    
    missing = []
    if not os.path.exists(live_path):
        missing.append(LIVE_SCRIPT_NAME)
    if not os.path.exists(backtest_path):
        missing.append(BACKTEST_SCRIPT_NAME)
    if not os.path.exists(protocol_path):
        missing.append(PROTOCOL_SCRIPT_NAME)
    if not os.path.exists(report_path):
        missing.append(REPORT_SCRIPT_NAME)
        
    return missing


def create_backtest_folders(script_dir):
    """
    v1.2 : Crée les sous-dossiers datés pour les archives de Backtest.
    """
    try:
        # Génère le nom du dossier basé sur la date et l'heure actuelle
        timestamp_folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Chemins de base demandés
        base_simple = os.path.join(script_dir, "Orion_Data", "1_BACKTEST_ARCHIVES", "SIMPLE_BACKTEST")
        base_wfa = os.path.join(script_dir, "Orion_Data", "1_BACKTEST_ARCHIVES", "WFA_ANALYSIS")
        
        # Chemins complets des nouveaux sous-dossiers
        new_folder_simple = os.path.join(base_simple, timestamp_folder_name)
        new_folder_wfa = os.path.join(base_wfa, timestamp_folder_name)
        
        # Création effective
        os.makedirs(new_folder_simple, exist_ok=True)
        os.makedirs(new_folder_wfa, exist_ok=True)
        
        print(f"[INFO] Dossier archive Backtest créé : {new_folder_simple}")
        print(f"[INFO] Dossier archive Backtest créé : {new_folder_wfa}")
        return True, timestamp_folder_name
        
    except Exception as e:
        print(f"[ERREUR] Création dossiers archives Backtest : {e}")
        return False, str(e)


def create_live_folders(script_dir):
    """
    v1.3 : Crée le sous-dossier daté pour les archives Live.
    """
    try:
        # Génère le nom du dossier basé sur la date et l'heure actuelle
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"LIVE_{timestamp}"
        
        # Chemin de base demandé
        base_live = os.path.join(script_dir, "Orion_Data", "2_LIVE_ARCHIVES")
        
        # Chemin complet du nouveau sous-dossier
        new_folder_live = os.path.join(base_live, folder_name)
        
        # Création effective
        os.makedirs(new_folder_live, exist_ok=True)
        
        print(f"[INFO] Dossier archive Live créé : {new_folder_live}")
        return True, folder_name
        
    except Exception as e:
        print(f"[ERREUR] Création dossier archive Live : {e}")
        return False, str(e)


def launch_script(script_name, status_label, root):
    """
    Lance un script Python en tant que nouveau processus.
    """
    python_exe = get_python_executable()
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(script_path):
        messagebox.showerror("Erreur Fichier", 
                             f"Le fichier '{script_name}' est introuvable.\n\n"
                             f"Vérifiez qu'il est dans le même dossier que ce lanceur.")
        return

    try:
        command = [python_exe, script_path]
        status_text = f"Lancement de {script_name}..."
        print(status_text)
        status_label.config(text=status_text, foreground="blue")
        
        subprocess.Popen(command)
        
        status_label.config(text=f"{script_name} lancé dans une nouvelle fenêtre.")
        
    except Exception as e:
        error_msg = f"Impossible de lancer {script_name}:\n{e}"
        print(error_msg)
        messagebox.showerror("Erreur de Lancement", error_msg)
        status_label.config(text="Échec du lancement.", foreground="red")


def launch_backtest_sequence(status_label, root):
    """
    v1.2 : Wrapper spécifique pour le Backtest (avec création de dossiers).
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    success, info = create_backtest_folders(script_dir)
    
    if not success:
        proceed = messagebox.askyesno("Erreur Dossiers", 
                                      f"Impossible de créer les dossiers d'archives Backtest :\n{info}\n\nVoulez-vous quand même lancer le Backtest ?")
        if not proceed:
            status_label.config(text="Lancement Backtest annulé.", foreground="orange")
            return

    launch_script(BACKTEST_SCRIPT_NAME, status_label, root)


def launch_live_sequence(status_label, root):
    """
    v1.3 : Wrapper spécifique pour le Live (avec création de dossiers).
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    success, info = create_live_folders(script_dir)
    
    if not success:
        proceed = messagebox.askyesno("Erreur Dossiers", 
                                      f"Impossible de créer le dossier d'archives Live :\n{info}\n\nVoulez-vous quand même lancer le Live ?")
        if not proceed:
            status_label.config(text="Lancement Live annulé.", foreground="orange")
            return

    launch_script(LIVE_SCRIPT_NAME, status_label, root)


def create_launcher_gui():
    """
    Crée l'interface graphique (GUI) principale du lanceur.
    """
    root = tk.Tk()
    root.title("Orion AI - Sélecteur de Module")
    # Taille ajustée pour accueillir les nouveaux boutons
    root.geometry("550x680")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TButton", padding=10, font=('Segoe UI', 12, 'bold'))
    
    # Styles existants
    style.configure("Green.TButton", background="#28a745", foreground="white")
    style.map("Green.TButton", background=[('active', '#218838')])
    
    style.configure("Blue.TButton", background="#007bff", foreground="white")
    style.map("Blue.TButton", background=[('active', '#0056b3')])

    # Nouveaux styles pour les nouveaux boutons
    style.configure("Purple.TButton", background="#6f42c1", foreground="white")
    style.map("Purple.TButton", background=[('active', '#5a32a3')])

    style.configure("Orange.TButton", background="#fd7e14", foreground="white")
    style.map("Orange.TButton", background=[('active', '#e36209')])
    
    main_frame = ttk.Frame(root, padding=25)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # --- Titres ---
    title_label = ttk.Label(main_frame, text="SÉLECTEUR DE MODULE ORION AI",
                            font=("Segoe UI", 20, "bold"), anchor="center")
    title_label.pack(pady=(10, 15))
    
    signature_label = ttk.Label(main_frame, text="Designed by Antoine Aoun",
                                font=("Segoe UI", 12, "bold"),
                                foreground="#28a745",
                                anchor="center")
    signature_label.pack(pady=(0, 10))
    
    subtitle_label = ttk.Label(main_frame, text="Choisissez le module à démarrer :",
                               font=("Segoe UI", 12), anchor="center")
    subtitle_label.pack(pady=(0, 20))

    # --- Bouton 1: Mode BACKTEST ---
    backtest_button = ttk.Button(main_frame, 
                                 text="🧠 Mode BACKTESTING / ANALYSE\n(Orion_Backtest_AI.py)", 
                                 style="Blue.TButton",
                                 command=lambda: launch_backtest_sequence(status_label, root))
    
    backtest_button.pack(fill=tk.X, pady=8, ipady=10)

    # --- Bouton 2: Mode LIVE ---
    live_button = ttk.Button(main_frame, 
                             text="🚀 Mode LIVE / PRÉDICTION\n(Orion_Live_AI.py)", 
                             style="Green.TButton",
                             command=lambda: launch_live_sequence(status_label, root))
    live_button.pack(fill=tk.X, pady=8, ipady=10)

    # --- Bouton 3: 5Y or 8Y ---
    protocol_button = ttk.Button(main_frame,
                                 text="📊 5Y or 8Y\n(Protocole_test_Orion.py)",
                                 style="Purple.TButton",
                                 command=lambda: launch_script(PROTOCOL_SCRIPT_NAME, status_label, root))
    protocol_button.pack(fill=tk.X, pady=8, ipady=10)

    # --- Bouton 4: Report ---
    report_button = ttk.Button(main_frame,
                               text="📑 Report\n(Orion-Report.py)",
                               style="Orange.TButton",
                               command=lambda: launch_script(REPORT_SCRIPT_NAME, status_label, root))
    report_button.pack(fill=tk.X, pady=8, ipady=10)
    
    # --- Séparateur et Statut ---
    ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(25, 15))

    status_label = ttk.Label(main_frame, text="Prêt. (Les modules se lanceront dans une nouvelle fenêtre.)",
                             font=("Segoe UI", 10, "italic"),
                             anchor="center")
    status_label.pack(fill=tk.X, pady=5)
    
    # --- Vérification des fichiers au démarrage ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() 
        
    missing_files = check_files_exist(script_dir)
    
    if missing_files:
        msg = "ATTENTION : Les fichiers suivants sont introuvables:\n\n" + \
              "\n".join(missing_files) + \
              "\n\nVeuillez vous assurer qu'ils sont dans le même dossier que ce lanceur."
              
        messagebox.showwarning("Fichiers Manquants", msg)
        
        if LIVE_SCRIPT_NAME in missing_files:
            live_button.config(state=tk.DISABLED)
            
        if BACKTEST_SCRIPT_NAME in missing_files:
            backtest_button.config(state=tk.DISABLED)
            
        if PROTOCOL_SCRIPT_NAME in missing_files:
            protocol_button.config(state=tk.DISABLED)

        if REPORT_SCRIPT_NAME in missing_files:
            report_button.config(state=tk.DISABLED)
            
        status_label.config(text="ERREUR: Fichiers modules introuvables!", foreground="red")

    root.mainloop()


if __name__ == "__main__":
    create_launcher_gui()