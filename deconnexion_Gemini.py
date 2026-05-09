import google.generativeai as genai
import time
from google.api_core import exceptions
import sys

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# REMPLACE CECI PAR TA VRAIE CLÉ API
MA_CLE_API = "TA_CLE_API_ICI"

# --- VÉRIFICATION ET SAISIE INTERACTIVE (Spécial Spyder/Anaconda) ---
if MA_CLE_API == "TA_CLE_API_ICI":
    print("\n⚠️ ATTENTION : Aucune clé API n'est écrite dans le fichier.")
    print("--------------------------------------------------")
    print("Ce n'est pas grave ! Vous pouvez la coller maintenant.")
    print("Lien pour la clé : https://aistudio.google.com/app/apikey")
    print("--------------------------------------------------")
    
    try:
        saisie = input("👉 Collez votre clé API (commençant par AIza...) ici et tapez Entrée : ")
        MA_CLE_API = saisie.strip()
    except Exception as e:
        print(f"Erreur de saisie : {e}")

if not MA_CLE_API or MA_CLE_API == "TA_CLE_API_ICI":
    print("\n❌ ERREUR : Clé API toujours manquante.")
    sys.exit()

genai.configure(api_key=MA_CLE_API)

# -------------------------------------------------------------------
# SÉLECTION AUTOMATIQUE ET INTERACTIVE DU MODÈLE
# -------------------------------------------------------------------
def trouver_meilleur_modele(preference="pro"):
    """
    Trouve le meilleur modèle disponible selon la préférence (pro ou flash).
    Cela évite les erreurs 404 en cherchant les noms exacts.
    """
    print(f"🔍 Recherche du meilleur modèle type '{preference}' disponible...")
    try:
        modeles_dispos = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modeles_dispos.append(m.name)
        
        # Définition de l'ordre de préférence selon le besoin
        if preference == "flash":
            # On cherche tout ce qui contient "flash" en priorité
            liste_pref = ["gemini-1.5-flash", "gemini-flash", "gemini-1.5-pro"]
        else:
            # On cherche la puissance "pro" en priorité
            liste_pref = ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash"]
        
        modele_choisi = None
        
        # Algorithme de recherche
        for pref in liste_pref:
            for m in modeles_dispos:
                if pref in m:
                    modele_choisi = m
                    break
            if modele_choisi: break
        
        if modele_choisi:
            nom_clean = modele_choisi.replace("models/", "")
            print(f"✅ Modèle trouvé : {nom_clean}")
            return nom_clean
        else:
            print("⚠️ Aucun modèle spécifique trouvé. On tente 'gemini-1.5-flash-latest'.")
            return "gemini-1.5-flash-latest"
            
    except Exception as e:
        print(f"⚠️ Impossible de lister les modèles ({e}). Utilisation standard.")
        return "gemini-1.5-flash"

# --- INTERFACE DE CHOIX DU MODE ---
print("\n--------------------------------------------------")
print("CHOIX DU MOTEUR D'INTELLIGENCE")
print("1. Mode PUISSANT (Gemini 1.5 Pro)  -> Idéal pour le code complexe.")
print("2. Mode RAPIDE (Gemini 1.5 Flash)  -> Idéal pour le multitâche (économise les quotas).")
print("--------------------------------------------------")
choix_user = input("Votre choix (1 ou 2) ? [Tapez Entrée pour le mode 1] : ").strip()

if choix_user == "2":
    pref_user = "flash"
    print("🚀 Mode RAPIDE sélectionné.")
else:
    pref_user = "pro"
    print("🧠 Mode PUISSANT sélectionné.")

# Initialisation du modèle
nom_modele_actuel = trouver_meilleur_modele(preference=pref_user)

instruction_systeme = ("Tu es un expert en développement logiciel. "
                       "Tu dois générer du code propre, optimisé et bien commenté. "
                       "Raisonnes étape par étape avant d'écrire la solution.")

try:
    model = genai.GenerativeModel(nom_modele_actuel, system_instruction=instruction_systeme)
except:
    model = genai.GenerativeModel(nom_modele_actuel)


# -------------------------------------------------------------------
# FONCTION PRINCIPALE (AVEC BASCULE AUTOMATIQUE FLASH)
# -------------------------------------------------------------------

def generer_code_robuste(prompt, tentatives_max=5):
    """
    Gère les timeouts ET les quotas dépassés (Erreur 429).
    """
    global model # Permet de changer le modèle globalement si besoin
    tentative = 0
    
    # CONFIGURATION ANTI-COUPURE (Tokens)
    # On autorise un très grand nombre de mots pour éviter que le code soit coupé au milieu
    config_generation = genai.types.GenerationConfig(
        max_output_tokens=8192
    )
    
    while tentative < tentatives_max:
        try:
            nom_actuel = model.model_name.replace('models/', '')
            print(f"🔄 Tentative {tentative + 1}/{tentatives_max} avec {nom_actuel}...")
            
            # APPEL API SÉCURISÉ
            response = model.generate_content(
                prompt,
                # PROTECTION 1 : Timeout de 300s (5 min) pour éviter la déconnexion réseau
                request_options={"timeout": 300},
                # PROTECTION 2 : Config pour éviter la coupure du texte (tokens)
                generation_config=config_generation
            )
            
            return response.text

        except exceptions.ResourceExhausted:
            print("⚠️ QUOTA DÉPASSÉ (Erreur 429) : Le modèle actuel est saturé.")
            print("➡️ SOLUTION : Bascule automatique vers un modèle plus léger...")
            
            # Si on était en Pro, on passe en Flash. Si on était déjà en Flash, on attend.
            nouveau_nom = trouver_meilleur_modele(preference="flash")
            
            try:
                # On met à jour le modèle global
                model = genai.GenerativeModel(nouveau_nom, system_instruction=instruction_systeme)
                print(f"✅ Bascule effectuée vers {nouveau_nom}. On réessaie dans 5 secondes...")
                time.sleep(5)
                continue 
            except:
                print("❌ Impossible de charger le modèle de secours.")

        except exceptions.DeadlineExceeded:
            print("⚠️ Timeout (trop long).")
        except exceptions.ServiceUnavailable:
            print("⚠️ Service surchargé (503).")
        except Exception as e:
            print(f"❌ Erreur inattendue : {e}")
        
        temps_attente = 5 * (tentative + 1) 
        print(f"⏳ On attend {temps_attente} secondes avant de réessayer...\n")
        
        time.sleep(temps_attente)
        tentative += 1

    return "❌ ÉCHEC : Impossible de générer le code."

# -------------------------------------------------------------------
# EXÉCUTION
# -------------------------------------------------------------------

if __name__ == "__main__":
    mon_prompt_code = """
    Écris une classe Python pour gérer un panier d'achat e-commerce.
    Elle doit inclure :
    1. Ajout d'articles (nom, prix, quantité).
    2. Calcul du total avec une TVA de 20%.
    3. Application d'une réduction si le total dépasse 100€.
    """
    
    print("--- Génération du code en cours ---")
    
    resultat = generer_code_robuste(mon_prompt_code)
    
    print("\n--- CODE GÉNÉRÉ ---")
    print(resultat)