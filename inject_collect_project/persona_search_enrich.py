import requests
import json
import re
import time
import os

# --- CONFIGURATION API ---
SERPER_API_KEY = "5d3fc4b14c2e8056675a3d9727cf7160f177cf3c"
HUNTER_API_KEY = "305ef979da18c10fdf9c44819f5c5fef14f13496"
SNOVIO_CLIENT_ID = "09e88d30096d79a3c1d72d82d50f9030"
SNOVIO_CLIENT_SECRET = "c0a5ae2d5c62d7b262b5309647e2d621"
TOMBA_API_KEY = "ta_gm3o7ln41r6e27effri76wi1wskm14sadvhds"
TOMBA_API_SECRET = "ts_c801a3af-843b-4652-b970-179921420e21"
AEROLEADS_API_KEY = "14d7acd6938c24ba059014c50f1eff35"

def clean_linkedin_name(raw_name):
    """Nettoie les scories de LinkedIn pour avoir un nom propre"""
    clean = raw_name.replace('\u200f', '').replace('\u200e', '')
    clean = re.split(r'[|,\-]', clean)[0]
    return clean.strip()

def split_name(full_name):
    """Sépare le prénom et le nom"""
    parts = full_name.split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    first_name = parts[0]
    last_name = " ".join(parts[1:])
    return first_name, last_name

def get_hunter_data(first_name, last_name, domain):
    if not first_name or not last_name:
        return "Non trouvé", "Non trouvé"
        
    print(f"   🔍 [Hunter.io] Recherche pour: {first_name} {last_name}...")
    url = f"https://api.hunter.io/v2/email-finder?domain={domain}&first_name={first_name}&last_name={last_name}&api_key={HUNTER_API_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("data", {})
            return data.get("email") or "Non trouvé", data.get("position") or "Non trouvé"
    except Exception:
        pass
    return "Non trouvé", "Non trouvé"

def get_snovio_token():
    url = "https://api.snov.io/v1/oauth/access_token"
    data = {"grant_type": "client_credentials", "client_id": SNOVIO_CLIENT_ID, "client_secret": SNOVIO_CLIENT_SECRET}
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception:
        pass
    return None

def get_snovio_data(first_name, last_name, domain, token):
    if not token or not first_name or not last_name:
        return "Non trouvé", "Non trouvé"
        
    print(f"   🔄 [Snov.io] Fallback activé pour: {first_name} {last_name}...")
    url = "https://api.snov.io/v1/get-emails-from-names"
    params = {"access_token": token}
    payload = {"firstName": first_name, "lastName": last_name, "domain": domain}
    
    try:
        response = requests.post(url, params=params, json=payload)
        if response.status_code == 200:
            emails = response.json().get("data", {}).get("emails", [])
            if emails:
                return emails[0].get("email", "Non trouvé"), "Non trouvé"
    except Exception:
        pass
    return "Non trouvé", "Non trouvé"

def get_tomba_data(first_name, last_name, domain):
    if not first_name or not last_name:
        return "Non trouvé", "Non trouvé", "Non trouvé"
        
    print(f"   📥 [Tomba.io] Fallback activé pour: {first_name} {last_name}...")
    url = "https://api.tomba.io/v1/email-finder"
    headers = {"X-Tomba-Key": TOMBA_API_KEY, "X-Tomba-Secret": TOMBA_API_SECRET}
    params = {"first_name": first_name, "last_name": last_name, "domain": domain}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json().get("data", {})
            return (data.get("email") or "Non trouvé", 
                    data.get("position") or "Non trouvé", 
                    data.get("phone_number") or "Non trouvé")
    except Exception:
        pass
    return "Non trouvé", "Non trouvé", "Non trouvé"

def get_aeroleads_data(first_name, last_name, company, linkedin_url):
    """Fallback ultime AeroLeads : Teste LinkedIn d'abord, puis Nom/Entreprise et retourne les données brutes"""
    email, position, phone = None, None, None
    raw_data = None
    print(f"   🚀 [AeroLeads] Enrichissement profond activé pour: {first_name} {last_name}...")

    # 1. Tentative via LinkedIn URL
    if linkedin_url:
        print(f"      -> Tentative via API LinkedIn...")
        try:
            resp = requests.get("https://aeroleads.com/api/get_linkedin_details", params={'api_key': AEROLEADS_API_KEY, 'linkedin_url': linkedin_url}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                raw_data = data # Capture des données brutes
                
                email = data.get('email')
                if not email and data.get('emails'): email = data['emails'][0].get('address')
                phone = data.get('phone_number') or data.get('phone')
                if not phone and data.get('phone_numbers'): phone = data['phone_numbers'][0]
                position = data.get('title') or data.get('designation')
        except Exception:
            pass

    # 2. Tentative via Nom/Entreprise si toujours incomplet
    if not email or not phone or not raw_data:
        print(f"      -> Données manquantes. Tentative via API Nom/Entreprise...")
        try:
            resp = requests.get("https://aeroleads.com/api/get_email_details", params={'api_key': AEROLEADS_API_KEY, 'first_name': first_name, 'last_name': last_name, 'company': company}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                raw_data = data # Capture des données brutes
                
                e_fall = data.get('email')
                if not e_fall and data.get('emails'): e_fall = data['emails'][0].get('address')
                if e_fall and not email: email = e_fall
                
                p_fall = data.get('phone_number') or data.get('phone')
                if not p_fall and data.get('phone_numbers'): p_fall = data['phone_numbers'][0]
                if p_fall and not phone: phone = p_fall
                
                pos_fall = data.get('title') or data.get('designation')
                if pos_fall and not position: position = pos_fall
        except Exception:
            pass

    return email or "Non trouvé", position or "Non trouvé", phone or "Non trouvé", raw_data


def search_and_enrich(domain, location, role="Sales"):
    """
    Recherche Serper + Cascade d'enrichissement.
    Retourne la liste des profils avec les données extraites à la racine.
    """
    company_name = domain.split('.')[0].capitalize()
    
    # Ajout du rôle ciblé dans la requête Google (ex: "Sales")
    dynamic_query = f'site:linkedin.com/in/ "{company_name}" "{location}" "{role}"'
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": dynamic_query, "gl": "tn", "hl": "fr", "num": 3})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    print(f"📡 Étape 1 : Recherche Serper (Focus: {role}) avec la requête : '{dynamic_query}'...")
    snovio_token = None
    final_profiles = []

    try:
        response = requests.post(url, headers=headers, data=payload)
        
        if response.status_code == 200:
            results = response.json()
            organic_results = results.get("organic", [])
            
            print(f"📡 Étape 2 : Traitement en cascade (Hunter -> Snovio -> Tomba -> AeroLeads)")
            
            for item in organic_results[:3]:
                title_raw = item.get("title", "")
                link = item.get("link", "")
                
                raw_full_name = title_raw.split(" - ")[0]
                clean_full_name = clean_linkedin_name(raw_full_name)
                first_name, last_name = split_name(clean_full_name)
                
                # Variables par défaut
                final_email, final_position = "Non trouvé", "Non trouvé"
                final_phone = "Non trouvé"
                final_aeroleads_raw_data = None
                source_used = "Serper"
                
                # --- NIVEAU 1 : HUNTER ---
                h_email, h_position = get_hunter_data(first_name, last_name, domain)
                if h_email != "Non trouvé" or h_position != "Non trouvé":
                    final_email = h_email
                    final_position = h_position
                    source_used += " + Hunter.io"
                
                # --- NIVEAU 2 : SNOV.IO ---
                if final_email == "Non trouvé" or final_position == "Non trouvé":
                    if not snovio_token: snovio_token = get_snovio_token()
                    snov_email, snov_position = get_snovio_data(first_name, last_name, domain, snovio_token)
                    
                    updated = False
                    if final_email == "Non trouvé" and snov_email != "Non trouvé": final_email, updated = snov_email, True
                    if final_position == "Non trouvé" and snov_position != "Non trouvé": final_position, updated = snov_position, True
                    if updated: source_used += " + Snov.io"

                # --- NIVEAU 3 : TOMBA ---
                if final_email == "Non trouvé" or final_position == "Non trouvé" or final_phone == "Non trouvé":
                    tomba_email, tomba_position, tomba_phone = get_tomba_data(first_name, last_name, domain)
                    
                    updated = False
                    if final_email == "Non trouvé" and tomba_email != "Non trouvé": final_email, updated = tomba_email, True
                    if final_position == "Non trouvé" and tomba_position != "Non trouvé": final_position, updated = tomba_position, True
                    if final_phone == "Non trouvé" and tomba_phone != "Non trouvé": final_phone, updated = tomba_phone, True
                    if updated: source_used += " + Tomba.io"

                # --- NIVEAU 4 : AEROLEADS (Exécuté SYSTÉMATIQUEMENT pour récupérer le profil profond) ---
                aero_email, aero_position, aero_phone, aero_raw = get_aeroleads_data(first_name, last_name, company_name, link)
                
                final_aeroleads_raw_data = aero_raw # Enregistrement de la donnée brute
                
                updated = False
                if final_email == "Non trouvé" and aero_email != "Non trouvé": final_email, updated = aero_email, True
                if final_position == "Non trouvé" and aero_position != "Non trouvé": final_position, updated = aero_position, True
                if final_phone == "Non trouvé" and aero_phone != "Non trouvé": final_phone, updated = aero_phone, True
                
                # Mise à jour de la source selon si on a bouché un trou ou juste enrichi le profil
                if updated: 
                    source_used += " + AeroLeads"
                elif aero_raw:
                    source_used += " + AeroLeads (Profil Enrichi)"

                # --- CONSTRUCTION DU PROFIL APLATI (FLATTENED) ---
                profile_data = {}
                
                # 1. On injecte toutes les données d'AeroLeads à la racine (Flattening)
                if isinstance(final_aeroleads_raw_data, dict):
                    # Parfois les API enveloppent les données dans une clé 'data' ou 'raw_data'
                    if "raw_data" in final_aeroleads_raw_data and isinstance(final_aeroleads_raw_data["raw_data"], dict):
                        profile_data.update(final_aeroleads_raw_data["raw_data"])
                    elif "data" in final_aeroleads_raw_data and isinstance(final_aeroleads_raw_data["data"], dict):
                        profile_data.update(final_aeroleads_raw_data["data"])
                    else:
                        profile_data.update(final_aeroleads_raw_data)
                
                # 2. On ajoute par-dessus nos attributs standardisés de la cascade.
                # Cela garantit que nos données consolidées écrasent les doublons d'AeroLeads
                profile_data["raw_name_from_google"] = raw_full_name
                profile_data["clean_name_used"] = clean_full_name
                profile_data["linkedin_url"] = link
                profile_data["company"] = company_name
                profile_data["email"] = final_email
                profile_data["title"] = final_position
                profile_data["phone"] = final_phone
                profile_data["source"] = source_used

                final_profiles.append(profile_data)
                
                time.sleep(1) # Sécurité Rate Limit pour les APIs

            # --- SAUVEGARDE DYNAMIQUE ---
            output_dir = "personas_discovered"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            safe_domain = domain.replace('.', '_')
            output_file = os.path.join(output_dir, f"{safe_domain}_personas.json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_profiles, f, indent=4, ensure_ascii=False)

            print(f"✅ Terminé ! {len(final_profiles)} profils testés et sauvegardés.")
            print(f"💾 Regarde le résultat dans : {output_file}")
            
            return final_profiles
            
        else:
            print(f"❌ Erreur Serper HTTP {response.status_code}")
            return []

    except Exception as e:
        print(f"❌ Erreur globale : {e}")
        return []

if __name__ == "__main__":
    TARGET_DOMAIN = "Tesla.com"
    TARGET_LOCATION = "USA"
    TARGET_ROLE = "IT" 
    
    profiles = search_and_enrich(TARGET_DOMAIN, TARGET_LOCATION, TARGET_ROLE)
    print(f"Profils retournés : {len(profiles)}")