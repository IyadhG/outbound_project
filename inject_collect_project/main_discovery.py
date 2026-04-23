import os
import json
from apollo_scraper import ApolloScraper
from database_manager import Neo4jManager
from smart_scraper_ai import SmartScraperAI
from persona_search_enrich import search_and_enrich  # <-- Ajout de ton module

# Ta clé API
APOLLO_KEY = "lwjI_IpYk0S28Lixu1MsXw"

def generate_merged_report(apollo_data, ai_json_path, output_dir="merged_profiles"):
    """
    Fusionne les données Apollo et SmartScraper AI en un seul profil unifié.
    Règles : 
    - Attributs exclusifs : conservés (Neo4j ou AI).
    - Si BDD est null mais IA a une valeur : l'IA est acceptée quel que soit son score.
    - Attributs communs (présents dans les 2) : l'IA remplace Apollo UNIQUEMENT SI confidence >= 0.8.
    - Attributs listes (keywords, technologies, suborganizations, funding_events) : Union unique des deux sources.
    - Score de qualité (DQS) : Recalculé selon la complétude, la synergie des sources et la confiance IA.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(ai_json_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de lire le fichier IA pour la fusion : {e}")
        return None

    domain = apollo_data.get("domain", "unknown_domain")
    CONFIDENCE_THRESHOLD = 0.8

    # --- HELPERS D'EXTRACTION & PARSING ---
    def get_ai_val_and_conf(category, field=None):
        """Retourne un tuple (valeur, confidence) depuis l'IA"""
        if field:
            target = ai_data.get(category, {}).get(field)
            if isinstance(target, dict):
                return target.get("value"), target.get("confidence", 0.0)
            return target, 0.0
        else:
            val = ai_data.get(category)
            if isinstance(val, dict) and "value" in val:
                return val.get("value"), val.get("confidence", 0.0)
            return val, 0.0

    def get_db_val(key):
        """Récupère la donnée depuis Apollo/Neo4j de manière sécurisée"""
        if key in apollo_data:
            return apollo_data[key]
        if key in ["city", "country", "raw_address", "state", "street_address", "postal_code"]:
            return apollo_data.get("location", {}).get(key, "Non renseigné")
        if key == "num_suborganizations":
            return apollo_data.get("hierarchy", {}).get(key, "Non renseigné")
        return "Non renseigné"

    def is_valid(val):
        """Vérifie si une donnée contient réellement de l'information"""
        invalid_strings = [
            None, "", "Non renseigné", "Non trouvé", "invalid or hallucinated link", 
            "Non trouvé par l'IA", "Non trouvé pour cette entité spécifique", "[]", "{}"
        ]
        if val in invalid_strings:
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        if isinstance(val, dict) and len(val) == 0:
            return False
        return True

    def parse_list_safely(val):
        """Convertit les strings JSON de Neo4j (ex: "[{...}]") en vraies listes Python"""
        if not is_valid(val):
            return []
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return parsed
            except:
                # Si c'est juste une string séparée par des virgules (ex: keywords de l'IA)
                if "[" not in val and "{" not in val:
                    return [v.strip() for v in val.split(',') if v.strip()]
                return []
        if isinstance(val, list):
            return val
        return []

    def calculate_final_quality_score(merged_data, ai_original_score):
        """Calcule le nouveau Data Quality Score basé sur Complétude, Synergie et Confiance IA"""
        # 1. Liste des champs critiques pour la qualité
        critical_fields = ["name", "domain", "industry", "country", "technologies", "linkedin_url"]
        
        # 2. Calcul de la complétude (0.0 à 1.0)
        filled_fields = [f for f in critical_fields if is_valid(merged_data.get(f))]
        completeness_score = len(filled_fields) / len(critical_fields)
        
        # 3. Bonus de synergie
        has_apollo = is_valid(merged_data.get("apollo_id"))
        has_ai = ai_original_score > 0
        
        synergy_score = 1.0 if (has_apollo and has_ai) else 0.5 if (has_apollo or has_ai) else 0.0

        # 4. Score final pondéré (50% Complétude, 30% Synergie, 20% IA)
        final_score = (completeness_score * 0.5) + (synergy_score * 0.3) + (ai_original_score * 0.2)
        
        return round(final_score, 2)

    # --- DICTIONNAIRE FINAL FUSIONNÉ ---
    merged_profile = {}

    # 1. LISTE DE TOUS LES ATTRIBUTS NEO4J
    all_db_attributes = [
        "alexa_ranking", "annual_revenue", "apollo_id", "captured_at", "category", "city", 
        "country", "created_at", "crunchbase", "crunchbase_url", "data", "departments", 
        "description", "domain", "estimated_num_employees", "facebook", "facebook_url", 
        "founded_year", "full_address", "funding_events", "funding_stage", "id", "industry", 
        "keywords", "last_update", "last_verified_at", "latest_funding_stage", "linkedin", 
        "linkedin_url", "logo", "logo_url", "name", "nodes", "num_suborganizations", 
        "owned_by_organization_id", "parent_apollo_id", "parent_id", "phone", "postal_code", 
        "raw_address", "relationships", "revenue", "seo_description", "short_description", 
        "siret", "size", "state", "street_address", "style", "sub_orgs_count", 
        "suborganizations", "tech_stack", "technologies", "total_funding", "twitter", 
        "twitter_url", "uid", "visualisation", "website", "website_url"
    ]

    # MAPPING DES ATTRIBUTS COMMUNS (db_key -> ai_category, ai_field)
    common_mapping = {
        "domain": ("identity", "domain"),
        "name": ("identity", "name"),
        "industry": ("identity", "industry"),
        "founded_year": ("identity", "founded_year"),
        "short_description": ("identity", "short_description"),
        "seo_description": ("identity", "seo_description"),
        "annual_revenue": ("performance", "annual_revenue_USD"),
        "total_funding": ("performance", "total_funding"),
        "estimated_num_employees": ("performance", "estimated_num_employees"),
        "latest_funding_stage": ("performance", "latest_funding_stage"),
        "linkedin_url": ("contact_social", "linkedin_url"),
        "twitter_url": ("contact_social", "twitter_url"),
        "facebook_url": ("contact_social", "facebook_url"),
        "phone": ("contact_social", "phone"),
        "num_suborganizations": ("hierarchy", "num_suborganizations"),
        "raw_address": ("location_detailed", "raw_address"),
        "city": ("location_detailed", "city"),
        "country": ("location_detailed", "country")
    }

    # --- 2. TRAITEMENT DE TOUS LES ATTRIBUTS NEO4J (COMMUNS ET EXCLUSIFS) ---
    for attr in all_db_attributes:
        db_val = get_db_val(attr)
        db_valid = is_valid(db_val)

        # ====== CAS SPÉCIAL 1 : FUSION DES TECHNOLOGIES (Structure Unifiée) ======
        if attr == "technologies":
            db_techs = parse_list_safely(db_val)
            ai_techs = ai_data.get("technologies", [])
            
            if db_valid or len(ai_techs) > 0:
                merged_techs = {}
                
                if db_valid:
                    for t in db_techs:
                        if isinstance(t, dict) and "name" in t:
                            tech_name = t["name"].lower()
                            merged_techs[tech_name] = {
                                "uid": t.get("uid", tech_name.replace(" ", "_")),
                                "name": t.get("name"),
                                "category": t.get("category", "Other")
                            }
                            
                for t in ai_techs:
                    if isinstance(t, dict) and "name" in t:
                        tech_name = t["name"].lower()
                        if tech_name not in merged_techs:
                            generated_uid = t.get("uid", tech_name.replace(" ", "_").replace("-", "_"))
                            merged_techs[tech_name] = {
                                "uid": generated_uid,
                                "name": t.get("name"),
                                "category": t.get("category", "Other")
                            }
                            
                merged_profile[attr] = list(merged_techs.values())
            else:
                merged_profile[attr] = db_val

        # ====== CAS SPÉCIAL 2 : FUSION DES KEYWORDS ======
        elif attr == "keywords":
            db_kws = parse_list_safely(db_val)
            ai_kw_raw, _ = get_ai_val_and_conf("keywords", None)
            ai_kws = parse_list_safely(ai_kw_raw)

            if db_valid and len(ai_kws) > 0:
                unique_kws = list(db_kws)
                lower_db_kws = {str(k).lower().strip() for k in db_kws}
                for k in ai_kws:
                    if str(k).lower().strip() not in lower_db_kws:
                        unique_kws.append(k)
                merged_profile[attr] = unique_kws
            elif len(ai_kws) > 0:
                merged_profile[attr] = ai_kws
            else:
                merged_profile[attr] = db_val

        # ====== CAS SPÉCIAL 3 : FUSION DES SUBORGANIZATIONS (Structure Unifiée) ======
        elif attr == "suborganizations":
            db_subs = parse_list_safely(db_val)
            ai_subs_raw, _ = get_ai_val_and_conf("hierarchy", "subsidiaries_list")
            ai_subs = parse_list_safely(ai_subs_raw)

            if db_valid or len(ai_subs) > 0:
                merged_subs = {}
                
                if db_valid:
                    for s in db_subs:
                        if isinstance(s, dict) and "name" in s:
                            sub_name = str(s["name"]).lower().strip()
                            merged_subs[sub_name] = {
                                "id": s.get("id", "Non renseigné"),
                                "name": str(s.get("name")),
                                "domain": s.get("domain", "Non renseigné")
                            }
                        elif isinstance(s, str):
                            sub_name = s.lower().strip()
                            merged_subs[sub_name] = {
                                "id": "Non renseigné",
                                "name": s,
                                "domain": "Non renseigné"
                            }

                for s in ai_subs:
                    if isinstance(s, dict) and "name" in s:
                        sub_name = str(s["name"]).lower().strip()
                        if sub_name not in merged_subs:
                            merged_subs[sub_name] = {
                                "id": s.get("id", "Non renseigné"),
                                "name": str(s.get("name")),
                                "domain": s.get("domain", "Non renseigné")
                            }
                    elif isinstance(s, str):
                        sub_name = s.lower().strip()
                        if sub_name not in merged_subs:
                            merged_subs[sub_name] = {
                                "id": "Non renseigné",
                                "name": s,
                                "domain": "Non renseigné"
                            }
                            
                merged_profile[attr] = list(merged_subs.values())
            else:
                merged_profile[attr] = db_val

        # ====== CAS SPÉCIAL 4 : FUSION DES FUNDING EVENTS ======
        elif attr == "funding_events":
            db_fundings = parse_list_safely(db_val)
            ai_fundings = ai_data.get("funding_events", [])

            if db_valid or len(ai_fundings) > 0:
                merged_fundings = []
                seen_signatures = set()

                def get_signature(f_dict):
                    """Crée une signature 'Année_Type' pour éviter les doublons grossiers"""
                    date_str = str(f_dict.get("date", "")).strip()
                    year = date_str[:4] if len(date_str) >= 4 else date_str
                    f_type = str(f_dict.get("type", "")).strip().lower()
                    return f"{year}_{f_type}"

                def normalize_funding(f_dict):
                    """Garantit que tous les attributs sont présents avec des valeurs par défaut"""
                    return {
                        "id": f_dict.get("id", "Non renseigné"),
                        "date": f_dict.get("date", "Non renseigné"),
                        "type": f_dict.get("type", "Non renseigné"),
                        "amount": f_dict.get("amount", "Non renseigné"),
                        "currency": f_dict.get("currency", "Non renseigné"),
                        "investors": f_dict.get("investors", "Non renseigné")
                    }

                # Ajout de la BDD Apollo
                for f in db_fundings:
                    if isinstance(f, dict):
                        sig = get_signature(f)
                        normalized_f = normalize_funding(f)
                        if sig == "_" or sig not in seen_signatures:
                            merged_fundings.append(normalized_f)
                            if sig != "_":
                                seen_signatures.add(sig)

                # Ajout de l'IA
                for f in ai_fundings:
                    if isinstance(f, dict):
                        sig = get_signature(f)
                        normalized_f = normalize_funding(f)
                        if sig == "_" or sig not in seen_signatures:
                            merged_fundings.append(normalized_f)
                            if sig != "_":
                                seen_signatures.add(sig)

                merged_profile[attr] = merged_fundings
            else:
                merged_profile[attr] = []

        # ====== CAS STANDARD : REMPLACEMENT SOUS CONDITION DE CONFIANCE ======
        elif attr in common_mapping:
            ai_cat, ai_field = common_mapping[attr]
            ai_val, ai_conf = get_ai_val_and_conf(ai_cat, ai_field)
            
            ai_valid = is_valid(ai_val)

            # RÈGLE 1 : BDD Null MAIS l'IA a trouvé l'info -> On prend l'IA
            if not db_valid and ai_valid:
                merged_profile[attr] = ai_val
                
            # RÈGLE 2 : Les deux sources ont l'info -> On applique le seuil de l'IA
            elif db_valid and ai_valid:
                if ai_conf >= CONFIDENCE_THRESHOLD:
                    merged_profile[attr] = ai_val
                else:
                    merged_profile[attr] = db_val
                    
            # RÈGLE 3 : Seule la BDD a l'info
            elif db_valid and not ai_valid:
                merged_profile[attr] = db_val
                
            # RÈGLE 4 : Personne n'a l'info
            else:
                merged_profile[attr] = None

        # ====== ATTRIBUTS EXCLUSIFS A APOLLO/NEO4J ======
        else:
            if db_valid:
                merged_profile[attr] = db_val

    # --- 3. AJOUT DES ATTRIBUTS EXCLUSIFS IA ---
    ai_only_mapping = {
        "fiscal_year_end": ("performance", "fiscal_year_end"),
        "is_subsidiary": ("hierarchy", "is_subsidiary"),
        "parent_company": ("hierarchy", "parent_company")
        # On exclut volontairement 'data_quality_score' ici pour le calculer juste après
    }

    for ai_key, (ai_cat, ai_field) in ai_only_mapping.items():
        ai_val, _ = get_ai_val_and_conf(ai_cat, ai_field)
        if is_valid(ai_val):
            merged_profile[ai_key] = ai_val

    # Extraction liste concurrents (exclusif IA)
    competitors = ai_data.get("market_intelligence", {}).get("competitors", [])
    if is_valid(competitors):
        merged_profile["competitors"] = competitors

    # --- 3.5 CALCUL DU NOUVEAU SCORE DE QUALITÉ DES DONNÉES (DQS) ---
    initial_ai_score_raw, _ = get_ai_val_and_conf("data_quality_score", None)
    try:
        initial_ai_score = float(initial_ai_score_raw) if initial_ai_score_raw else 0.0
    except ValueError:
        initial_ai_score = 0.0
        
    merged_profile["data_quality_score"] = calculate_final_quality_score(merged_profile, initial_ai_score)

    # --- 4. SAUVEGARDE DU FICHIER UNIFIÉ ---
    safe_domain = domain.replace(".", "_")
    output_file = os.path.join(output_dir, f"{safe_domain}_MERGED.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_profile, f, indent=4, ensure_ascii=False)
        
    print(f"📊 Profil consolidé avec succès (Union + Seuil Confiance + Schemas Unifiés) : {output_file}")
    
    return merged_profile


def discover_and_inject():
    # --- CONFIGURATION DYNAMIQUE ---
    TARGET_INDUSTRY = "It"
    TARGET_LOCATION = "Spain"
    MAX_COMPANIES_TO_GET = 2  

    scraper = ApolloScraper(APOLLO_KEY)
    db = Neo4jManager()
    
    # Initialisation du scraper IA
    ai_scraper = SmartScraperAI() 

    # Dictionnaire pour stocker temporairement les personas à injecter
    all_discovered_personas = {}

    # 1. PHASE DE DÉCOUVERTE (SEARCH)
    print(f"🔎 Recherche initiale de {MAX_COMPANIES_TO_GET} entreprises en {TARGET_LOCATION}...")
    basic_companies = scraper.search_companies(
        industries=[TARGET_INDUSTRY],
        locations=[TARGET_LOCATION],
        limit=MAX_COMPANIES_TO_GET
    )

    if not basic_companies:
        print("❌ Résultats de recherche vides. Vérifiez les filtres ou la clé API.")
        db.close()
        return

    # 2. PHASE D'ENRICHISSEMENT & SCRAPING IA
    apollo_batch = []
    merged_batch = []
    
    print(f"💎 Enrichissement profond de {len(basic_companies)} entreprises...")

    for company in basic_companies:
        # --- RÉCUPÉRATION DU DOMAINE ---
        domain = company.get('domain')
        valid_domain = domain and domain != "Non renseigné"

        # On initialise avec les données de base (sera écrasé si l'enrichissement réussit)
        full_data = company
        merged_data = None

        # --- ENRICHISSEMENT APOLLO (Seulement si domaine valide) ---
        if valid_domain:
            print(f"💎 Tentative d'enrichissement pour le domaine : {domain}...")
            try:
                # Appel de l'enrichissement en utilisant uniquement le domaine
                enriched_data = scraper.enrich_organization(domain=domain, target_location=TARGET_LOCATION)
                if enriched_data:
                    full_data = enriched_data
                else:
                    print(f"🟠 Fallback : Enrichissement échoué. Utilisation des données basiques.")
            except Exception as e:
                print(f"⚠️ Erreur API lors de l'enrichissement : {e}")
                print(f"🟠 Fallback : Utilisation des données basiques.")
        else:
            print(f"⏩ Aucun domaine pour {company.get('name', 'Compagnie inconnue')}. Enrichissement Apollo ignoré.")

        # --- ÉTAPE : SMART SCRAPER AI ---
        # Récupération de l'URL pour l'envoyer à l'IA
        website_url = full_data.get('website_url') or full_data.get('website')
        
        # Extraction de l'adresse pour aider l'IA à cibler le bon HQ
        loc = full_data.get("location", {})
        street = loc.get("street_address", "")
        city = loc.get("city", "")
        country = loc.get("country", "")
        
        addr_parts = [p for p in [street, city, country] if p and p != "Non renseigné"]
        target_addr = ", ".join(addr_parts)

        # On lance l'IA seulement si on a un domaine ET une URL valide
        if valid_domain and website_url and website_url != "Non renseigné" and website_url.startswith("http"):
            print(f"🤖 Lancement de SmartScraperAI pour le site : {website_url}")
            if target_addr:
                print(f"📍 Cible géographique transmise à l'IA : {target_addr}")
                
            try:
                path_to_json = ai_scraper.scrape_and_save(website_url, target_address=target_addr) 
                
                if path_to_json and path_to_json.endswith(".json"):
                    print(f"✅ Scraping IA terminé. Fichier généré : {path_to_json}")
                    
                    # --- APPEL DE LA FONCTION DE FUSION ---
                    merged_data = generate_merged_report(full_data, path_to_json)
                    
                else:
                    print(f"⚠️ Le scraping IA n'a pas retourné de chemin valide ou de JSON.")
                    
            except Exception as e:
                print(f"❌ Erreur lors du scraping IA de {website_url} : {e}")
        else:
            if not valid_domain:
                print(f"⚠️ SmartScraperAI ignoré car le domaine est manquant.")
            else:
                print(f"⚠️ Pas d'URL valide pour {full_data.get('name', 'cette entreprise')}, SmartScraperAI ignoré.")

        # --- RECHERCHE DE PERSONAS (SALES) ---
        if valid_domain:
            target_obj = merged_data if merged_data else full_data
            
            # Déterminer le pays à utiliser pour la recherche Serper
            company_country = target_obj.get("country")
            if not company_country or company_country == "Non renseigné":
                company_country = target_obj.get("location", {}).get("country", TARGET_LOCATION)
                
            print(f"👤 Lancement de la recherche de personas (Sales) pour {domain} en {company_country}...")
            try:
                # L'appel à la fonction gère la sauvegarde locale en JSON
                personas_found = search_and_enrich(domain=domain, location=company_country, role="Sales")
                
                if personas_found:
                    # On stocke temporairement pour les injecter APRES la création de l'entreprise
                    all_discovered_personas[domain] = personas_found
                    print(f"👥 {len(personas_found)} personas découverts et mis en attente pour l'injection Neo4j.")
                else:
                    print(f"ℹ️ Aucun persona trouvé pour {domain}.")
                
            except Exception as e:
                print(f"❌ Erreur lors de la recherche des personas : {e}")
                
            # Mise à jour de l'objet source pour l'injection (sans les personas imbriqués)
            if merged_data:
                merged_data = target_obj
            else:
                full_data = target_obj

        # --- FILTRAGE DE SÉCURITÉ (Anti-Collision) ---
        # Si l'entreprise n'a ni domaine ni ID, on lui crée un domaine fictif unique
        # pour éviter qu'elles ne fusionnent toutes sous "Non renseigné" dans Neo4j.
        current_domain = full_data.get('domain')
        current_id = full_data.get('apollo_id')
        if (not current_domain or current_domain == "Non renseigné") and (not current_id or current_id == "Non renseigné"):
            safe_name = str(full_data.get('name', 'unknown')).replace(' ', '_').lower()
            full_data['domain'] = f"unknown_{safe_name}"
            print(f"🔧 Correction Anti-Collision appliquée : domaine fictif généré -> {full_data['domain']}")

        # --- PRÉPARATION DES BATCHS POUR INJECTION ---
        print(f"💾 Préparation de l'injection en base pour : {full_data.get('name', 'Compagnie inconnue')}")
        if merged_data:
            merged_batch.append(merged_data)
        else:
            apollo_batch.append(full_data)

    # 3. INJECTION DANS NEO4J (ENTREPRISES D'ABORD)
    if apollo_batch:
        print(f"🚀 Injection de {len(apollo_batch)} profils classiques (Apollo)...")
        db.bulk_import_companies(apollo_batch)
        
    if merged_batch:
        print(f"🚀 Injection de {len(merged_batch)} profils fusionnés (Apollo+IA)...")
        db.import_merged_profiles(merged_batch)
        
    if not apollo_batch and not merged_batch:
        print("⚠️ Aucun profil n'a été conservé pour l'injection Neo4j.")

    # 4. INJECTION DANS NEO4J (PERSONAS ENSUITE POUR SÉCURISER LES RELATIONS)
    if all_discovered_personas:
        print(f"\n🚀 Injection des personas en base de données pour {len(all_discovered_personas)} entreprises...")
        for comp_domain, personas_list in all_discovered_personas.items():
            db.import_personas(personas_list, comp_domain)
    
    db.close()
    print("✅ Processus complet terminé avec succès.")

if __name__ == "__main__":
    discover_and_inject()