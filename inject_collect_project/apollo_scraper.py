import requests

class ApolloScraper:
    def __init__(self, api_key):
        self.api_key = api_key
        # Endpoints officiels d'Apollo v1
        self.search_url = "https://api.apollo.io/v1/organizations/search"
        self.enrich_url = "https://api.apollo.io/v1/organizations/enrich"
        
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "X-Api-Key": self.api_key 
        }

    def search_companies(self, industries=None, locations=None, limit=5):
        """
        Phase 1 : Découverte (Search) avec Double-Mapping dynamique.
        """
        safe_limit = min(limit * 2, 25) # On en demande un peu plus pour anticiper les rejets
        
        data = {
            "organization_locations": locations if locations else [],
            "q_organization_keyword_tags": industries if industries else [],
            "per_page": safe_limit
        }
        
        # --- FEATURE 3 : DOUBLE-MAPPING DYNAMIQUE ---
        # Si une localisation est demandée, on l'injecte dans le nom pour forcer 
        # l'API à faire remonter les entités locales (ex: "Sofrecom Tunisia")
        if locations and len(locations) > 0:
            data["q_organization_name"] = locations[0]
            
        try:
            target_str = locations[0] if locations else "Global"
            print(f"📡 Apollo Search : Recherche dynamique pour [{target_str}]...")
            response = requests.post(self.search_url, json=data, headers=self.headers)
            
            if response.status_code == 200:
                results = response.json().get('organizations', [])
                return self._format_org_data(results[:limit], enriched=False)
            else:
                print(f"⚠️ Erreur Apollo Search {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Erreur lors du search : {e}")
            return []

    # MODIFICATION : Suppression de org_id des paramètres
    def enrich_organization(self, domain=None, target_location=None):
        """
        Phase 2 : Enrichissement basé uniquement sur le domaine avec Sanity Check.
        """
        # --- LOGIQUE STRICTE SUR LE DOMAINE ---
        valid_domain = domain and domain != "Non renseigné"

        if not valid_domain:
            print("⏩ Aucun domaine valide fourni. Enrichissement annulé.")
            return None

        params = {
            "domain": domain
        }
        identifier_used = domain
        # ------------------------------------------------------

        try:
            print(f"💎 Enrichissement profond pour : {identifier_used}...")
            response = requests.get(self.enrich_url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                org = response.json().get('organization', {})
                
                # --- FEATURE 2 & 4 : SANITY CHECK + SUBORGANIZATIONS ---
                if target_location:
                    actual_country = org.get('country', '').lower()
                    target = target_location.lower()
                    
                    # --- CORRECTION DE LA LOGIQUE HQ VS BRANCHE ---
                    if target in actual_country or actual_country in target or (target == "usa" and actual_country in ["united states", "us"]):
                        print(f"✅ {org.get('name')} est déjà localisée en {org.get('country')}. Acceptation du profil.")
                    
                    # Sinon, si le siège n'est PAS dans le pays cible
                    else:
                        print(f"⚠️ {org.get('name')} a son siège en {org.get('country')}. Recherche de la branche locale...")
                        
                        subs = org.get('suborganizations', [])
                        local_domain = None
                        
                        # On fouille les sous-organisations pour trouver le pays cible
                        for sub in subs:
                            sub_name = sub.get('name', '').lower()
                            if target in sub_name:
                                local_domain = sub.get('primary_domain')
                                break
                                
                        if local_domain and local_domain != domain:
                            print(f"🎯 Filiale locale identifiée : {local_domain}. Basculement automatique...")
                            # RECURSION : On relance l'enrichissement uniquement sur le domaine de la filiale
                            return self.enrich_organization(domain=local_domain, target_location=target_location)
                        else:
                            print(f"⏩ Aucune filiale '{target_location}' trouvée pour {identifier_used}. Profil ignoré.")
                            return None 
                # ---------------------------------------------------------

                return self._format_org_data([org], enriched=True)[0]
            else:
                print(f"⚠️ Erreur Apollo Enrichment {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Erreur lors de l'enrichissement : {e}")
            return None

    def _format_org_data(self, results, enriched=False):
        """
        Nettoyeur centralisé pour structurer la donnée avant l'insertion Neo4j.
        """
        formatted_list = []
        default_val = "Non renseigné"

        for org in results:
            def clean(val):
                return val if val not in [None, "", [], {}] else default_val

            # --- 1. IDENTITÉ & SEO ---
            data = {
                # On conserve la récupération de l'ID au cas où tu en as besoin pour ta BDD, 
                # même s'il n'est plus utilisé pour chercher.
                "apollo_id": clean(org.get("id")),
                "domain": clean(org.get("primary_domain")),
                "name": clean(org.get("name")),
                "industry": clean(org.get("industry")),
                "founded_year": clean(org.get("founded_year")),
                "logo_url": clean(org.get("logo_url")),
                "website_url": clean(org.get("website_url")),
                "short_description": clean(org.get("short_description")),
                "seo_description": clean(org.get("seo_description")),
                
                # --- 2. PERFORMANCE & CHIFFRES ---
                "alexa_ranking": clean(org.get("alexa_ranking")),
                "annual_revenue": clean(org.get("annual_revenue")),
                "total_funding": clean(org.get("total_funding_printed")),
                "estimated_num_employees": clean(org.get("estimated_num_employees")),
                "latest_funding_stage": clean(org.get("latest_funding_stage")),

                # --- 3. RÉSEAUX SOCIAUX & CONTACT ---
                "linkedin_url": clean(org.get("linkedin_url")),
                "twitter_url": clean(org.get("twitter_url")),
                "facebook_url": clean(org.get("facebook_url")),
                "crunchbase_url": clean(org.get("crunchbase_url")),
                "phone": clean(org.get("phone")),

                # --- 4. LOCALISATION DÉTAILLÉE ---
                "location": {
                    "raw_address": clean(org.get("raw_address")),
                    "street_address": clean(org.get("street_address")),
                    "city": clean(org.get("city")),
                    "state": clean(org.get("state")),
                    "postal_code": clean(org.get("postal_code")),
                    "country": clean(org.get("country"))
                },

                # --- 5. STRUCTURE HIÉRARCHIQUE ---
                "hierarchy": {
                    "num_suborganizations": org.get("num_suborganizations") or 0,
                    "owned_by_organization_id": clean(org.get("owned_by_organization_id")),
                }
            }

            # --- 6. DONNÉES COMPLEXES (Seulement si enrichi) ---
            if enriched:
                # Technologies
                techs = org.get("current_technologies", [])
                data["technologies"] = [
                    {
                        "uid": clean(t.get("uid")),
                        "name": clean(t.get("name")),
                        "category": clean(t.get("category"))
                    } for t in techs
                ] if techs else []

                # Headcount par Département
                data["departments"] = org.get("departmental_head_count", {})

                # Historique de Financement
                fundings = org.get("funding_events", [])
                data["funding_events"] = [
                    {
                        "id": clean(fe.get("id")),
                        "date": clean(fe.get("date")),
                        "type": clean(fe.get("type")),
                        "amount": clean(fe.get("amount")),
                        "currency": clean(fe.get("currency")),
                        "investors": clean(fe.get("investors"))
                    } for fe in fundings
                ] if fundings else []

                # Liste détaillée des sous-organisations
                subs = org.get("suborganizations", [])
                data["suborganizations"] = [
                    {
                        "id": clean(s.get("id")),
                        "name": clean(s.get("name")),
                        "domain": clean(s.get("primary_domain"))
                    } for s in subs
                ] if subs else []

                # Mots-clés (Keywords)
                data["keywords"] = clean(org.get("keywords"))

            formatted_list.append(data)
            
        return formatted_list