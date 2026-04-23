from neo4j import GraphDatabase
import sys
import json

class Neo4jManager:
    def __init__(self):
        # Configuration des accès AuraDB
        self.uri = "neo4j+s://151c0242.databases.neo4j.io"
        self.user = "151c0242" 
        self.password = "1b56m_IXQlXOlENuIQmkjOswv-CddFai8TACtl5JXXo" 
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("✅ Connectivité Neo4j établie.")
        except Exception as e:
            print(f"❌ Erreur de connexion : {e}")
            sys.exit(1)

    def bulk_import_companies(self, companies_list):
        """
        Importation avec Versioning Exhaustif.
        Synchronisé à 100% avec les sorties du scraper Apollo.
        """
        default_msg = "Non renseigné"

        # --- 1. PRÉPARATION DES DONNÉES (Sérialisation JSON) ---
        for company in companies_list:
            for key in ["departments", "funding_events", "suborganizations", "technologies"]:
                if key in company and not isinstance(company[key], str):
                    company[key] = json.dumps(company[key])

        query = """
        UNWIND $batch AS data
        
        // 1. CRÉATION DU DOMAINE DE SÉCURITÉ (ANTI-COLLISION)
        WITH data,
             CASE 
                 WHEN data.domain IS NULL OR data.domain = 'Non renseigné' OR trim(data.domain) = ''
                 THEN 'unknown_' + replace(toLower(COALESCE(data.name, 'unknown')), ' ', '_') + '_' + replace(toLower(COALESCE(data.country, data.location.country, 'unknown')), ' ', '_')
                 ELSE data.domain
             END AS safe_domain

        // 2. Identification ou création de l'entreprise racine
        MERGE (c:Company {domain: safe_domain})
        SET c.uid = COALESCE(data.apollo_id, safe_domain),
            c.name = COALESCE(data.name, c.name, $msg),
            c.apollo_id = COALESCE(data.apollo_id, c.apollo_id)
        
        // 3. Récupération de la version actuelle
        OPTIONAL MATCH (c)-[old_rel:CURRENT]->(old_v:Version)
        
        // 4. LOGIQUE DE COMPARAISON EXHAUSTIVE (Correction du bug de fallback $msg)
        WITH c, data, old_rel, old_v, safe_domain,
             (old_v IS NULL 
              // Identité & SEO
              OR COALESCE(old_v.name, $msg) <> COALESCE(data.name, $msg)
              OR COALESCE(old_v.industry, $msg) <> COALESCE(data.industry, $msg)
              OR COALESCE(old_v.founded_year, $msg) <> COALESCE(data.founded_year, $msg)
              OR COALESCE(old_v.logo_url, $msg) <> COALESCE(data.logo_url, $msg)
              OR COALESCE(old_v.website_url, $msg) <> COALESCE(data.website_url, $msg)
              OR COALESCE(old_v.short_description, $msg) <> COALESCE(data.short_description, $msg)
              OR COALESCE(old_v.seo_description, $msg) <> COALESCE(data.seo_description, $msg)
              
              // Performance & Chiffres
              OR COALESCE(old_v.alexa_ranking, $msg) <> COALESCE(data.alexa_ranking, $msg)
              OR COALESCE(old_v.annual_revenue, $msg) <> COALESCE(data.annual_revenue, $msg)
              OR COALESCE(old_v.total_funding, $msg) <> COALESCE(data.total_funding, $msg)
              OR COALESCE(old_v.estimated_num_employees, $msg) <> COALESCE(data.estimated_num_employees, $msg)
              OR COALESCE(old_v.latest_funding_stage, $msg) <> COALESCE(data.latest_funding_stage, $msg)
              
              // Réseaux Sociaux & Contact
              OR COALESCE(old_v.linkedin_url, $msg) <> COALESCE(data.linkedin_url, $msg)
              OR COALESCE(old_v.twitter_url, $msg) <> COALESCE(data.twitter_url, $msg)
              OR COALESCE(old_v.facebook_url, $msg) <> COALESCE(data.facebook_url, $msg)
              OR COALESCE(old_v.crunchbase_url, $msg) <> COALESCE(data.crunchbase_url, $msg)
              OR COALESCE(old_v.phone, $msg) <> COALESCE(data.phone, $msg)
              
              // Localisation
              OR COALESCE(old_v.raw_address, $msg) <> COALESCE(data.location.raw_address, $msg)
              OR COALESCE(old_v.street_address, $msg) <> COALESCE(data.location.street_address, $msg)
              OR COALESCE(old_v.city, $msg) <> COALESCE(data.location.city, $msg)
              OR COALESCE(old_v.state, $msg) <> COALESCE(data.location.state, $msg)
              OR COALESCE(old_v.postal_code, $msg) <> COALESCE(data.location.postal_code, $msg)
              OR COALESCE(old_v.country, $msg) <> COALESCE(data.location.country, $msg)
              
              // Structure Hiérarchique
              OR COALESCE(old_v.num_suborganizations, 0) <> COALESCE(data.hierarchy.num_suborganizations, 0)
              OR COALESCE(old_v.owned_by_organization_id, $msg) <> COALESCE(data.hierarchy.owned_by_organization_id, $msg)
              
              // Données Complexes
              OR COALESCE(old_v.technologies, "[]") <> COALESCE(data.technologies, "[]")
              OR COALESCE(old_v.departments, "{}") <> COALESCE(data.departments, "{}")
              OR COALESCE(old_v.funding_events, "[]") <> COALESCE(data.funding_events, "[]")
              OR COALESCE(old_v.suborganizations, "[]") <> COALESCE(data.suborganizations, "[]")
              OR COALESCE(old_v.keywords, []) <> COALESCE(data.keywords, [])
             ) AS has_changed

        // 5. ARCHIVAGE UNIQUEMENT SI CHANGEMENT
        FOREACH (_ IN CASE WHEN old_rel IS NOT NULL AND has_changed THEN [1] ELSE [] END |
            DELETE old_rel
            MERGE (c)-[:ARCHIVED]->(old_v)
        )

        // 6. CRÉATION NOUVELLE VERSION UNIQUEMENT SI CHANGEMENT
        FOREACH (_ IN CASE WHEN has_changed THEN [1] ELSE [] END |
            CREATE (new_v:Version)
            SET new_v.captured_at = datetime(),
                
                new_v.name = COALESCE(data.name, $msg),
                new_v.industry = COALESCE(data.industry, $msg),
                new_v.founded_year = COALESCE(data.founded_year, $msg),
                new_v.logo_url = COALESCE(data.logo_url, $msg),
                new_v.website_url = COALESCE(data.website_url, $msg),
                new_v.short_description = COALESCE(data.short_description, $msg),
                new_v.seo_description = COALESCE(data.seo_description, $msg),
                new_v.alexa_ranking = COALESCE(data.alexa_ranking, $msg),
                new_v.annual_revenue = COALESCE(data.annual_revenue, $msg),
                new_v.total_funding = COALESCE(data.total_funding, $msg),
                new_v.estimated_num_employees = COALESCE(data.estimated_num_employees, $msg),
                new_v.latest_funding_stage = COALESCE(data.latest_funding_stage, $msg),
                new_v.linkedin_url = COALESCE(data.linkedin_url, $msg),
                new_v.twitter_url = COALESCE(data.twitter_url, $msg),
                new_v.facebook_url = COALESCE(data.facebook_url, $msg),
                new_v.crunchbase_url = COALESCE(data.crunchbase_url, $msg),
                new_v.phone = COALESCE(data.phone, $msg),
                new_v.raw_address = COALESCE(data.location.raw_address, $msg),
                new_v.street_address = COALESCE(data.location.street_address, $msg),
                new_v.city = COALESCE(data.location.city, $msg),
                new_v.state = COALESCE(data.location.state, $msg),
                new_v.postal_code = COALESCE(data.location.postal_code, $msg),
                new_v.country = COALESCE(data.location.country, $msg),
                new_v.num_suborganizations = COALESCE(data.hierarchy.num_suborganizations, 0),
                new_v.owned_by_organization_id = COALESCE(data.hierarchy.owned_by_organization_id, $msg),
                new_v.technologies = COALESCE(data.technologies, "[]"),
                new_v.departments = COALESCE(data.departments, "{}"),
                new_v.funding_events = COALESCE(data.funding_events, "[]"),
                new_v.suborganizations = COALESCE(data.suborganizations, "[]"),
                new_v.keywords = COALESCE(data.keywords, [])
            
            CREATE (c)-[:CURRENT]->(new_v)
        )

        // 7. MISE À JOUR DU TIMESTAMP SI LES DONNÉES SONT IDENTIQUES
        FOREACH (_ IN CASE WHEN NOT has_changed THEN [1] ELSE [] END |
            SET old_v.last_verified_at = datetime()
        )

        // 8. HIÉRARCHIE (Relations entre Company)
        WITH data, safe_domain
        WHERE data.hierarchy.owned_by_organization_id IS NOT NULL 
          AND data.hierarchy.owned_by_organization_id <> "Non renseigné"
        
        MATCH (parent:Company {apollo_id: data.hierarchy.owned_by_organization_id})
        MATCH (child:Company {domain: safe_domain})
        WHERE parent <> child
        MERGE (parent)-[:OWNS]->(child)
        MERGE (child)-[:OWNED_BY]->(parent)
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, batch=companies_list, msg=default_msg)
                print(f"🚀 Graph Engine : {len(companies_list)} profils synchronisés avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l'import : {e}")

    def import_merged_profiles(self, merged_list):
        """
        Importation des profils fusionnés (Apollo + AI).
        Maintient les anciens attributs, met à jour les existants, et ajoute les nouveaux (ex: IA).
        """
        default_msg = "Non renseigné"

        # --- PRÉPARATION DES DONNÉES ---
        safe_list = []
        for data in merged_list:
            safe_data = data.copy()
            complex_keys = ["departments", "funding_events", "suborganizations", "technologies", "competitors", "keywords"]
            for key in complex_keys:
                if key in safe_data and safe_data[key] is not None and not isinstance(safe_data[key], str):
                    safe_data[key] = json.dumps(safe_data[key])
            safe_list.append(safe_data)

        query = """
        UNWIND $batch AS data
        
        // 1. CRÉATION DU DOMAINE DE SÉCURITÉ
        WITH data,
             CASE 
                 WHEN data.domain IS NULL OR data.domain = 'Non renseigné' OR trim(data.domain) = ''
                 THEN 'unknown_' + replace(toLower(COALESCE(data.name, 'unknown')), ' ', '_') + '_' + replace(toLower(COALESCE(data.country, data.location.country, 'unknown')), ' ', '_')
                 ELSE data.domain
             END AS safe_domain
             
        // 2. Identification de l'entreprise
        MERGE (c:Company {domain: safe_domain})
        SET c.name = COALESCE(data.name, c.name, $msg),
            c.apollo_id = COALESCE(data.apollo_id, c.apollo_id),
            c.uid = COALESCE(data.uid, c.uid, safe_domain)
        
        OPTIONAL MATCH (c)-[old_rel:CURRENT]->(old_v:Version)
        
        // 3. VÉRIFICATION DYNAMIQUE DES CHANGEMENTS (Nouvelle logique ajoutée !)
        WITH c, data, safe_domain, old_rel, old_v,
             (old_v IS NULL 
              // Vérifie si N'IMPORTE QUELLE clé de 'data' est différente de l'ancienne version
              OR any(k IN keys(data) WHERE COALESCE(toString(old_v[k]), 'NULL') <> COALESCE(toString(data[k]), 'NULL'))
             ) AS has_changed

        // 4. Archivage automatique (UNIQUEMENT SI CHANGEMENT)
        FOREACH (_ IN CASE WHEN old_rel IS NOT NULL AND has_changed THEN [1] ELSE [] END |
            DELETE old_rel
            MERGE (c)-[:ARCHIVED]->(old_v)
        )

        // 5. CRÉATION DE LA NOUVELLE VERSION DYNAMIQUE (UNIQUEMENT SI CHANGEMENT)
        FOREACH (_ IN CASE WHEN has_changed THEN [1] ELSE [] END |
            CREATE (new_v:Version)
            SET new_v += CASE WHEN old_v IS NOT NULL THEN properties(old_v) ELSE {} END
            SET new_v += data
            SET new_v.captured_at = datetime(),
                new_v.source = "Merged_Apollo_AI"
            CREATE (c)-[:CURRENT]->(new_v)
        )

        // 6. MISE À JOUR DU TIMESTAMP (SI AUCUN CHANGEMENT)
        FOREACH (_ IN CASE WHEN NOT has_changed THEN [1] ELSE [] END |
            SET old_v.last_verified_at = datetime()
        )
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, batch=safe_list, msg=default_msg)
                print(f"🚀 Graph Engine : {len(safe_list)} profils FUSIONNÉS (Apollo+IA) synchronisés avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l'import fusionné : {e}")

    def import_personas(self, personas_list, company_domain):
        """
        Importe les personas en sauvegardant TOUS les attributs fournis.
        Gère dynamiquement les formats simples et complexes (AeroLeads).
        """
        processed_personas = []
        
        for p in personas_list:
            # On crée une copie pour ne pas modifier l'original en cours de route
            p_clean = p.copy()
            
            # --- NORMALISATION DES CHAMPS CLÉS ---
            # On s'assure d'avoir un nom et un titre lisibles peu importe le format reçu
            p_clean['full_name'] = p.get('full_name') or p.get('clean_name_used') or "Inconnu"
            p_clean['title'] = p.get('title') or p.get('job_title_role') or "Non trouvé"
            
            # --- SÉRIALISATION DES DONNÉES COMPLEXES ---
            # Neo4j n'accepte pas les listes/dicts comme valeurs de propriétés directes.
            # On les transforme en chaînes de caractères JSON.
            for complex_field in ['education', 'experience', 'skills', 'emails', 'aeroleads_raw_data', 'interests']:
                if complex_field in p_clean and isinstance(p_clean[complex_field], (list, dict)):
                    p_clean[complex_field] = json.dumps(p_clean[complex_field], ensure_ascii=False)

            processed_personas.append(p_clean)

        query = """
        UNWIND $batch AS p_data
        MATCH (c:Company {domain: $domain})
        
        // On utilise l'URL LinkedIn (ou un substitut) comme identifiant unique pour le Persona
        MERGE (p:Persona {linkedin_url: COALESCE(p_data.linkedin_url, p_data.full_name + '_' + $domain)})
        
        // L'opérateur '+=' ajoute dynamiquement TOUTES les clés présentes dans p_data au nœud 'p'
        SET p += p_data,
            p.last_updated = datetime()
            
        // Création de la relation entre le Persona et l'Entreprise
        MERGE (p)-[:WORKS_AT]->(c)
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, batch=processed_personas, domain=company_domain)
                print(f"👥 {len(processed_personas)} personas importés avec succès et liés à {company_domain}.")
        except Exception as e:
            print(f"❌ Erreur lors de l'import dynamique des personas : {e}")

    def close(self):
        self.driver.close()