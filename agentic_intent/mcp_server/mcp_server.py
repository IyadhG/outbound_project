from fastmcp import FastMCP
from ddgs import DDGS
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json
import sys

# --- CONFIGURATION ET CHARGEMENT ---
load_dotenv()

# Ajout du chemin pour les imports locaux (utile pour IntentStore)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CLASSE NEO4J MANAGER (UNIFIÉE) ---
class Neo4jManager:
    def __init__(self):
        # Configuration des accès AuraDB
        self.uri = "neo4j+s://151c0242.databases.neo4j.io"
        self.user = "151c0242" 
        self.password = "1b56m_IXQlXOlENuIQmkjOswv-CddFai8TACtl5JXXo" 
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("[OK] Neo4j connectivity established.")
        except Exception as e:
            print(f"[ERROR] Connection error: {e}")
            sys.exit(1)

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Exécute une requête Cypher et retourne les résultats sous forme de liste de dictionnaires."""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

# --- INITIALISATION ---
mcp = FastMCP("intent-tools")
db = Neo4jManager()


# ==========================================
# 1. OUTILS DE RECHERCHE WEB (DDGS)
# ==========================================

@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_funding(company_name: str) -> list:
    """
    Search for recent news about a company related to funding.
    """
    query = f"{company_name} funding"
    results_list = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            for r in results:
                results_list.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body")
                })
    except Exception as e:
        return [{"error": str(e)}]
    return results_list


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_partnerships(company_name: str) -> list:
    """
    Search for recent news about a company related to partnerships.
    """
    query = f"{company_name} partnership"
    results_list = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            for r in results:
                results_list.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body")
                })
    except Exception as e:
        return [{"error": str(e)}]
    return results_list


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": False,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def search_company_news(company_name: str) -> list:
    """
    Search for recent general news about a company.
    """
    query = f"{company_name} news"
    results_list = []
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=5)
            for r in results:
                results_list.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("body"),
                    "date": r.get("date"),
                    "source": r.get("source")
                })
    except Exception as e:
        return [{"error": str(e)}]
    return results_list


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "idempotentHint": True,
        "destructiveHint": False,
        "openWorldHint": True
    }
)
def retrieve_company_intent(company_name: str, format_type: str = "llm_context") -> str:
    """
    Retrieve stored intelligence about a company. 
    Use this to get context before answering questions about the company.
    """
    from utils.intent_store import IntentStore
    intent_store = IntentStore()

    if format_type == "llm_context":
        return intent_store.get_llm_context(company_name)
    elif format_type == "full_data":
        data = intent_store.retrieve_intent(company_name, "all")
        return json.dumps(data, indent=2, default=str)
    elif format_type == "summary":
        data = intent_store.retrieve_intent(company_name, "long_term")
        memories = data.get("long_term", [])
        if memories:
            return "\n".join([m["content"] for m in memories])
        return f"No stored memories for {company_name}"
    else:
        return intent_store.get_llm_context(company_name)


# ==========================================
# 2. OUTILS NEO4J (LES 13 REQUÊTES)
# ==========================================

@mcp.tool()
def get_full_graph() -> str:
    """Retrieve the entire graph: Returns companies, versions, and personas (Limited to 100 to avoid overload)."""
    query = """
    MATCH (c:Company)
    OPTIONAL MATCH (c)-[r:CURRENT|ARCHIVED]->(v:Version)
    OPTIONAL MATCH (p:Persona)-[w:WORKS_AT]->(c)
    RETURN c, r, v, p, w LIMIT 100
    """
    return json.dumps(db.run_query(query), indent=2, default=str)


@mcp.tool()
def get_company_by_name(company_name: str) -> str:
    """Retrieve company by name: Returns all information of the CURRENT relation for a company using its name."""
    query = """
    MATCH (c:Company {name: $name})-[r:CURRENT]->(v:Version)
    RETURN c.name AS CompanyName, r, v
    """
    return json.dumps(db.run_query(query, {"name": company_name}), indent=2, default=str)


@mcp.tool()
def get_company_by_name_and_region(company_name: str, country: str) -> str:
    """Retrieve company by name and region: Returns the CURRENT version of a company using its name and country."""
    query = """
    MATCH (c:Company {name: $name})-[:CURRENT]->(v:Version {country: $country})
    RETURN c.name AS companyName, v
    """
    return json.dumps(db.run_query(query, {"name": company_name, "country": country}), indent=2, default=str)


@mcp.tool()
def get_company_by_domain(domain: str) -> str:
    """Retrieve company by domain: Returns all information of the CURRENT relation using the domain."""
    query = """
    MATCH (c:Company {domain: $domain})-[r:CURRENT]->(v:Version)
    RETURN c.name AS CompanyName, r AS Relation, v AS VersionInfo
    """
    return json.dumps(db.run_query(query, {"domain": domain}), indent=2, default=str)


@mcp.tool()
def get_companies_by_industry_and_country(industry: str, country: str) -> str:
    """Retrieve companies by industry and country: Returns companies from the CURRENT relation based on industry and country."""
    query = """
    MATCH (c:Company)-[r:CURRENT]->(v:Version)
    WHERE v.industry = $industry AND v.country = $country
    RETURN c.name AS company_name, r, v
    """
    return json.dumps(db.run_query(query, {"industry": industry, "country": country}), indent=2, default=str)


@mcp.tool()
def get_personas_by_company_and_country(company_name: str, country: str) -> str:
    """Retrieve personas by company name and country: Returns all personas of a company in a given country."""
    query = """
    MATCH (c:Company {name: $name})-[r:CURRENT]->(v:Version {country: $country})
    MATCH (p:Persona)-[:WORKS_AT]->(c)
    RETURN p
    """
    return json.dumps(db.run_query(query, {"name": company_name, "country": country}), indent=2, default=str)


@mcp.tool()
def get_personas_by_company_name(company_name: str) -> str:
    """Retrieve personas by company name: Returns all personas linked to a company by its name."""
    query = """
    MATCH (c:Company {name: $name})<-[:WORKS_AT]-(p:Persona)
    RETURN p
    """
    return json.dumps(db.run_query(query, {"name": company_name}), indent=2, default=str)


@mcp.tool()
def get_personas_by_domain(domain: str) -> str:
    """Retrieve personas by company domain: Returns all personas of a company using its domain."""
    query = """
    MATCH (c:Company {domain: $domain})<-[:WORKS_AT]-(p:Persona)
    RETURN p
    """
    return json.dumps(db.run_query(query, {"domain": domain}), indent=2, default=str)


@mcp.tool()
def list_unique_industries() -> str:
    """Display unique industries: Displays all business sectors (industries) registered in the graph."""
    query = """
    MATCH (v:Version)
    WHERE v.industry IS NOT NULL
    RETURN DISTINCT v.industry AS industries
    """
    return json.dumps(db.run_query(query), indent=2, default=str)


@mcp.tool()
def list_unique_countries() -> str:
    """Display unique countries: Displays all countries registered in the graph."""
    query = """
    MATCH (v:Version)
    WHERE v.country IS NOT NULL
    RETURN DISTINCT v.country AS countries
    """
    return json.dumps(db.run_query(query), indent=2, default=str)


@mcp.tool()
def add_intent_by_name(company_name: str, intent_text: str) -> str:
    """Insert intent (company name): Adds an intent attribute to the CURRENT version of a company using its name."""
    query = """
    MATCH (c:Company {name: $name})
    MATCH (c)-[r:CURRENT]->(v:Version)
    SET v.intent = $intent
    RETURN v
    """
    results = db.run_query(query, {"name": company_name, "intent": intent_text})
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def add_intent_by_domain(domain: str, intent_text: str) -> str:
    """Insert intent (company domain): Adds an intent attribute to the CURRENT version of a company using its domain."""
    query = """
    MATCH (c:Company {domain: $domain})
    MATCH (c)-[r:CURRENT]->(v:Version)
    SET v.intent = $intent
    RETURN v
    """
    results = db.run_query(query, {"domain": domain, "intent": intent_text})
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def add_intent_by_name_and_country(company_name: str, country: str, intent_text: str) -> str:
    """Insert intent (company name, country): Adds an intent attribute to the CURRENT version of a company using its name and country."""
    query = """
    MATCH (c:Company {name: $name})-[:CURRENT]->(v:Version {country: $country})
    SET v.intent = $intent
    RETURN v
    """
    results = db.run_query(query, {"name": company_name, "country": country, "intent": intent_text})
    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        db.close()