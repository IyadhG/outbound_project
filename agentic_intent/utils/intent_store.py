import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import sqlite3
from contextlib import contextmanager
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class IntentStore:
    """Manages company intent storage and retrieval"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(project_root, "intents.db")
        self.db_path = db_path
        self.init_db()
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy load the embedding model"""
        if self._embedding_model is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embedding_model = OpenAIEmbeddings(
                    model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small"),
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=os.getenv("OPENROUTER_BASE_URL"),
                )
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            # Events table - stores all events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    title_embedding TEXT,
                    confidence REAL,
                    date_value TEXT,
                    date_confidence REAL,
                    source TEXT,
                    source_url TEXT,
                    amount TEXT,
                    amount_confidence REAL,
                    investor TEXT,
                    investor_confidence REAL,
                    flag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Snapshots table - stores full structured data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company TEXT NOT NULL,
                    snapshot_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()

    def _get_embedding(self, text: str) -> Optional[str]:
        """Get embedding for text"""
        if self.embedding_model is None:
            return None
        try:
            embedding = self.embedding_model.embed_query(text)
            print("embedding successful")
            return json.dumps(embedding)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        a = np.array(emb1)
        b = np.array(emb2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def store_intent(self, structured_data: Dict):
        """Store intent data from the structured output"""
        with self.get_connection() as conn:
            # Clear old events for these companies
            for company_name in structured_data.get("companies", {}).keys():
                conn.execute("DELETE FROM events WHERE company = ?", (company_name,))
            
            # Store new events
            for company_name, data in structured_data.get("companies", {}).items():
                # Store funding events
                for event in data.get("funding_events", []):
                    title = event["event"]["title"]
                    embedding = self._get_embedding(title)
                    
                    conn.execute("""
                        INSERT INTO events 
                        (company, event_type, title, title_embedding, confidence, date_value, date_confidence,
                         source, source_url, amount, amount_confidence, investor, investor_confidence, flag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        company_name,
                        "funding",
                        title,
                        embedding,
                        event["event"]["confidence"],
                        event["date"]["value"],
                        event["date"]["confidence"],
                        event["source"]["name"],
                        event["source"]["url"],
                        event["financial_details"]["amount"],
                        event["financial_details"]["amount_confidence"],
                        event["financial_details"]["investor"],
                        event["financial_details"]["investor_confidence"],
                        "funding"
                    ))
                
                # Store news events
                for event in data.get("news_events", []):
                    title = event["event"]["title"]
                    embedding = self._get_embedding(title)
                    
                    conn.execute("""
                        INSERT INTO events 
                        (company, event_type, title, title_embedding, confidence, date_value, date_confidence,
                         source, source_url, flag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        company_name,
                        "news",
                        title,
                        embedding,
                        event["event"]["confidence"],
                        event["date"],
                        0.8,
                        event["source"]["name"],
                        event["source"]["url"],
                        "news"
                    ))
                
                # Store full snapshot
                conn.execute("""
                    INSERT INTO snapshots (company, snapshot_data)
                    VALUES (?, ?)
                """, (company_name, json.dumps(data)))
            
            conn.commit()
    
    def get_recent_events(self, company: str, limit: int = 5) -> List[Dict]:
        """Get the most recent events for a company"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM events 
                WHERE company = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (company, limit)).fetchall()
            
            return [dict(r) for r in rows]
    
    def get_older_events(self, company: str, exclude_recent: int = 5) -> List[Dict]:
        """Get events older than the most recent ones"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM events 
                WHERE company = ? 
                AND id NOT IN (
                    SELECT id FROM events 
                    WHERE company = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                )
                ORDER BY created_at DESC
            """, (company, company, exclude_recent)).fetchall()
            
            return [dict(r) for r in rows]

    def search_events_by_similarity(self, company: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for events similar to the query using embeddings.
        Falls back to text search if embeddings aren't available.
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        with self.get_connection() as conn:
            # Get all events for this company that have embeddings
            rows = conn.execute("""
                SELECT * FROM events 
                WHERE company = ? AND title_embedding IS NOT NULL
                ORDER BY created_at DESC
            """, (company,)).fetchall()
            
            if query_embedding and rows:
                # Use embedding similarity
                query_emb = json.loads(query_embedding)
                scored_events = []
                
                for row in rows:
                    event = dict(row)
                    event_emb = json.loads(event["title_embedding"])
                    similarity = self._cosine_similarity(query_emb, event_emb)
                    scored_events.append((similarity, event))
                
                # Sort by similarity (highest first)
                scored_events.sort(key=lambda x: x[0], reverse=True)
                
                # Return top_k events with similarity scores
                results = []
                for score, event in scored_events[:top_k]:
                    event_data = self._format_event_full(event)
                    event_data["similarity_score"] = float(score)
                    results.append(event_data)
                
                return results
            else:
                # Fallback: simple text search
                search_term = f"%{query}%"
                rows = conn.execute("""
                    SELECT * FROM events 
                    WHERE company = ? AND title LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (company, search_term, top_k)).fetchall()
                
                return [self._format_event_full(dict(r)) for r in rows]

    def _format_event_full(self, event: Dict) -> Dict:
        """Format an event with all details"""
        event_data = {
            "id": event["id"],
            "title": event["title"],
            "type": event["event_type"],
            "confidence": event["confidence"],
            "date": {
                "value": event["date_value"],
                "confidence": event["date_confidence"]
            },
            "source": {
                "name": event["source"],
                "url": event["source_url"]
            },
            "flag": event["flag"],
            "stored_at": event["created_at"]
        }
        
        if event["event_type"] == "funding":
            event_data["funding_details"] = {
                "amount": event["amount"],
                "amount_confidence": event["amount_confidence"],
                "investor": event["investor"],
                "investor_confidence": event["investor_confidence"]
            }
        
        return event_data

    def get_event_by_id(self, company: str, event_id: int) -> Optional[Dict]:
        """Get a specific event by ID (with company check for security)"""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM events 
                WHERE company = ? AND id = ?
            """, (company, event_id)).fetchone()
            
            if row:
                return self._format_event_full(dict(row))
        return None

    def get_latest_snapshot(self, company: str) -> Dict:
        """Get the most recent full snapshot"""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM snapshots 
                WHERE company = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (company,)).fetchone()
            
            if row:
                data = dict(row)
                data["snapshot_data"] = json.loads(data["snapshot_data"])
                return data
            return None
    
    def retrieve_essential(self, company: str) -> Dict:
        """
        Retrieval Type 1: Essential information
        - Last 5 events detailed
        - Summary of older events
        """
        recent_events = self.get_recent_events(company, 5)
        older_events = self.get_older_events(company, 5)
        
        # Format recent events with essential attributes
        formatted_recent = []
        for event in recent_events:
            event_data = {
                "title": event["title"],
                "confidence": event["confidence"],
                "date": {
                    "value": event["date_value"],
                    "confidence": event["date_confidence"]
                },
                "source": {
                    "name": event["source"],
                    "url": event["source_url"]
                },
                "type": event["event_type"],
                "flag": event["flag"]
            }
            
            # Add funding-specific attributes
            if event["event_type"] == "funding":
                event_data["funding_details"] = {
                    "amount": event["amount"],
                    "amount_confidence": event["amount_confidence"],
                    "investor": event["investor"],
                    "investor_confidence": event["investor_confidence"]
                }
            
            formatted_recent.append(event_data)
        
        # Prepare older events for summary generation
        older_summary_data = []
        for event in older_events:
            summary_event = {
                "title": event["title"],
                "type": event["event_type"],
                "confidence": event["confidence"],
                "date": event["date_value"],
                "source": event["source"]
            }
            
            if event["event_type"] == "funding":
                summary_event["amount"] = event["amount"]
                summary_event["investor"] = event["investor"]
            
            older_summary_data.append(summary_event)
        
        # Generate summary from older events
        summary = self.generate_summary(company, older_summary_data) if older_summary_data else "No historical events to summarize."
        
        return {
            "company": company,
            "retrieval_type": "essential",
            "retrieved_at": datetime.now().isoformat(),
            "recent_events": formatted_recent,
            "historical_summary": summary,
            "older_events_count": len(older_events)
        }
    
    def retrieve_full(self, company: str) -> Dict:
        """
        Retrieval Type 2: Full data
        - Last 5 events with all details
        - Complete metadata
        """
        recent_events = self.get_recent_events(company, 5)
        snapshot = self.get_latest_snapshot(company)
        
        # Format recent events with all attributes
        formatted_events = []
        for event in recent_events:
            event_data = {
                "id": event["id"],
                "title": event["title"],
                "type": event["event_type"],
                "confidence": event["confidence"],
                "date": {
                    "value": event["date_value"],
                    "confidence": event["date_confidence"]
                },
                "source": {
                    "name": event["source"],
                    "url": event["source_url"]
                },
                "flag": event["flag"],
                "stored_at": event["created_at"]
            }
            
            if event["event_type"] == "funding":
                event_data["funding_details"] = {
                    "amount": event["amount"],
                    "amount_confidence": event["amount_confidence"],
                    "investor": event["investor"],
                    "investor_confidence": event["investor_confidence"]
                }
            
            formatted_events.append(event_data)
        
        # Get metadata from snapshot if available
        metadata = {}
        if snapshot and "snapshot_data" in snapshot:
            snap_data = snapshot["snapshot_data"]
            if "summary" in snap_data:
                metadata["summary"] = snap_data["summary"]
            if "metadata" in snap_data:
                metadata["generated_info"] = snap_data["metadata"]
        
        return {
            "company": company,
            "retrieval_type": "full",
            "retrieved_at": datetime.now().isoformat(),
            "metadata": metadata,
            "recent_events": formatted_events,
            "total_events_available": len(formatted_events)
        }

    def generate_summary(self, company: str, events: List[Dict]) -> str:
        """
        Generate a brief summary of older events using LLM.
        Falls back to a basic summary if LLM is unavailable.
        """
        if not events:
            return f"No historical events found for {company}."
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage
            
            llm = ChatGroq(
                model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Prepare events for summarization
            events_text = []
            for i, event in enumerate(events, 1):
                event_desc = f"{i}. [{event['type'].upper()}] {event['title']}"
                if event.get('amount'):
                    event_desc += f" - Amount: {event['amount']}"
                    if event.get('investor'):
                        event_desc += f" from {event['investor']}"
                event_desc += f" (Date: {event.get('date', 'unknown')}, Confidence: {event.get('confidence', 0):.0%})"
                events_text.append(event_desc)
            
            prompt = f"""
    Summarize the key trends and insights for {company} based on these historical events.

    EVENTS:
    {chr(10).join(events_text)}

    TASK:
    Provide a concise summary (2-4 sentences) highlighting:
    - Main funding trends or notable investments
    - Key business developments or news patterns
    - Overall sentiment or direction of the company

    Keep it brief and focused on actionable insights.
    """
            
            response = llm.invoke([SystemMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            # Fallback: generate basic summary without LLM
            funding_events = [e for e in events if e['type'] == 'funding']
            news_events = [e for e in events if e['type'] == 'news']
            
            parts = []
            if funding_events:
                amounts = [e.get('amount') for e in funding_events if e.get('amount')]
                investors = [e.get('investor') for e in funding_events if e.get('investor')]
                parts.append(f"{len(funding_events)} funding events")
                if amounts:
                    parts.append(f"amounts ranging from {min(amounts)} to {max(amounts)}")
                if investors:
                    parts.append(f"involving {', '.join(investors[:3])}")
            
            if news_events:
                parts.append(f"{len(news_events)} news events covering various developments")
            
            return f"Historical data for {company}: {'. '.join(parts)}."