"""
Gemini Embedder for semantic similarity ranking
"""

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class GeminiEmbedder:
    """Gemini-based text embedder for semantic similarity."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-embedding-001"):
        """Initialize Gemini embedder."""
        import google.generativeai as genai
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided")
        
        genai.configure(api_key=self.api_key)
        self.model = model
        self._embedding_model = genai.GenerativeModel(model)
        logger.info(f"Gemini Embedder initialized with model: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        import google.generativeai as genai
        
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            embedding = result['embedding']
            logger.debug(f"Embedded text (length: {len(text)} chars, embedding dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        from scipy.spatial.distance import cosine
        
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
