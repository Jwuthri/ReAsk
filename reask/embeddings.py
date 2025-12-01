"""Embedding-based similarity detection"""

import numpy as np
from typing import Optional
from openai import OpenAI


class EmbeddingService:
    """Handles text embedding and similarity"""
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "text-embedding-3-small"
    ):
        self.client = client or OpenAI()
        self.model = model
        self._cache: dict[str, list[float]] = {}
    
    def embed(self, text: str) -> list[float]:
        """Get embedding for text, with caching"""
        if text in self._cache:
            return self._cache[text]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding = response.data[0].embedding
        self._cache[text] = embedding
        return embedding
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))
        
        # Cosine similarity
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()

