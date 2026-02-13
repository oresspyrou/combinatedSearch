import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class VectorRanker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_scores(self, query_text: str, doc_texts: List[str]) -> np.ndarray:
        if not doc_texts:
            return np.array([])

        query_embedding = self.model.encode([query_text]).astype('float32')
        doc_embeddings = self.model.encode(doc_texts).astype('float32')

        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(doc_embeddings)

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) 
        index.add(doc_embeddings)

        distances, indices = index.search(query_embedding, len(doc_texts))
        
        final_scores = np.zeros(len(doc_texts))
        for score, idx in zip(distances[0], indices[0]):
            final_scores[idx] = score
            
        return final_scores