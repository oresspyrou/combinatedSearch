import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from scripts.logger import setup_logger

logger = setup_logger(log_name="VectorRanker")

class VectorRanker:
    """
    Υλοποιεί το Semantic Search (Reranking).
    Χρησιμοποιεί SBERT για embeddings και FAISS για υπολογισμό ομοιότητας.
    """

    def __init__(self, model_name: str):
        """
        Φορτώνει το μοντέλο Transformer στη μνήμη.
        """
        logger.info(f"⏳ Loading SBERT model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load model {model_name}: {e}")
            raise e

    def get_scores(self, query: str, docs: List[str]) -> np.ndarray:
        """
        Υπολογίζει το Cosine Similarity μεταξύ του Query και μιας λίστας κειμένων.
        
        Διαδικασία (σύμφωνα με την εκφώνηση):
        1. Embeddings (Query + Docs)
        2. Dynamic FAISS Index
        3. Search
        """
        if not docs:
            return np.array([])

        try:
            # 1. Δημιουργία Embeddings
            # normalize_embeddings=True σημαίνει ότι το Dot Product ισοδυναμεί με Cosine Similarity
            doc_embeddings = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

            d = doc_embeddings.shape[1] 
            
            index = faiss.IndexFlatIP(d) 
            index.add(doc_embeddings)

            k = len(docs)
            scores, indices = index.search(query_embedding, k)

            aligned_scores = np.zeros(k)
            
            for rank, original_idx in enumerate(indices[0]):
                aligned_scores[original_idx] = scores[0][rank]

            return aligned_scores

        except Exception as e:
            logger.error(f"Error during vector scoring: {e}")
            return np.zeros(len(docs))