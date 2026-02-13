import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from scripts.logger import setup_logger

# Δημιουργία logger για αυτό το module
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

            # 2. Δημιουργία Δυναμικού FAISS Index (On-the-fly)
            d = doc_embeddings.shape[1] # Διάσταση διανύσματος (π.χ. 384)
            
            # IndexFlatIP = Inner Product. Επειδή τα διανύσματα είναι normalized, είναι Cosine Similarity.
            index = faiss.IndexFlatIP(d) 
            index.add(doc_embeddings)

            # 3. Αναζήτηση (Search)
            # Ζητάμε scores για όλα τα έγγραφα (k = len(docs))
            k = len(docs)
            scores, indices = index.search(query_embedding, k)

            # 4. Ευθυγράμμιση (Un-sorting)
            # Το FAISS επιστρέφει τα αποτελέσματα ταξινομημένα (best match first).
            # Εμείς θέλουμε τα scores να αντιστοιχούν 1-προς-1 στην αρχική λίστα 'docs'
            # για να μπορέσουμε να τα προσθέσουμε με τα BM25 scores.
            
            # Φτιάχνουμε έναν άδειο πίνακα στο μέγεθος των εγγράφων
            aligned_scores = np.zeros(k)
            
            # indices[0] περιέχει τα ID των εγγράφων με τη σειρά που τα έβγαλε το FAISS
            # scores[0] περιέχει τα αντίστοιχα σκορ
            for rank, original_idx in enumerate(indices[0]):
                aligned_scores[original_idx] = scores[0][rank]

            return aligned_scores

        except Exception as e:
            logger.error(f"Error during vector scoring: {e}")
            # Σε περίπτωση λάθους επιστρέφουμε μηδενικά για να μην σκάσει το pipeline
            return np.zeros(len(docs))