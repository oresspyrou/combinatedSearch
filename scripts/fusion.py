import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
from scripts.logger import setup_logger

logger = setup_logger(log_name="ScoreFusion")

class ScoreFusion:
    """
    Υπεύθυνος για την κανονικοποίηση και τον συνδυασμό των βαθμολογιών (Hybrid Fusion).
    """

    @staticmethod
    def normalize(scores: np.ndarray) -> np.ndarray:
        """
        Μετατρέπει τα scores σε κλίμακα [0, 1] χρησιμοποιώντας Min-Max Scaling.
        """
        if scores.size == 0:
            return scores
        
        if np.max(scores) == np.min(scores):
            return np.ones_like(scores)

        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    @staticmethod
    def fuse(candidates: List[Dict[str, Any]], vector_scores: np.ndarray, alpha: float) -> List[Dict[str, Any]]:
        """
        Συνδυάζει τα BM25 scores και τα Vector scores.
        
        Formula: FinalScore = (Alpha * NormBM25) + ((1-Alpha) * NormVector)
        
        Args:
            candidates: Λίστα με τα έγγραφα και τα BM25 scores τους.
            vector_scores: Numpy array με τα Vector scores (στην ίδια σειρά με τα candidates).
            alpha: Το βάρος του BM25 (0.0 - 1.0).
        
        Returns:
            List: Η λίστα ταξινομημένη με βάση το νέο Hybrid Score.
        """
        if not candidates or len(vector_scores) == 0:
            logger.warning("Empty candidates or vector scores provided for fusion.")
            return []

        try:
            bm25_scores = np.array([c['bm25_score'] for c in candidates])

            norm_bm25 = ScoreFusion.normalize(bm25_scores)
            norm_vec = ScoreFusion.normalize(vector_scores)

            logger.debug(f"Fusion Stats -> Alpha: {alpha}")
            logger.debug(f"BM25 range: [{np.min(bm25_scores):.2f}, {np.max(bm25_scores):.2f}]")
            logger.debug(f"Vec range:  [{np.min(vector_scores):.2f}, {np.max(vector_scores):.2f}]")

            fused_results = []
            for i, cand in enumerate(candidates):
                hybrid_score = (alpha * norm_bm25[i]) + ((1 - alpha) * norm_vec[i])
                
                cand['final_score'] = hybrid_score
                cand['norm_bm25'] = norm_bm25[i]  
                cand['norm_vec'] = norm_vec[i]
                
                fused_results.append(cand)

            fused_results.sort(key=lambda x: x['final_score'], reverse=True)

            return fused_results

        except Exception as e:
            logger.error(f"Error during score fusion: {e}")
            return candidates 
        