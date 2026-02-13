import pandas as pd
from tqdm import tqdm
from scripts.logger import setup_logger
from scripts.retriever_ES import ElasticRetriever
from scripts.ranker_FAISS import VectorRanker
from scripts.fusion import ScoreFusion
from config import AppConfig

# Logger setup
logger = setup_logger(log_name="HybridPipeline")

class HybridPipeline:
    """
    Ο ενορχηστρωτής (Orchestrator) της διαδικασίας.
    Ενώνει Retriever, Ranker και Fusion σε μια ενιαία ροή εκτέλεσης.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        
        logger.info("🚀 Initializing Hybrid Search Pipeline...")
        
        # 1. Αρχικοποίηση Components
        self.retriever = ElasticRetriever(host=self.cfg.ES_HOST, index_name=self.cfg.ES_INDEX)
        self.ranker = VectorRanker(model_name=self.cfg.MODEL_NAME)
        
    def load_queries(self) -> pd.DataFrame:
        """Φορτώνει τα ερωτήματα από το CSV."""
        logger.info(f"📂 Loading queries from: {self.cfg.QUERIES_PATH}")
        try:
            # Διαβάζουμε το CSV
            df = pd.read_csv(self.cfg.QUERIES_PATH)
            
            # Καθαρισμός ονομάτων στηλών (για να είμαστε σίγουροι)
            df.columns = df.columns.str.strip()
            
            # Έλεγχος αν υπάρχουν οι σωστές στήλες
            if 'QueryID' not in df.columns or 'Query' not in df.columns:
                # Αν δεν υπάρχουν, υποθέτουμε ότι είναι οι 2 πρώτες
                logger.warning("Columns 'QueryID'/'Query' not found. Using 1st and 2nd columns.")
                df.columns = ['QueryID', 'Query'] + list(df.columns[2:])
            
            logger.info(f"✅ Loaded {len(df)} queries.")
            return df
            
        except Exception as e:
            logger.critical(f"❌ Failed to load queries: {e}")
            raise e

    def run(self):
        """Εκτελεί το Main Loop της αναζήτησης."""
        queries_df = self.load_queries()
        results_buffer = []

        logger.info(f"🔥 Starting Execution (Alpha={self.cfg.ALPHA}, N={self.cfg.N_RETRIEVE})...")

        # Χρήση tqdm για μπάρα προόδου
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Processing"):
            qid = str(row['QueryID'])
            query_text = str(row['Query'])

            # --- ΒΗΜΑ 1: Retrieve (Elasticsearch) ---
            candidates = self.retriever.retrieve_candidates(query_text, size=self.cfg.N_RETRIEVE)
            
            if not candidates:
                logger.warning(f"⚠️ No candidates found for QID: {qid}")
                continue

            # --- ΒΗΜΑ 2: Rerank (Vector/FAISS) ---
            # Παίρνουμε μόνο τα κείμενα για να φτιάξουμε embeddings
            doc_texts = [c['text'] for c in candidates]
            vector_scores = self.ranker.get_scores(query_text, doc_texts)

            # --- ΒΗΜΑ 3: Fusion (Combine Scores) ---
            ranked_results = ScoreFusion.fuse(candidates, vector_scores, alpha=self.cfg.ALPHA)

            # --- ΒΗΜΑ 4: Formatting (TREC Style) ---
            # Κρατάμε τα top-50 (ή όσο λέει το config)
            # Format: qid Q0 doc_id rank score run_id
            for rank, item in enumerate(ranked_results[:50]):
                line = f"{qid} Q0 {item['id']} {rank+1} {item['final_score']:.6f} HYBRID_SYS"
                results_buffer.append(line)

        # --- ΒΗΜΑ 5: Save Results ---
        self.save_results(results_buffer)

    def save_results(self, lines: list):
        """Αποθηκεύει τα αποτελέσματα στο αρχείο εξόδου."""
        output_path = self.cfg.HYBRID_RESULTS_PATH
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"💾 Results saved successfully to: {output_path}")
            logger.info("🎉 Pipeline completed successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    # Για testing αν τρέξεις απευθείας το αρχείο
    # (Κανονικά θα το τρέχουμε από το main.py)
    config = AppConfig()
    pipeline = HybridPipeline(config)
    pipeline.run()