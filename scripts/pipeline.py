import pandas as pd
from tqdm import tqdm
from scripts.logger import setup_logger
from scripts.retriever_ES import ElasticRetriever
from scripts.ranker_FAISS import VectorRanker
from scripts.fusion import ScoreFusion
from config import AppConfig, config as default_cfg

logger = setup_logger(log_name="HybridPipeline")

class HybridPipeline:
    """
    Ο ενορχηστρωτής (Orchestrator) της διαδικασίας.
    Ενώνει Retriever, Ranker και Fusion σε μια ενιαία ροή εκτέλεσης.
    """

    def __init__(self, config: AppConfig = None):
        # Διόρθωση: Αν δεν μας δώσουν config, παίρνουμε το default
        self.cfg = config if config else default_cfg
        
        logger.info("Initializing Hybrid Search Pipeline...")
        
        # Σύνδεση με Elasticsearch και FAISS χρησιμοποιώντας τις ρυθμίσεις του config
        self.retriever = ElasticRetriever(host=self.cfg.ES_HOST, index_name=self.cfg.ES_INDEX)
        self.ranker = VectorRanker(model_name=self.cfg.MODEL_NAME)
        
    def load_queries(self) -> pd.DataFrame:
        """Φορτώνει τα ερωτήματα από το CSV."""
        logger.info(f"Loading queries from: {self.cfg.QUERIES_PATH}")
        try:
            df = pd.read_csv(self.cfg.QUERIES_PATH)
            
            df.columns = df.columns.str.strip()
            
            # Έλεγχος αν οι στήλες QueryID/Query υπάρχουν, αλλιώς παίρνουμε τις 2 πρώτες
            if 'QueryID' not in df.columns or 'Query' not in df.columns:
                logger.warning("Columns 'QueryID'/'Query' not found. Using 1st and 2nd columns.")
                df.columns = ['QueryID', 'Query'] + list(df.columns[2:])
            
            logger.info(f"Loaded {len(df)} queries.")
            return df
            
        except Exception as e:
            logger.critical(f"Failed to load queries: {e}")
            raise e

    def run(self):
        """Εκτελεί το Main Loop της αναζήτησης."""
        queries_df = self.load_queries()
        results_buffer = []

        logger.info(f"Starting Execution (Alpha={self.cfg.ALPHA}, N={self.cfg.N_RETRIEVE})...")

        # Επεξεργασία κάθε ερωτήματος
        for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Processing"):
            # Διόρθωση ID: Μετατροπή σε float και μετά σε int για να φύγει το .0 (π.χ. 1.0 -> 1)
            try:
                qid = str(int(float(row['QueryID'])))
            except:
                qid = str(row['QueryID'])
                
            query_text = str(row['Query'])

            # 1. Ανάκτηση Υποψηφίων από Elasticsearch (BM25)
            candidates = self.retriever.retrieve_candidates(query_text, size=self.cfg.N_RETRIEVE)
            
            if not candidates:
                logger.warning(f"No candidates found for QID: {qid}")
                continue

            # 2. Υπολογισμός Semantic Scores (FAISS) για τους υποψηφίους
            doc_texts = [c['text'] for c in candidates]
            vector_scores = self.ranker.get_scores(query_text, doc_texts)

            # 3. Συνδυασμός Σκορ (Fusion)
            ranked_results = ScoreFusion.fuse(candidates, vector_scores, alpha=self.cfg.ALPHA)

            # 4. Αποθήκευση στο Buffer (Χρησιμοποιούμε TOP_K αντί για σταθερό 50)
            # Το trec_eval χρειάζεται μέχρι 1000 αποτελέσματα για σωστό MAP
            for rank, item in enumerate(ranked_results[:self.cfg.N_RETRIEVE]):
                # Format: QID Q0 DOCID RANK SCORE RUN_ID
                line = f"{qid} Q0 {item['id']} {rank+1} {item['final_score']:.6f} HYBRID_SYS"
                results_buffer.append(line)

        self.save_results(results_buffer)

    def save_results(self, lines: list):
        """Αποθηκεύει τα αποτελέσματα στο αρχείο εξόδου."""
        output_path = self.cfg.HYBRID_RESULTS_PATH
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            logger.info(f"Results saved successfully to: {output_path}")
            logger.info("Pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    # Για testing αν τρέξεις απευθείας το αρχείο
    pipeline = HybridPipeline(default_cfg)
    pipeline.run()