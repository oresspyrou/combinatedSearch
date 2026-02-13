import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    QUERIES_PATH: Path = DATA_DIR / "IR2025" / "queries.csv"
    DOCUMENTS_PATH: Path = DATA_DIR / "IR2025" / "documents.csv"
    
    RESULTS_DIR: Path = DATA_DIR / "results"
    HYBRID_RESULTS_PATH: Path = RESULTS_DIR / "hybrid_results.txt"
    LOGS_DIR: Path = DATA_DIR / "logs"

    ES_HOST: str = "http://localhost:9200"
    ES_INDEX: str = "ir2025_documents"
    
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # --- ΟΙ ΣΗΜΑΝΤΙΚΕΣ ΑΛΛΑΓΕΣ ---
    # Ζητάμε 1000 από τη βάση για να έχουμε μεγάλο εύρος
    N_RETRIEVE: int = 1000
    
    # Γράφουμε 1000 στο αρχείο αποτελεσμάτων (όχι 50!)
    TOP_K: int = 1000
    
    # Κρατάμε το Alpha χαμηλά γιατί το BM25 είναι δυνατό
    ALPHA: float = 0.80      

    def __post_init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

config = AppConfig()