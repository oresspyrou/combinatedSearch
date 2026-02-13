import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    """
    Κεντρική κλάση ρυθμίσεων της εφαρμογής.
    Συγκεντρώνει όλες τις σταθερές και τις διαδρομές αρχείων.
    """

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
    
    N_RETRIEVE: int = 200    
    ALPHA: float = 0.5       

    def __post_init__(self):
        """Διασφαλίζει ότι οι φάκελοι εξόδου υπάρχουν."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

config = AppConfig()