import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(log_name: str = 'HybridSearchLogger', log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Ρυθμίζει και επιστρέφει έναν Logger έτοιμο για χρήση.
    
    Χαρακτηριστικά:
    - Singleton Pattern: Αποτρέπει τα διπλά logs αν κληθεί πολλές φορές.
    - Console Output: Εμφανίζει μόνο INFO και πάνω (καθαρή εικόνα στο τερματικό).
    - File Output: Αποθηκεύει DEBUG και πάνω (πλήρες ιστορικό στο αρχείο).
    - Auto-Directory: Δημιουργεί αυτόματα τον φάκελο logs αν λείπει.
    - UTF-8: Υποστηρίζει ελληνικούς χαρακτήρες.

    Args:
        log_name (str): Το όνομα του logger.
        log_dir (Path, optional): Ο φάκελος αποθήκευσης. Αν είναι None, 
                                  βρίσκει αυτόματα το 'data/logs' στο root του project.
    """
    
    if log_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "data" / "logs"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create log directory {log_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    log_filename = f"{datetime.now().strftime('%Y_%m_%d')}.log"
    log_filepath = log_dir / log_filename

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    try:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

        fh = logging.FileHandler(log_filepath, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        logger.debug(f"Logger initialized successfully. Log file: {log_filepath}")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to set up logger handlers: {e}", file=sys.stderr)
        sys.exit(1)

    return logger