import sys
from config import AppConfig
from scripts.pipeline import HybridPipeline
from scripts.logger import setup_logger

logger = setup_logger(log_name="MainApp")

def main():
    """
    Η κεντρική συνάρτηση που εκκινεί την εφαρμογή.
    """
    try:
        logger.info("Application is starting...")

        config = AppConfig()

        logger.info(f"Configuration loaded. Mode: Hybrid (Alpha={config.ALPHA})")

        pipeline = HybridPipeline(config)

        pipeline.run()

        logger.info("Application finished successfully.")

    except KeyboardInterrupt:
        logger.warning("\n Execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical Error in Main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()