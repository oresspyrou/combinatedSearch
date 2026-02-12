from config import AppConfig
from scripts.pipeline import HybridPipeline

if __name__ == "__main__":
    cfg = AppConfig()
    pipe = HybridPipeline(cfg)
    pipe.run()