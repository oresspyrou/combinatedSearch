import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
from config import AppConfig

cfg = AppConfig()

def index_data():
    print(f"Connecting to Elasticsearch at {cfg.ES_HOST}...")
    es = Elasticsearch(cfg.ES_HOST)
    
    if not es.ping():
        print("Elasticsearch is not running!")
        return

    print(f"Reading documents from {cfg.DOCUMENTS_PATH}...")
    df = pd.read_csv(cfg.DOCUMENTS_PATH) 
    
    df.columns = df.columns.str.strip() 

    print(f"Indexing {len(df)} documents into '{cfg.ES_INDEX}'...")

    if not es.indices.exists(index=cfg.ES_INDEX):
        es.indices.create(index=cfg.ES_INDEX)

    actions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = {
            "_index": cfg.ES_INDEX,
            "_source": {
                "ID": str(row['ID']),     
                "Text": str(row['Text']) 
            }
        }
        actions.append(doc)

    success, _ = bulk(es, actions)
    print(f"Successfully indexed {success} documents!")

if __name__ == "__main__":
    index_data()