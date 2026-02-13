from elasticsearch import Elasticsearch
from typing import List, Dict, Any
from scripts.logger import setup_logger

logger = setup_logger(log_name="ElasticRetriever")

class ElasticRetriever:
    """
    Διαχειρίζεται την επικοινωνία με την Elasticsearch.
    Εκτελεί το 1ο στάδιο της ανάκτησης (BM25 Retrieval).
    """

    def __init__(self, host: str, index_name: str):
        """
        Αρχικοποιεί τη σύνδεση με την Elasticsearch.

        Args:
            host (str): Η διεύθυνση του server (π.χ. http://localhost:9200).
            index_name (str): Το όνομα του ευρετηρίου (index).
        """
        self.client = Elasticsearch(host)
        self.index = index_name
        
        if self.client.ping():
            logger.info(f"Connected to Elasticsearch at {host}")
        else:
            logger.error(f"Could not connect to Elasticsearch at {host}!")


    def retrieve_candidates(self, query: str, size: int = 200) -> List[Dict[str, Any]]:
        """
        Εκτελεί αναζήτηση BM25 και επιστρέφει τους υποψήφιους.

        Args:
            query (str): Το κείμενο της ερώτησης.
            size (int): Πόσα αποτελέσματα να φέρει (N >> k).

        Returns:
            List[Dict]: Λίστα με {id, text, bm25_score}.
        """
        try:
            response = self.client.search(
                index=self.index,
                body={
                    "query": {
                        "match": {
                            "Text": query  
                        }
                    },
                    "size": size
                }
            )

            hits = response['hits']['hits']
            logger.debug(f"Retrieved {len(hits)} documents for query: '{query[:30]}...'")

            candidates = []
            for hit in hits:
                source = hit['_source']
                candidates.append({
                    'id': source.get('ID', 'UNKNOWN'),   
                    'text': source.get('Text', ''),      
                    'bm25_score': hit['_score']         
                })

            return candidates

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []