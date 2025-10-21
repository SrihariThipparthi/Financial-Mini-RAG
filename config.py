import os
from dataclasses import dataclass

@dataclass
class Config:
    FAQS_PATH = "data/faqs.csv"
    PERFORMANCE_PATH = "data/funds.csv"
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Retrieval settings
    TOP_K = 5
    SEMANTIC_WEIGHT = 0.7
    LEXICAL_WEIGHT = 0.3
    
    # API settings
    HOST = "127.0.0.1"
    PORT = 8000

config = Config()