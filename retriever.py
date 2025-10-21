import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
from typing import List, Dict, Any, Tuple
import re

class MiniRAGRetriever:
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.faiss_index = None
        self.documents = []
        self.document_embeddings = None
        
    def preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def build_indices(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        texts = [doc['content'] for doc in documents]
        
        self.document_embeddings = self.embedding_model.encode(texts)
        self.faiss_index = faiss.IndexFlatIP(self.config.EMBEDDING_DIM)
        self.faiss_index.add(self.document_embeddings.astype('float32'))
        
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        if top_k is None:
            top_k = self.config.TOP_K
            
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((idx, float(score)))
                
        return results
    
    def lexical_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        if top_k is None:
            top_k = self.config.TOP_K
            
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((idx, float(similarities[idx])))
                
        return results
    
    def hybrid_search(self, query: str, top_k: int = None) -> List[Tuple[int, float]]:
        if top_k is None:
            top_k = self.config.TOP_K
            
        semantic_results = self.semantic_search(query, top_k * 2)
        lexical_results = self.lexical_search(query, top_k * 2)
        
        scores = {}
        
        for idx, score in semantic_results:
            scores[idx] = scores.get(idx, 0) + score * self.config.SEMANTIC_WEIGHT
        
        for idx, score in lexical_results:
            scores[idx] = scores.get(idx, 0) + score * self.config.LEXICAL_WEIGHT
        
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score) for idx, score in sorted_results]
    
    def retrieve(self, query: str, mode: str = "semantic", top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.config.TOP_K
            
        if mode == "semantic":
            results = self.semantic_search(query, top_k)
        elif mode == "lexical":
            results = self.lexical_search(query, top_k)
        elif mode == "hybrid":
            results = self.hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        retrieved_docs = []
        for idx, score in results:
            doc = self.documents[idx].copy()
            doc['score'] = score
            retrieved_docs.append(doc)
            
        return retrieved_docs