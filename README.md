# Mini RAG with Financial Data

A simple but robust Retrieval-Augmented Generation system for financial data, handling both textual FAQs and numerical fund performance data.

## Design

### Architecture
1. **Data Layer**: Loads and processes both CSV datasets
2. **Retrieval Layer**: 
   - Semantic search using FAISS + Sentence Transformers
   - Lexical search using TF-IDF
   - Hybrid search combining both approaches
3. **API Layer**: FastAPI endpoint for queries

### Key Components
- **DataLoader**: Handles CSV ingestion and document creation
- **MiniRAGRetriever**: Implements all retrieval strategies
- **FastAPI App**: Provides REST API interface

### Document Structure
Each document contains:
- `id`: Unique identifier
- `content`: Textual representation
- `type`: 'faq' or 'fund'
- `metadata`: Original data fields
- `score`: Relevance score (added during retrieval)

## Trade-offs

1. **Simplicity vs. Performance**: 
   - Uses lightweight models for faster inference
   - Could be enhanced with larger models for better accuracy

2. **Memory vs. Speed**:
   - FAISS for fast similarity search
   - TF-IDF matrices stored in memory

3. **Hybrid Search**:
   - Simple weighted combination
   - Could use more sophisticated fusion techniques

## Assumptions

1. **Data Format**: CSVs with expected column names
2. **Query Types**: Mix of factual and performance questions
3. **Scale**: Suitable for hundreds to thousands of documents
4. **Hardware**: CPU-only environment (FAISS CPU version)

## Setup Instructions

1. **Install dependencies**:
```bash
pip install -r requirements.txt
 ```

2. **Run the app**
 ```bash
 python app.py
 ```

3. **Endpoint Health Check**
```bash
curl -X GET "http://127.0.0.1:8000/stats"
```

4. **To query the endpoint**
 ```bash
 curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which funds have the best Sharpe ratio?",
    "retrieval_mode": "hybrid",
    "top_k": 5
  }'

```