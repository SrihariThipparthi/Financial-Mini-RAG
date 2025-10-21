from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from data_loader import DataLoader
from retriever import MiniRAGRetriever
from config import config

data_loader = DataLoader(config.FAQS_PATH, config.PERFORMANCE_PATH)
documents = data_loader.load_all_data()
retriever = MiniRAGRetriever(config)
retriever.build_indices(documents)

app = FastAPI(title="Financial RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    retrieval_mode: str = "semantic"
    top_k: Optional[int] = None

class DocumentResponse(BaseModel):
    id: str
    content: str
    type: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: List[DocumentResponse]
    retrieval_mode: str

def generate_answer(query: str, retrieved_docs: List[Dict]) -> str:
    fund_docs = [doc for doc in retrieved_docs if doc['type'] == 'fund']
    faq_docs = [doc for doc in retrieved_docs if doc['type'] == 'faq']
    
    answer_parts = []
    
    if any(keyword in query.lower() for keyword in ['fund', 'return', 'performance', 'sharpe', 'volatility', 'cagr']):
        if fund_docs:
            answer_parts.append("Based on the available fund performance data:")
            for doc in fund_docs[:3]:
                answer_parts.append(f"- {doc['content']}")
        else:
            answer_parts.append("No specific fund performance data found for your query.")
    
    if faq_docs or not fund_docs:
        if faq_docs:
            answer_parts.append("\nRelated information:")
            for doc in faq_docs[:3]:
                metadata = doc.get('metadata', {})
                answer = metadata.get('answer', doc['content'])
                answer_parts.append(f"- {answer}")
        else:
            answer_parts.append("\nNo specific FAQ information found for your query.")
    
    if not answer_parts:
        return "I couldn't find specific information to answer your question. Please try rephrasing your query."
    
    return "\n".join(answer_parts)

@app.get("/")
async def root():
    return {"message": "Financial RAG API", "status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query_financial_data(request: QueryRequest):
    try:
        if request.retrieval_mode not in ["semantic", "lexical", "hybrid"]:
            raise HTTPException(status_code=400, detail="Invalid retrieval mode")
        
        retrieved_docs = retriever.retrieve(
            query=request.query,
            mode=request.retrieval_mode,
            top_k=request.top_k
        )
        
        answer = generate_answer(request.query, retrieved_docs)
        
        response = QueryResponse(
            answer=answer,
            retrieved_sources=[
                DocumentResponse(
                    id=doc['id'],
                    content=doc['content'],
                    type=doc['type'],
                    score=doc['score'],
                    metadata=doc.get('metadata', {})
                ) for doc in retrieved_docs
            ],
            retrieval_mode=request.retrieval_mode
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats")
async def get_stats():
    faq_count = len([doc for doc in documents if doc['type'] == 'faq'])
    fund_count = len([doc for doc in documents if doc['type'] == 'fund'])
    
    return {
        "total_documents": len(documents),
        "faq_count": faq_count,
        "fund_count": fund_count,
        "retrieval_modes": ["semantic", "lexical", "hybrid"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)