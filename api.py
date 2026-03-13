import os
import vertexai
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel
import uvicorn

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
CHROMA_BASE_DIR = os.path.expanduser("~/.rag_demo_chroma")

vertexai.init(project=PROJECT_ID, location="us-central1")

api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    user_hash: str = ""

class QueryResponse(BaseModel):
    answer: str
    route: str

def query_with_rag(question: str, user_hash: str) -> tuple[str, str]:
    context = "No documents uploaded."
    route = "rag"
    
    if user_hash:
        persist_dir = os.path.join(CHROMA_BASE_DIR, user_hash)
        if os.path.exists(persist_dir):
            try:
                embeddings = VertexAIEmbeddings(
                    model_name="text-embedding-005",
                    project=PROJECT_ID,
                    location="us-central1"
                )
                chroma_client = chromadb.PersistentClient(path=persist_dir)
                collection_name = f"user_{user_hash}"
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name=collection_name,
                    embedding_function=embeddings,
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                if docs:
                    context = "\n\n".join(doc.page_content for doc in docs)
            except Exception as e:
                print(f"RAG error: {e}")

    prompt = f"""Answer the question based only on the context below.
Context:
{context}

Question: {question}
Answer:"""
    
    model = GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text, route

@api_app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    answer, route = query_with_rag(req.question, req.user_hash)
    return QueryResponse(answer=answer, route=route)

@api_app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8081, log_level="info")
