# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from typing import Optional


# Load environment variables
dotenv.load_dotenv()

app = FastAPI(title="Anime Gemini NLP API")

# Path to your Chroma DB
CHROMA_DIR = "chroma_db"



# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyA-cNhADhdhzBcImJYWp6wb9pmexi0TgWY")

# ✅ Initialize embedding model & vector database
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_model)

# ----- Request/Response Models -----
class QueryRequest(BaseModel):
    query: str
    top_k: int = 4


class SearchResult(BaseModel):
    name: Optional[str] = None
    text: str
    metadata: dict

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]

# ----- API Endpoints -----
@app.get("/")
def root():
    return {"message": "✅ Anime Gemini NLP API is running"}

@app.post("/answer", response_model=AnswerResponse)
def answer_query(q: QueryRequest):
    # Retrieve similar documents
    docs = vectordb.similarity_search(q.query, k=q.top_k)
    context = "\n\n".join([d.page_content for d in docs])

    # Prompt for Gemini
    prompt = f"""
You are an expert anime recommender. Use the dataset context below to answer the user's question.

Context:
{context}

User Question: {q.query}

Provide a concise, factual, and clear answer.
"""

    # Generate response from Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Prepare source data
    sources = [
        {"name": d.metadata.get("name"), "text": d.page_content, "metadata": d.metadata}
        for d in docs
    ]

    return AnswerResponse(query=q.query, answer=answer, sources=sources)
