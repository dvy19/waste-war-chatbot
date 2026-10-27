from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from EmbeddingManager import EmbeddingManager

from VectorStore import VectorStore
from Generator import Generator

embeddingManager=EmbeddingManager()
vectorStore=VectorStore()
generator=Generator()

app=FastAPI()

class QueryRequest(BaseModel):
    question:str

class QueryResponse(BaseModel):
    question:str
    answer:str
    source:str

@app.get("/")
def home():
    return {"message":"waste managemenet rag api is working"}


@app.post("/ask" , response_model=QueryResponse)
def ask_question(request:QueryRequest):

    query=request.question.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="question can not be empty"
        )
    results=vectorStore.search(query)

    retrieved_chunks=results["documents"][0]

    if retrieved_chunks:
        answer=generator.generate(query , retrieved_chunks)
    else:
        answer="i could not find the answer in the document"

    source="chromadb"

    return QueryResponse(
        question=query,
        answer=answer,
        source=source
    )



# uvicorn main:app --reload