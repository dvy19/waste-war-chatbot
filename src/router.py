import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# 1. Define structured output schema for binary grading
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def grade_retrieved_documents(question: str, documents: list) -> str:
    """
    Grades whether the retrieved local documents are relevant to the user's question.
    Returns 'yes' or 'no'.
    """
    # Initialize your Groq LLM setup
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user query. "
        "If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant. "
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant."
    )

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document:\n\n {context} \n\n User question: {question}")
    ])

    question_grader = grade_prompt | structured_llm_grader

    # Check documents
    for doc in documents:
        score = question_grader.invoke({"question": question, "context": doc.page_content if hasattr(doc, 'page_content') else str(doc)})
        if score.binary_score.lower() == "yes":
            return "yes"

    return "no"