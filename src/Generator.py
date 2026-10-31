
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Define structured output schema for binary document grading
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class Generator:

    def __init__(self):
        # Initialize your Groq LLM once here
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Standard generation prompt
        self.generate_prompt = ChatPromptTemplate.from_template("""
            You are an expert assistant. Answer the question using **only** the provided context sources. 
            Whenever you use information from a source, cite it explicitly using its source number in brackets (e.g., [1], [2]).
            
            If the answer is not available in the context, say: 
            "I could not find the answer in the provided sources or web search."
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
        """)

        # Grading prompt template
        system_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user query. "
            "If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant. "
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant."
        )

        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved document:\n\n {context} \n\n User question: {question}")
        ])

    def grade_documents(self, question: str, retrieved_chunks: list[str]) -> str:
        """
        Grades whether the retrieved local documents are relevant to the user's question.
        Returns 'yes' or 'no'.
        """
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        question_grader = self.grade_prompt | structured_llm_grader

        # Check each retrieved chunk
        for doc in retrieved_chunks:
            doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            score = question_grader.invoke({"question": question, "context": doc_text})
            if score.binary_score.lower() == "yes":
                return "yes"

        return "no"

    def generate(self, question: str, retrieved_chunks: list[str]) -> str:
        """
        Generates a final response based on the context and question.
        """
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_chunks])

        messages = self.generate_prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)

        return response.content