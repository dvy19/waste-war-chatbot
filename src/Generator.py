
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class Generator:

    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question using only the provided context.

        If the answer is not available in the context,
        say: "I could not find the answer in the document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

    def generate(self, question: str, retrieved_chunks: list[str]):

        context = "\n\n".join(retrieved_chunks)

        messages = self.prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)

        return response.content