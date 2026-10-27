import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from state import GraphState
from nodes import (
    retrieve_node,
    grade_documents_node,
    web_search_node,
    generate_node,
    route_decision
)

# Load environment variables (.env)
load_dotenv()

def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Use route_decision for conditional routing
    workflow.add_conditional_edges(
        "grade_documents",
        route_decision,
        {
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    app = workflow.compile()
    return app


if __name__ == "__main__":
    print("Initializing WasteWar Agentic RAG Chatbot...")
    app = build_graph()

    # Interactive CLI loop to test your Agentic RAG
    print("\nChatbot is ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("User Question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        if not user_input.strip():
            continue

        # Initial state payload
        initial_state = {
            "question": user_input,
            "generation": "",
            "documents": []
        }

        print("\n--- RUNNING AGENTIC WORKFLOW ---")
        try:
            # Invoke the LangGraph app
            final_state = app.invoke(initial_state)

            print("\n==============================")
            print("FINAL ANSWER:")
            print(final_state.get("generation"))
            print("==============================\n")

        except Exception as e:
            print(f"\nAn error occurred during execution: {e}\n")