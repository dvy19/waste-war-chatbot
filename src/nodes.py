from src.VectorStore import VectorStore
from src.TavilySearch import TavilySearch
from src.Generator import Generator
from src.EmbeddingManager import EmbeddingManager

# Initialize components
embedding_manager = EmbeddingManager()
vector_store = VectorStore(embedding_manager=embedding_manager)
tavily_search = TavilySearch()
generator = Generator()
def retrieve_node(state: dict):
    """
    Retrieves documents from the local Chroma vector store based on the user question.
    """
    print("---NODE: RETRIEVING FROM LOCAL VECTOR STORE---")
    question = state["question"]

    search_results = vector_store.search(question, n_results=3)

    documents = []
    # 1. Safely check if search_results is not None and contains documents
    if search_results and isinstance(search_results, dict) and "documents" in search_results:
        texts = search_results["documents"]
        if texts and len(texts) > 0 and texts[0]:
            chunk_texts = texts[0]

            # 2. Safely extract metadata with a fallback if 'metadatas' is missing or None
            metadatas = search_results.get("metadatas")
            if not metadatas or not metadatas[0]:
                metadatas = [[{}] * len(chunk_texts)]

            meta_list = metadatas[0]

            for text, meta in zip(chunk_texts, meta_list):
                # Handle case where meta might be None
                safe_meta = meta if isinstance(meta, dict) else {}
                source_name = safe_meta.get("source", "Local PDF Document (3R_handi / solid_waste)")
                page_num = safe_meta.get("page", "Unknown Page")

                documents.append({
                    "content": text,
                    "source": f"{source_name} (Page {page_num})"
                })

    return {"documents": documents, "question": question}

def grade_documents_node(state: dict):
    """
    Node that grades whether the retrieved documents are relevant.
    Must return a state dictionary!
    """
    print("---NODE: CHECKING DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    score = generator.grade_documents(question, documents)

    # Store the grading score/decision in the state if needed, and return state dict
    return {"documents": documents, "router_decision": score}


def route_decision(state: dict) -> str:
    """
    Conditional edge router function. Inspects the state and returns
    the string name of the next node to execute.
    """
    decision = state.get("router_decision", "no")
    if decision == "yes":
        print("---DECISION: DOCUMENTS RELEVANT -> GENERATE---")
        return "generate"
    else:
        print("---DECISION: DOCUMENTS NOT RELEVANT -> WEB SEARCH---")
        return "web_search"

def web_search_node(state: dict):
    """
    Performs web search using Tavily when local documents are insufficient.
    """
    print("---NODE: RUNNING TAVILY WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    search_results = tavily_search.search(question)

    # Assuming search_results is a list of dicts from Tavily
    if search_results:
        for item in search_results:
            if isinstance(item, dict):
                documents.append({
                    "content": item.get("content", ""),
                    "source": f"Tavily Web Search: {item.get('title', 'Web Result')} ({item.get('url', '')})"
                })
            else:
                documents.append({
                    "content": str(item),
                    "source": "Tavily Web Search"
                })

    return {"documents": documents}

def generate_node(state: dict):
    """
    Generates the final answer using the Groq LLM and available context sources.
    """
    print("---NODE: GENERATING FINAL ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # Build an explicit context string with bracketed source numbers
    context_str = ""
    for i, doc in enumerate(documents, 1):
        if isinstance(doc, dict):
            context_str += f"\n--- Source [{i}]: {doc['source']} ---\n{doc['content']}\n"
        else:
            context_str += f"\n--- Source [{i}]: Local Document ---\n{str(doc)}\n"

    # Pass the formatted context string to your generator
    generation = generator.generate(question, context_str)

    return {"generation": generation}