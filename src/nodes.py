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
    if search_results and "documents" in search_results:
        nested_docs = search_results["documents"]
        if nested_docs and len(nested_docs) > 0:
            documents = nested_docs[0]

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

    # Call your Tavily search method (returns list[str])
    search_results = tavily_search.search(question)

    # Extend the documents list with the web search results strings
    if isinstance(search_results, list):
        documents.extend(search_results)
    else:
        documents.append(str(search_results))

    return {"documents": documents}


def generate_node(state: dict):
    """
    Generates the final answer using the Groq LLM and available context.
    """
    print("---NODE: GENERATING FINAL ANSWER---")
    question = state["question"]
    documents = state["documents"]

    generation = generator.generate(question, documents)

    return {"generation": generation}