from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

def load_documents():

    '''
    loader1=PyPDFLoader()
    loader2=PyPDFLoader()

    document1=loader1.load()
    document2=loader2.load()

    documents=document1+document2

    return documents
    '''

    paths=[DATA_DIR/"solid_waste.pdf" , DATA_DIR/"3R_handi.pdf"]

    documents=[]

    for p in paths:

        loader=PyPDFLoader(p)
        documents.extend(loader.load())

    return documents