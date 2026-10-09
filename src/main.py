from document_loader import load_documents
from text_splitter import  split_documents

res=load_documents()

#print(res[0].page_content)

'''
for doc in res:
    print(doc.page_content)
'''

chunks=split_documents(documents=res)

print(len(chunks))
#print(chunks)

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

