from document_loader import load_documents
from text_splitter import  split_documents

from EmbeddingManager import EmbeddingManager

from VectorStore import VectorStore
res=load_documents()

from Generator import Generator

#print(res[0].page_content)

'''
for doc in res:
    print(doc.page_content)
'''

chunks=split_documents(documents=res)

print(len(chunks))
#print(chunks)

'''
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)

'''

texts=[]

for chunk in chunks:
    texts.append(chunk.page_content)

embeddingManager=EmbeddingManager()

embeddings=embeddingManager.generate_embeddings(texts)


vectorStore=VectorStore()

vectorStore.add_documents(texts , embeddings)

print(embeddings.shape)

query = "How does recycling help manage waste?"

results = vectorStore.search(query)

print("User question:", query)
print("\nRetrieved chunks:")

retrieved_chunks = results["documents"][0]

for chunk in retrieved_chunks:
    print(chunk)

generator = Generator()

answer = generator.generate(query, retrieved_chunks)

print("\nGenerated Answer:")
print(answer)




