from document_loader import load_documents
from text_splitter import  split_documents

from EmbeddingManager import EmbeddingManager
from VectorStore import VectorStore
from Generator import Generator

# from VectorStore import VectorStore
res=load_documents()

# from Generator import Generator

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
embeddingManager = EmbeddingManager()

embeddings = embeddingManager.generate_embeddings(texts)

print("Embeddings shape:", embeddings.shape)

# Step 2: Store documents and embeddings
vectorStore = VectorStore(embeddingManager)

vectorStore.add_documents(texts, embeddings)

# Step 3: Retrieve relevant chunks
query = "How does recycling help manage waste?"

results = vectorStore.search(query)

print("\nUser question:", query)
print("\nRetrieved chunks:")

retrieved_chunks = results["documents"][0] if results["documents"] else []

for chunk in retrieved_chunks:
    print(chunk)

# Step 4: Generate an answer
generator = Generator()

if retrieved_chunks:
    answer = generator.generate(query, retrieved_chunks)
    print("\nGenerated Answer:")
    print(answer)
else:
    print("No relevant chunks found.")



