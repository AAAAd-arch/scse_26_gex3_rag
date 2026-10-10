### COMPLETE THE CODE  ###

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from policy_loader import load_policy_documents


## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="university_it_support",
    embedding_function=embeddings
)

batch_size = 20

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    vector_store.add_documents(batch)


def retrieve_documents(question):
    return vector_store.similarity_search(
        question,
        k=2
    )