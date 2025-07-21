import os
import pinecone
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def retrieve_from_rag(user_query: str, index_name: str, top_k: int = 3, namespace: str = "") -> list[str]:
    """
    Retrieve relevant documents from pinecone vector store based on user query

    Args: 
    user_query: input from user
    index_name: name of pinecone index
    top_k: number of documents to retrieve

    returns: list of document contents matching the query
    """

    pc = pinecone.Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index, 
        OpenAIEmbeddings(), 
        "text"  # Metadata field where text is stored
    )


    results = vector_store.similarity_search(query=user_query, k=top_k, namespace=namespace)

    return [doc.page_content for doc in results]

print(retrieve_from_rag("What is the meaning of life?", "sakinah-app"))