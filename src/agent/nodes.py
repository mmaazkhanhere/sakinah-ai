import os
import logging
import pinecone
from colorama import Fore, Style, Back
from dotenv import load_dotenv

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.agent.schema import AgentState

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)

def retrieve_data(state: AgentState):
    """
    Retrieve relevant documents from pinecone vector store based on user query
    """

    logging.info(f"{Back.BLUE} Retrieve Data Node {Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}User Message: {state['user_message']}{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}Chat History: {state['chat_history']}{Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}Context: {state['context']}{Style.RESET_ALL}")

    chat_history = state["chat_history"]
    chat_history.append({"role": "user", "content": state["user_message"]})

    user_message = state["user_message"]

    pc = pinecone.Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )
    index = pc.Index("sakinah-app")

    vector_store = PineconeVectorStore(
        index, 
        OpenAIEmbeddings(), 
        "text"  # Metadata field where text is stored
    )


    results = vector_store.similarity_search(query=user_message, k=5)

    context = [doc.page_content for doc in results]
    state["context"] = context

    logging.info(f"{Fore.GREEN}Retrieved Data: {context}{Style.RESET_ALL}")

    return state

# def retrieve_data_relevancy_analyze(state: AgentState):
#     return state

def generate_response(state: AgentState):
    """
    Generate response based on user query and retrieved documents
    """

    logging.info(f"{Back.BLUE} Generate Response Node {Style.RESET_ALL}")

    chat_history: list[dict[str, str]] = state["chat_history"]
    user_message: str = state["user_message"]
    context: list[str] = state["context"]

    logging.info(f"{Fore.CYAN}Creating a retriever")

    embeddings: OpenAIEmbeddings = OpenAIEmbeddings()
    vector_store: PineconeVectorStore = PineconeVectorStore(
        index_name="sakinah-app",
        embedding=embeddings,
        text_key="text",
        namespace="sakinah-app"
    )

    retriever: VectorStoreRetriever = vector_store.as_retriever(search_kwargs={"k": 5})

    logging.info(f"{Fore.GREEN}Retriever created{Style.RESET_ALL}")

    template: str = """You are an empathetic therapist and spiritual guide.
        Your role is to listen, acknowledge the user's feelings, and offer emotional healing and guidance using references from the Quran provided in the context below.
        Always respond with compassion and understanding.
        If you cannot find relevant information in the provided context, state that you are unable to provide specific guidance from the texts for this particular query, but offer general empathetic support. Be precise and factual.
        The output should be concise, ideally within 150-200 tokens (5-7 sentences), ensuring clarity and impact.

        Begin by acknowledging how the user feels. Then, gently guide them towards emotional and spiritual healing by explaining how the provided Quranic verses  offer perspective, comfort, or a path forward.
        When referencing the Quran, you MUST include the full Arabic text, followed by its English translation, and then the Surah name (English), Surah number, and Ayah number (e.g., "Surah Al-Fatihah (1:1)").
        ").

        Context:
        {context}

        Question: {input}

        Previous Chat Messages: {chat_history}
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["context", "user_query", "chat_history"]
    )

    # prompt = prompt_template.format(
    #     context=context,
    #     user_query=user_message,
    #     chat_history=chat_history
    # )

    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        verbose=True,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    logging.info(f"{Fore.GREEN}QA Chain created{Style.RESET_ALL}")
    logging.info(f"{Fore.BLUE}Generating response{Style.RESET_ALL}")
    response = qa_chain.invoke({
        "input": user_message,  # Key changed to match PromptTemplate
        "context": context,
        "chat_history": chat_history
    })

    logging.info(f"{Fore.GREEN}Response generated: {response['answer']}{Style.RESET_ALL}")


    state["answer"] = response["answer"]
    state["chat_history"].append({"role": "AI", "content": response["answer"]})
    return state