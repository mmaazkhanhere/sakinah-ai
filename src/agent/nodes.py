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

from src.agent.schema import AgentState, QuranAyah, Hadith, AgentResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)

def parse_quran_ayah(data: str) -> QuranAyah:
    lines = data.split('\n')
    metadata = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = value
        elif not line.strip():
            break
    
    return QuranAyah(
        surah_no=int(metadata.get('surah_no', 0)),
        surah_name=metadata.get('surah_name_en', ''),
        ayah_no_surah=int(metadata.get('ayah_no_surah', 0)),
        ayah_eng=metadata.get('ayah_en', ''),
        ayah_arabic=metadata.get('ayah_ar', '')
    )

def parse_hadith(data: str) -> Hadith:
    header, _, body = data.partition('\n')
    narrator = ""
    hadith_text = body.strip()
    
    # Extract hadith number and book reference
    hadith_number = 0
    book_ref = ""
    if 'Number' in header:
        num_start = header.find('Number') + 6
        num_end = header.find(':', num_start)
        if num_end == -1:
            num_end = len(header)
        hadith_number = int(''.join(filter(str.isdigit, header[num_start:num_end])))
        book_ref = header.split(':')[0].replace(f'Number {hadith_number}', '').strip()
    
    # Extract narrator
    if body.startswith('Narrated'):
        narrator_line = body.split('\n')[0]
        narrator = narrator_line.replace('Narrated', '').strip()
    
    return Hadith(
        hadith=hadith_text,
        hadith_number=hadith_number,
        narrator=narrator,
        book_reference=book_ref
    )


# function to retrieve quran ayahs
def retrieve_quran_data(state: AgentState):
    logging.info(f"{Back.BLUE} Retrieve Quran Data {Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}User Message: {state['user_message']}{Style.RESET_ALL}")

    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("sakinah-app")

    vector_store = PineconeVectorStore(
        index, 
        OpenAIEmbeddings(), 
        "text",
        namespace="quran"
    )

    results = vector_store.similarity_search(query=state["user_message"], k=5)

    parsed_ayahs = [parse_quran_ayah(doc.page_content) for doc in results]
    state["quran_data"] = parsed_ayahs

    logging.info(f"{Fore.GREEN}Retrieved Data: {parsed_ayahs}{Style.RESET_ALL}")
    return state



#function to retrieve relevant hadith
def retrieve_hadith_data(state: AgentState):
    logging.info(f"{Back.BLUE} Retrieve Hadith Data {Style.RESET_ALL}")
    logging.info(f"{Fore.CYAN}User Message: {state['user_message']}{Style.RESET_ALL}")

    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("sakinah-app")

    vector_store = PineconeVectorStore(
        index, 
        OpenAIEmbeddings(), 
        "text",
        namespace="hadith"
    )

    results = vector_store.similarity_search(query=state["user_message"], k=5)

    parsed_hadiths = [parse_hadith(doc.page_content) for doc in results]
    state["hadith_data"] = parsed_hadiths


    logging.info(f"{Fore.GREEN}Retrieved Data: {parsed_hadiths}{Style.RESET_ALL}")

    return state


# function to generate response
def generate_response(state: AgentState):
    logging.info(f"{Back.BLUE} Generate Response Node {Style.RESET_ALL}")

    user_message: str = state["user_message"]
    quran_data: list[QuranAyah] = state["quran_data"]
    hadith_data: list[Hadith] = state["hadith_data"]
    chat_history = state["chat_history"]


    # Fixed prompt template with correct variables
    template: str = """You are an empathetic therapist and spiritual guide.
        Your role is to listen, acknowledge the user's feelings, and offer emotional healing and guidance.
        Always respond with compassion and understanding.
        Be precise and factual.

        Quran Ayahs:
        {quran_context}

        Hadith:
        {hadith_context}

        Current Question: {input}

        Previous Conversation:
        {chat_history}
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["quran_context", "hadith_context", "input", "chat_history"]
    )

    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4,  # Slightly higher for more empathetic responses
        verbose=True,
    )

    structure_llm = llm.with_structured_output(AgentResponse)

    # Create direct chain without retriever
    chain = prompt_template | structure_llm
    
    logging.info(f"{Fore.GREEN}Generating response{Style.RESET_ALL}")
    response: AgentResponse = chain.invoke({
        "input": user_message,
        "quran_context": quran_data,
        "hadith_context": hadith_data,
        "chat_history": chat_history
    })

    # Extract content from AIMessage

    logging.info(f"{Fore.GREEN}Response generated: {response}{Style.RESET_ALL}")
    # logging.info(f"\n\n{Fore.GREEN}Answer generated: {response}{Style.RESET_ALL}")
    # Update state
    agent_response = state["response"]
    agent_response = response
    state["chat_history"].append({"role": "user", "content": user_message})
    state["chat_history"].append({"role": "AI", "content": response.answer})
    
    return state