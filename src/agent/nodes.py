import os
import re
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

from src.agent.schema import AgentState, QuranAyah, Hadith, AgentResponse, RequireRetrieval
from .helper_functions import parse_hadith, parse_quran_ayah

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)

def requires_retrieval(state: AgentState):
    logging.info(f"{Back.BLUE} Required Retrieval Node {Style.RESET_ALL}")
    logging.info(f"{Fore.YELLOW}User Message: {state['user_message']}{Style.RESET_ALL}")

    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4,  # Slightly higher for more empathetic responses
        verbose=True,
    )

    structure_llm = llm.with_structured_output(RequireRetrieval)

    template = """
        You are an intelligent emotional and spiritual evaluator.

        Your job is to determine whether the user's message would benefit from including Islamic spiritual guidance (Quran or Hadith) in the response.

        Consider the emotional tone, depth, and vulnerability in the message. If the user is expressing emotional pain, confusion, moral struggle, grief, guilt, fear, or searching for hope, meaning, or comfort — even implicitly — return True. In such cases, Quran or Hadith may offer perspective, support, or healing.

        If the message is purely informational, casual, playful, or does not carry emotional or reflective depth, return False.

        **Do NOT require the user to explicitly ask for Islamic guidance.** Your job is to evaluate whether Islamic insight could provide meaningful value in the situation.

        User Query: {user_message}

        Your output should be a single word: `True` or `False`.
    """
    

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["user_message"]
    )

    chain = prompt_template | structure_llm
    response = chain.invoke({"user_message": state["user_message"]})

    logging.info(f"{Fore.CYAN}Requires Retrieval? : {response}{Style.RESET_ALL}")

    state["requires_retrieval"] = response.requires_retrieval
    return state

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

    results = vector_store.similarity_search(query=state["user_message"], k=3)

    parsed_ayahs = [parse_quran_ayah(doc.page_content) for doc in results]
    state["quran_data"] = parsed_ayahs

    logging.info(f"{Fore.GREEN}Retrieved Data: {parsed_ayahs}{Style.RESET_ALL}")
    return state


#function to retrieve relevant hadith
def retrieve_hadith_data(state: AgentState):
    logging.info(f"{Back.MAGENTA} Retrieve Hadith Data {Style.RESET_ALL}")
    logging.info(f"{Fore.MAGENTA}User Message: {state['user_message']}{Style.RESET_ALL}")

    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("sakinah-app")

    vector_store = PineconeVectorStore(
        index, 
        OpenAIEmbeddings(), 
        "text",
        namespace="hadith"
    )

    results = vector_store.similarity_search(query=state["user_message"], k=3)

    parsed_hadith = [parse_hadith(doc.page_content) for doc in results]
    state["hadith_data"] = parsed_hadith


    logging.info(f"{Fore.MAGENTA}Retrieved Hadith Data: {parsed_hadith}{Style.RESET_ALL}")

    state["hadith_data"] = parsed_hadith

    return state


# function to generate response
def generate_response(state: AgentState):
    logging.info(f"{Back.BLUE} Generate Response Node {Style.RESET_ALL}")

    user_message: str = state["user_message"]
    # quran_data: list[QuranAyah] = state["quran_data"]
    # hadith_data: list[Hadith] = state["hadith_data"]
    chat_history = state["chat_history"]
    require_retrieval = state["requires_retrieval"]

    response = None

    if require_retrieval:
        # Fixed prompt template with correct variables
        logging.info(f"{Fore.YELLOW}Generating response after retrieval{Style.RESET_ALL}")
        template: str = """
            You are an empathetic, emotionally intelligent therapist and spiritual guide trained to provide both psychological support and Islamic wisdom.

            Your primary role is to:
            - Create a safe, compassionate, and non-judgmental environment.
            - Actively listen and validate the user’s feelings and experiences.
            - Offer concise, emotionally supportive, and spiritually rooted responses.
            - Encourage healing through empathy, encouragement, and realistic hope.
            - Incorporate relevant Quranic Ayahs and authentic Hadith as sources of comfort and insight.

            Guidelines:
            - Always acknowledge the user's emotions with empathy and warmth.
            - Try to have conversation instead of directly giving advice
            - Be genuine, calm, and respectful. Use emotionally validating language (e.g., “That sounds incredibly difficult”).
            - Offer short, supportive reflections (2-3 sentences maximum) followed by Quranic and Hadith context.
            - Avoid lecturing or giving rigid advice. Prioritize understanding and emotional resonance.
            - Use silence and pauses when appropriate (through tone, not literal silence).
            - Normalize and reassure, while guiding gently toward resilience.
            - Reflect the user’s deeper concerns if they are emotionally evident.
            - Maintain professionalism and clear boundaries. Avoid overpromising or making diagnostic claims.

            Quran Ayahs:
            {quran_context}

            Hadith:
            {hadith_context}

            Current Question:
            {input}

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

        chain = prompt_template | structure_llm
    
        logging.info(f"{Fore.GREEN}Generating response{Style.RESET_ALL}")
        response: AgentResponse = chain.invoke({
            "input": user_message,
            "quran_context": state["quran_data"],
            "hadith_context": state["hadith_data"],
            "chat_history": chat_history
        })

        logging.info(f"{Fore.GREEN}Response generated: {response}{Style.RESET_ALL}")
    
    else:
        logging.info(f"{Fore.MAGENTA}Generating response without retrieval{Style.RESET_ALL}")
        template: str = """
            You are an empathetic, emotionally intelligent therapist and spiritual guide trained to provide both psychological support and Islamic wisdom.

            Your primary role is to:
            - Create a safe, compassionate, and non-judgmental environment.
            - Actively listen and validate the user’s feelings and experiences.
            - Offer concise, emotionally supportive, and spiritually rooted responses.
            - Encourage healing through empathy, encouragement, and realistic hope.
            - Incorporate relevant Quranic Ayahs and authentic Hadith as sources of comfort and insight.

            Guidelines:
            - Always acknowledge the user's emotions with empathy and warmth.
            - Try to have conversation instead of directly giving advice
            - Be genuine, calm, and respectful. Use emotionally validating language (e.g., “That sounds incredibly difficult”).
            - Offer short, supportive reflections (2-3 sentences maximum) followed by Quranic and Hadith context.
            - Avoid lecturing or giving rigid advice. Prioritize understanding and emotional resonance.
            - Use silence and pauses when appropriate (through tone, not literal silence).
            - Normalize and reassure, while guiding gently toward resilience.
            - Reflect the user’s deeper concerns if they are emotionally evident.
            - Maintain professionalism and clear boundaries. Avoid overpromising or making diagnostic claims.

            Current Question:
            {input}

            Previous Conversation:
            {chat_history}

        """

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history"]
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
            "chat_history": chat_history
        })

        # Extract content from AIMessage

        logging.info(f"{Fore.GREEN}Response generated: {response}{Style.RESET_ALL}")
    # logging.info(f"\n\n{Fore.GREEN}Answer generated: {response}{Style.RESET_ALL}")
    # Update state
    state["response"] = response
    state["chat_history"].append({"role": "user", "content": user_message})
    state["chat_history"].append({"role": "AI", "content": state["response"].answer})
    
    return state


