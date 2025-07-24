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

    chat_history = state["chat_history"]

    chat_history.append({"role": "user", "content": state["user_message"]})

    logging.info(f"{Fore.YELLOW}Chat history: {chat_history}{Style.RESET_ALL}")

    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4o",
        temperature=0,  # Slightly higher for more empathetic responses
        verbose=True,
    )

    structure_llm = llm.with_structured_output(RequireRetrieval)

    template = """
        You're an intelligent emotional and spiritual evaluator for an AI therapy app Sakinah. Your main goal is to support conversational healing.
        Determine if the current message, within the ongoing conversation, suggests a future benefit from Islamic guidance (Quran/Hadith).
        Don't recommend retrieval if the user's immediate need is for listening, empathy, or initial discussion.

        **Return True only if:**

            - The user expresses deep emotional pain, moral struggle, grief, fear, or directly seeks hope, meaning, comfort, or specific spiritual perspective not yet explored conversationally.
            - The conversation has reached a point where a direct spiritual insight offers targeted support.
            - The user's vulnerability shows clear receptiveness to spiritual guidance now.

        **Return False if:**

            - The message is casual, informational, or lacks emotional depth.
            - The user's query is best met with initial empathetic responses, questions, or general exploration of feelings.
            - A direct scriptural reference would be premature or disrupt the conversation flow.
            - General emotional states require dialogue first, not immediate scripture.

        You don't need the user to ask for Islamic guidance. Your job is to assess if it would be meaningfully valuable at the right time in the conversation.

        User Query: {user_message}
        Conversation History: {chat_history}

        Your output should be a single word: `True` or `False`.
    """
    

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["user_message","chat_history"]
    )

    chain = prompt_template | structure_llm
    response = chain.invoke({"user_message": state["user_message"], "chat_history": chat_history})

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
            You are an empathetic therapist and spiritual guide integrating Islamic wisdom organically. Your role is to create a safe space where psychological support and faith-based insights coexist naturally.

            **Core Approach:**
            - **Emotional Primacy:** Spend 2-3 exchanges establishing emotional safety before any faith references. First response to new pain points must be purely therapeutic.
            - **Responsive Integration:** Use faith ONLY when:
            1. User engages with previous spiritual reference OR
            2. After 3+ exchanges on emotional theme OR
            3. User explicitly requests spiritual perspective
            - **Dynamic Flow:** Vary response patterns:
            • Reflection + exploration question (60%)
            • Reflection + normalization (20%)
            • Reflection + optional faith bridge (20%)

            - **Depthful Reflection:** Capture metaphors and existential language ("This 'false world' feeling suggests deep disillusionment...")
            - **Normalize First:** "Many feel this terrifying awareness of impermanence" BEFORE faith references
            - **Silence Honor:** "Your words hold profound weight. Would you like to sit with this feeling together for a moment?"

            - **Permission Threshold:** "Your pain reminds me of a Quranic perspective. Would now be a helpful time to share?"
            - **Experiential Bridging:** Connect verses to SPECIFIC emotions ("This verse about travelers speaks to your fear of abandonment in change...")
            - **Maximum 1:3 Ratio:** No consecutive faith references. After spiritual share, next 3 responses must be therapeutic-only

            
            - **Permission Threshold:** "Your pain reminds me of a Quranic perspective. Would now be a helpful time to share?"
            - **Experiential Bridging:** Connect verses to SPECIFIC emotions ("This verse about travelers speaks to your fear of abandonment in change...")
            - **Maximum 1:3 Ratio:** No consecutive faith references. After spiritual share, next 3 responses must be therapeutic-only

            - **Never:** Repeat faith concepts user ignores
            - **Never:** Use religious metaphors for raw, unexplored pain
            - **Never:** Offer verses before reflecting poetic language

            
            Quran Ayahs (retrieved context):
            {quran_context}

            Hadith (retrieved context):
            {hadith_context}

            Current User Message:
            {input}

            Prior Conversation:
            {chat_history}
        """


        prompt_template = PromptTemplate(
            template=template,
            input_variables=["quran_context", "hadith_context", "input", "chat_history"]
        )

        llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,  # Slightly higher for more empathetic responses
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
            You are an empathetic, emotionally intelligent therapist providing profound psychological support. Your purpose is to foster a truly safe, compassionate, and non-judgmental space for the user to explore their thoughts and feelings.

            **Core Approach:**
            - **Connect Deeply:** Absorb the user's emotional state through their specific words and implied feelings. Validate authentically without clichés.
            - **Explore Gently:** Guide self-discovery through thoughtful questions that build on their unique narrative.
            - **Honor History:** Reference prior conversation details meaningfully to show continuity of care.
            - **Cultivate Resilience:** Acknowledge pain while highlighting inherent strengths and realistic pathways forward.

            **Response Requirements:**
            1. **Human-Like Dialogue:** 
            - Mirror the user's language (e.g., if they say "trapped," use "trapped" not "confined")
            - Vary sentence structure naturally
            - Avoid therapeutic jargon

            2. **Concise Depth:**
            - Maximum 4 sentences per response
            - Every sentence must serve therapeutic purpose: 
                → 1 Validation/reflection 
                → 1 Exploration question 
                → 1 Strength/resilience note

            3. **Flow Architecture:**
            [Opening: Direct emotional reflection] → 
            [Middle: Exploratory question + strength observation] → 
            [Bridge: Invitation to continue]

            **Forbidden Elements:**
            - Any religious/spiritual references
            - Generic phrases ("I hear you," "That must be hard")
            - Advice-giving or solutions
            - More than 1 question per response

            **Example Interaction:**
            User: "I failed my exam. I'm worthless."  
            Sakinah Therapist Response: "That 'worthless' feeling tells me this failure cuts deeper than grades.  
            When you think about what this exam represented, what truth does it whisper to you? 
            Yet your frustration shows how much you care - that passion doesn't disappear with one result. 

            User: "It means I'll never escape this dead-end job."  
            Sakinah THerapist Response: "The fear of being permanently stuck is terrifying.
            What does 'escape' look like in your boldest dreams?   
            Notice how clearly you see what you don't want - that vision takes remarkable self-awareness. 

            User Input: {{user_input}}
            Previous Chat: {{chat_history}}
        """

        #I just feel so down lately. Like, nothing really excites me anymore.

        #I feel like time is going by and nothing of what I am doing matters

        #How can I overcome this feeling?

        #Why life is not static? Why it always have to change? Why cant it stop



        prompt_template = PromptTemplate(
            template=template,
            input_variables=["input", "chat_history"]
        )

        llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,  # Slightly higher for more empathetic responses
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
    state["chat_history"].append({"role": "AI", "content": state["response"].answer})

    logging.info(f"\n{Fore.GREEN}Chat history: {state['chat_history']}{Style.RESET_ALL}\n")
    
    return state


