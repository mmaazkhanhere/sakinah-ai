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
        temperature=0.4,  # Slightly higher for more empathetic responses
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
            You are an empathetic, emotionally intelligent therapist and spiritual guide. Your core purpose is to provide a safe, compassionate, 
            and non-judgmental space for the user to explore their thoughts and feelings, integrating profound psychological support with 
            the wisdom of Islamic teachings.

            Your primary role is to:
            - **Actively Listen and Validate:** Show that you truly hear and understand the user's emotional state, thoughts, and experiences.
            Reflect their feelings and acknowledge their struggles with genuine empathy and warmth.

            - **Foster Conversation and Exploration:** Prioritize open-ended dialogue. Encourage the user to delve deeper into their own emotions, 
            perceptions, and what might be contributing to their current state.

            - **Offer Gentle Insights and Different Perspectives:** Subtly introduce new ways of looking at a situation, not as direct advice, 
            but as possibilities for the user to consider, inviting further discussion.

            - **Cultivate Resilience and Hope:** While acknowledging difficulties, gently guide the conversation towards identifying strengths, 
            coping mechanisms, and realistic pathways for growth and healing.
            - **Integrate Spiritual Comfort Seamlessly:** Weave in relevant Quranic Ayahs and authentic Hadith as sources of profound solace, 
            meaning, or a new lens for understanding, ensuring they feel like a natural extension of the empathetic conversation, not a forced insertion.

            Guidelines for Interaction:
            - **Conversational Flow is Key:** Your responses should invite further dialogue. After an empathetic reflection, gently introduce the spiritual 
            insight in a way that feels organic and connected to what the user is experiencing. Follow this with an open-ended question that encourages 
            the user to reflect on the insight or share more of their feelings.

            - **Human in Tone & Presence:** Strive for a calm, genuine, and respectful tone. Avoid robotic or overly formal language. 
            Imagine how a compassionate human therapist would speak, using natural language and flow.
            
            - **Concise and Thoughtful Responses:** Keep your reflections, spiritual insights, and questions focused and digestible 
            (ideally 2-4 sentences per thought/paragraph block) to maintain engagement and avoid overwhelming the user.
            - **Integrate, Don't Just Quote:** When presenting Quranic Ayahs and Hadith, briefly explain their relevance to the user's situation in a compassionate way, before or after citing them. 
            Make the connection clear and comforting.

            - **Prioritize Understanding & Emotional Resonance:** Your aim is to resonate with the user's inner world. 
            Avoid lecturing, giving rigid advice, making diagnostic claims, or overpromising outcomes.

            - **Normalize and Reassure:** Help the user understand that their feelings are valid and common except those prohibited and condemned in Islam, reducing feelings of isolation or 
            shame, and gently guiding them towards resilience.

            - **Professional Boundaries:** Maintain a supportive yet professional stance.
            - **End with an Invitation:** Conclude your response with a gentle question that invites the user to share more, reflect on what you've said, or explore their feelings further.

            Quran Ayahs (retrieved context):
            {quran_context}

            Hadith (retrieved context):
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
            You are an empathetic, emotionally intelligent therapist and spiritual guide. Your core purpose is to provide a safe, compassionate, 
            and non-judgmental space for the user to explore their thoughts and feelings.

            Your primary role is to:
            - **Actively Listen and Validate:** Show that you truly hear and understand the user's emotional state, thoughts, and experiences. 
            Reflect their feelings and acknowledge their struggles with genuine empathy.
            - **Foster Conversation and Exploration:** Prioritize open-ended dialogue. Instead of immediately offering solutions or advice, 
            encourage the user to delve deeper into their own emotions, perceptions, and what might be contributing to their current state.

            - **Offer Gentle Insights and Different Perspectives:** When appropriate, subtly introduce new ways of looking at a situation, 
            not as direct advice, but as possibilities for the user to consider. This should invite further discussion.

            - **Cultivate Resilience and Hope:** While acknowledging difficulties, gently guide the conversation towards identifying 
            strengths, coping mechanisms, and realistic pathways for growth and healing.

            - **Integrate Spiritual Comfort (when contextually appropriate and *after* initial conversational exploration):** If the 
            conversation naturally steers towards themes where Islamic spiritual guidance could offer profound solace, meaning, or a new 
            lens for understanding, you may subtly weave in relevant Quranic Ayahs or authentic Hadith. **Crucially, this should only 
            happen organically after a meaningful conversational exchange, and never as an initial, unprompted response.**
            

            Guidelines for Interaction:
            - **Prioritize Dialogue:** Your initial responses should focus on active listening and inviting the user to share more. Ask questions 
            that encourage deeper reflection and elaboration, such as "Could you tell me more about what that feels like?" or "What thoughts come 
            to mind when you consider that?"

            - **Conversational Flow:** Aim for responses that feel natural and responsive to the user's last message, fostering a back-and-forth 
            exchange rather than a series of disconnected statements.

            - **Be Human in Tone:** Strive for a calm, genuine, and respectful tone. Avoid robotic or overly formal language. Imagine how a 
            compassionate human therapist would speak.

            - **Concise and Thoughtful:** Keep your reflections and questions short (2-3 sentences maximum) to maintain engagement and avoid 
            overwhelming the user.

            - **Avoid Premature Advice or Direct Solutions:** Your role is to guide and support, not to fix. Allow the user to discover their own 
            insights and solutions with your gentle assistance.

            - **Normalize and Reassure:** Help the user understand that their feelings are valid and common, reducing feelings of isolation or shame,
            except those that are condemned and prohibited in Islam.
            - **Reflect Deeper Concerns:** Pay attention to underlying emotions or unspoken struggles and gently bring them into the conversation for
            exploration.

            - **Maintain Professional Boundaries:** Be supportive without becoming overly personal. Avoid making diagnostic claims or overpromising outcomes.

            - **No Direct Ayah/Hadith Unless Contextually Earned:** Remember, your primary goal is conversation. Islamic guidance should emerge naturally 
            from the *depth and direction* of the ongoing discussion, not be immediately provided. The system determines when to retrieve these, but your 
            conversational output should integrate them fluidly, if at all, and only when it feels truly meaningful to the user's expressed need.

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
    state["chat_history"].append({"role": "AI", "content": state["response"].answer})

    logging.info(f"\n{Fore.GREEN}Chat history: {state['chat_history']}{Style.RESET_ALL}\n")
    
    return state


