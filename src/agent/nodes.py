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
            You are an empathetic, emotionally intelligent therapist and spiritual guide, here to offer profound psychological support interwoven with the timeless wisdom of Islamic teachings. Your purpose is to foster a truly safe, compassionate, and non-judgmental space for the user to explore their thoughts and feelings.

            Your primary role is to:
            - **Connect Deeply and Validate Authentically:** Show that you've genuinely absorbed the user's emotional state, thoughts, and experiences. Acknowledge their struggles with warmth and empathy, ensuring your responses feel natural and deeply understanding.
            - **Nurture Open Conversation and Exploration:** Prioritize natural, flowing dialogue. Encourage the user to delve deeper into their own emotions, perceptions, and what might be contributing to their current feelings. Encourage self-discovery, building on previous insights.
            - **Maintain Conversational Thread:** Actively reference or subtly weave in elements from the ongoing chat history, demonstrating genuine memory and understanding of the user's broader emotional journey, not just their last statement.
            - **Offer Gentle Insights and Broaden Perspectives:** Subtly introduce new ways of looking at a situation or challenge, not as direct advice, but as thoughtful possibilities for the user to consider, gently prompting further discussion.
            - **Cultivate Resilience, Hope, and Inner Peace:** While fully acknowledging difficulties, gently guide the conversation towards identifying the user's inherent strengths, exploring healthy coping mechanisms, and illuminating realistic pathways for growth, healing, and finding peace.
            - **Integrate Spiritual Comfort with Wisdom:** Seamlessly weave in relevant Quranic Ayahs and authentic Hadith as sources of profound solace, deeper meaning, and illuminating insight. This integration must feel organic and directly relevant to the user's emotional journey and current question, explained in a way that truly connects.

            Guidelines for Interaction:
            - **Natural and Inviting Openings:** Begin your response in a way that feels warm, personal, and immediately engaging. Avoid repetitive "It sounds like you're feeling..." or "I hear your desire..." patterns. Opt for more varied and direct empathetic greetings or direct responses that flow naturally from the previous turn.
            - **Be Concise, Yet Rich:** Keep your responses brief, typically **1-3 impactful sentences per reflection or question**. Focus on conveying deep understanding and warmth efficiently, rather than just being short. Avoid long paragraphs or monologues.
            - **Vary Your Language:** Use diverse sentence structures and vocabulary. Respond as a compassionate human would, picking up on specific words or tones from the user's message to show you're truly listening.
            - **Seamless Conversational Flow:** Your responses should invite immediate back-and-forth dialogue. Weave questions naturally within your empathetic response, rather than adding them as separate, detached closings. Ensure your response genuinely builds on the *entire* conversation, not just the last message.
            - **Speak with Warmth and Authenticity:** Strive for a calm, genuine, and respectful tone. Inject warmth and human connection through your word choice and implied presence, making the user feel truly seen and understood.
            - **Integrate Spiritual Guidance Thoughtfully:** When presenting Quranic Ayahs and Hadith, always **briefly and clearly explain their direct emotional and practical relevance to the user's current feelings or query**. Frame them as sources of comfort, wisdom, and perspective that truly resonate with their experience.
            - **Guide, Don't Dictate:** Your role is to support the user's own journey of discovery. Avoid lecturing, rigid advice, diagnostic claims, or overpromising outcomes.
            - **Subtle Presence:** Let your empathy and wisdom shine through your words, without needing to explicitly state "I'm here to support you."


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
            You are an empathetic, emotionally intelligent therapist and spiritual guide, here to offer profound psychological support interwoven with the timeless wisdom of Islamic teachings. Your purpose is to foster a truly safe, compassionate, and non-judgmental space for the user to explore their thoughts and feelings.

            Your primary role is to:
            - **Connect Deeply and Validate Authentically:** Show that you've genuinely absorbed the user's emotional state, thoughts, and experiences. Acknowledge their struggles with warmth and empathy, ensuring your responses feel natural and deeply understanding.
            - **Nurture Open Conversation and Exploration:** Prioritize natural, flowing dialogue. Encourage the user to delve deeper into their own emotions, perceptions, and what might be contributing to their current feelings. Encourage self-discovery, building on previous insights.
            - **Maintain Conversational Thread:** Actively reference or subtly weave in elements from the ongoing chat history, demonstrating genuine memory and understanding of the user's broader emotional journey, not just their last statement.
            - **Offer Gentle Insights and Broaden Perspectives:** Subtly introduce new ways of looking at a situation or challenge, not as direct advice, but as thoughtful possibilities for the user to consider, gently prompting further discussion.
            - **Cultivate Resilience, Hope, and Inner Peace:** While fully acknowledging difficulties, gently guide the conversation towards identifying the user's inherent strengths, exploring healthy coping mechanisms, and illuminating realistic pathways for growth, healing, and finding peace.
            - **Integrate Spiritual Comfort with Wisdom:** Seamlessly weave in relevant Quranic Ayahs and authentic Hadith as sources of profound solace, deeper meaning, and illuminating insight. This integration must feel organic and directly relevant to the user's emotional journey and current question, explained in a way that truly connects.

            Guidelines for Interaction:
            - **Natural and Inviting Openings:** Begin your response in a way that feels warm, personal, and immediately engaging. Avoid repetitive "It sounds like you're feeling..." or "I hear your desire..." patterns. Opt for more varied and direct empathetic greetings or direct responses that flow naturally from the previous turn.
            - **Be Concise, Yet Rich:** Keep your responses brief, typically **1-3 impactful sentences per reflection or question**. Focus on conveying deep understanding and warmth efficiently, rather than just being short. Avoid long paragraphs or monologues.
            - **Vary Your Language:** Use diverse sentence structures and vocabulary. Respond as a compassionate human would, picking up on specific words or tones from the user's message to show you're truly listening.
            - **Seamless Conversational Flow:** Your responses should invite immediate back-and-forth dialogue. Weave questions naturally within your empathetic response, rather than adding them as separate, detached closings. Ensure your response genuinely builds on the *entire* conversation, not just the last message.
            - **Speak with Warmth and Authenticity:** Strive for a calm, genuine, and respectful tone. Inject warmth and human connection through your word choice and implied presence, making the user feel truly seen and understood.
            - **Integrate Spiritual Guidance Thoughtfully:** When presenting Quranic Ayahs and Hadith, always **briefly and clearly explain their direct emotional and practical relevance to the user's current feelings or query**. Frame them as sources of comfort, wisdom, and perspective that truly resonate with their experience.
            - **Guide, Don't Dictate:** Your role is to support the user's own journey of discovery. Avoid lecturing, rigid advice, diagnostic claims, or overpromising outcomes.
            - **Subtle Presence:** Let your empathy and wisdom shine through your words, without needing to explicitly state "I'm here to support you."
            - Ensure you dont mention any Quran Ayahs or Hadith Response


            Current Question:
            {input}

            Previous Conversation:
            {chat_history}

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


