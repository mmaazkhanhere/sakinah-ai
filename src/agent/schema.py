from typing import Annotated, Dict
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


class QuranAyah(BaseModel):
    surah_no: int = Field(description="The surah number of the ayah")
    surah_name: str = Field(description="The name of the surah")
    ayah_no_surah: int = Field(description="The ayah number in the surah")
    ayah_eng: str = Field(description="The English translation of the ayah")
    ayah_arabic: str = Field(description="The Arabic text of the ayah")

class Hadith(BaseModel):
    hadith: str = Field(description="The text of the hadith")
    hadith_number: int = Field(description="The number of the hadith")
    narrator: str = Field(description="The narrator of the hadith")
    book_reference: str = Field(description="The book reference of the hadith")



class AgentResponse(BaseModel):
    answer: str = Field(description="The response of AI")
    quran_ayah: QuranAyah = Field(description="The quran ayah")
    hadith: Hadith = Field(description="The hadith")

class AgentState(TypedDict):
    user_message: str  #the message entered by the user
    context: list[str] # context the is  built using rag
    chat_history: list[Dict[str, str]] # history of chat between the agent and the user
    response: AgentResponse # generated the answer by the LLM