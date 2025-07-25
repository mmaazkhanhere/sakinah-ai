from typing import Annotated, Dict
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

class RequireRetrieval(BaseModel):
    requires_retrieval: bool = Field(description="Check whether the user query requires retrieval")

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
    answer: str = Field(description="The therapy response without Quran Ayah and Hadith")
    quran_ayah: QuranAyah | None = Field(description="The quran ayah")
    hadith: Hadith | None = Field(description="The hadith")

class AgentState(TypedDict):
    user_message: str  #the message entered by the user
    requires_retrieval: bool
    quran_data: list[QuranAyah] | None
    hadith_data: list[Hadith] | None
    chat_history: list[Dict[str, str]] # history of chat between the agent and the user
    response: AgentResponse # generated the answer by the LLM