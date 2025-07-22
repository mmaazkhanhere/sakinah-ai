import re
import logging
from colorama import Fore, Style

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)
logger = logging.getLogger(__name__)

def chunk_hadiths_with_metadata(pdf_path):
    """
    Load a PDF document, identify individual hadiths, and chunk them
    along with their related metadata (Volume, Book, Number, Narrator).

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: List of document chunks, each representing a hadith
                        with its associated metadata.
    """
    logger.info(f"{Fore.GREEN}\n{'='*50}")
    print(f"{Fore.GREEN}Starting Hadith Chunking Pipeline")
    print(f"{Fore.GREEN}{'='*50}")

    print(f"{Fore.YELLOW}Source PDF: {pdf_path}\n")

    print(f"{Fore.YELLOW}Loading PDF document...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"{Fore.GREEN}✅ Successfully loaded {len(pages)} pages\n")

    hadith_chunks = []
    current_hadith_content = []
    current_hadith_metadata = {}
    
    # Regex to identify the start of a new hadith and capture metadata
    hadith_start_pattern = re.compile(r"Volume (\d+), Book (\d+), Number (\d+):")
    narrator_pattern = re.compile(r"Narrated (.+?):")

    print(f"{Fore.CYAN}Starting Hadith identification and chunking...\n")

    for page_idx, page in enumerate(pages, 1):
        lines = page.page_content.split('\n')
        
        # Set book_name directly to "Sahih Bukhari" as requested
        current_book_name = "Sahih Bukhari"


        for i, line in enumerate(lines):
            hadith_match = hadith_start_pattern.match(line.strip())
            
            if hadith_match:
                # If a new hadith is found, save the previous one (if any)
                if current_hadith_content:
                    hadith_chunks.append(
                        Document(
                            page_content="\n".join(current_hadith_content).strip(),
                            metadata=current_hadith_metadata
                        )
                    )
                    print(f"{Fore.GREEN}Chunked Hadith: Volume {current_hadith_metadata.get('volume', '')}, Book {current_hadith_metadata.get('book_number', '')}, Hadith {current_hadith_metadata.get('hadith_number', '')}, Narrator: {current_hadith_metadata.get('narrator', '')}, Book Name: {current_hadith_metadata.get('book_name', '')}")

                # Start new hadith
                current_hadith_metadata = {
                    "volume": int(hadith_match.group(1)),
                    "book_number": int(hadith_match.group(2)),
                    "hadith_number": int(hadith_match.group(3)),
                    "source_page": page_idx,  # Store the page number where the hadith started
                    "book_name": current_book_name # Add the extracted book name
                }
                current_hadith_content = [line.strip()] # Add the current line (Volume, Book, Number)

                # Try to find narrator in the next line
                if i + 1 < len(lines):
                    narrator_match = narrator_pattern.match(lines[i+1].strip())
                    if narrator_match:
                        current_hadith_metadata["narrator"] = narrator_match.group(1).strip()
                        current_hadith_content.append(lines[i+1].strip()) # Add narrator line
                        
            elif current_hadith_content: # Continue adding content to the current hadith
                current_hadith_content.append(line.strip())
                
    # Add the last hadith after the loop
    if current_hadith_content:
        hadith_chunks.append(
            Document(
                page_content="\n".join(current_hadith_content).strip(),
                metadata=current_hadith_metadata
            )
        )
        print(f"{Fore.GREEN}Chunked Hadith: Volume {current_hadith_metadata.get('volume', '')}, Book {current_hadith_metadata.get('book_number', '')}, Hadith {current_hadith_metadata.get('hadith_number', '')}, Narrator: {current_hadith_metadata.get('narrator', '')}, Book Name: {current_hadith_metadata.get('book_name', '')}")


    print(f"{Fore.MAGENTA}\n{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Processing Complete!{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Total Hadith Chunks Generated: {len(hadith_chunks)}{Style.RESET_ALL}")
    print(f"\n✨{Fore.MAGENTA} Hadiths chunked and metadata extracted!\n{Style.RESET_ALL}")

    return hadith_chunks

pdf_path = "data/sahih_bukhari.pdf" 


try:
    chunks = chunk_hadiths_with_metadata(pdf_path)

    if chunks:
        print(f"{Fore.BLUE}\n--- First Chunk ---")
        print(f"{Fore.BLUE}Content:\n{chunks[0].page_content}\n{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Metadata: {chunks[0].metadata}\n{Style.RESET_ALL}")
        
        if len(chunks) > 1:
            print(f"{Fore.WHITE}--- Second Chunk ---{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Content:\n{chunks[1].page_content}\n{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Metadata: {chunks[1].metadata}\n{Style.RESET_ALL}")

        if len(chunks) > 2:
            print(f"{Fore.LIGHTRED_EX}--- Third Chunk ---{Style.RESET_ALL}")
            print(f"{Fore.LIGHTRED_EX}Content:\n{chunks[2].page_content}\n{Style.RESET_ALL}")
            print(f"{Fore.LIGHTRED_EX}Metadata: {chunks[2].metadata}\n{Style.RESET_ALL}")

        print(f"{Fore.LIGHTBLUE_EX}--- Last Chunk ---")
        print(f"{Fore.LIGHTBLUE_EX}Content:\n{chunks[-1].page_content}\n{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLUE_EX}Metadata: {chunks[-1].metadata}\n{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}No hadith chunks were generated. Please check the PDF content and patterns.{Style.RESET_ALL}")

except Exception as e:
    print(f"\n{Fore.RED}An error occurred: {e}")
    print(f"{Fore.RED}Please ensure that path_pdf points to the correct PDF file and that you have the necessary libraries installed (e.g., `langchain-community`, `pypdf`).")