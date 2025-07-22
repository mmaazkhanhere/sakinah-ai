from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
from colorama import Fore, Style
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def chunk_hadith_sahi_muslim_pdf(pdf_path):
    logging.info(Fore.BLUE + f"Loading PDF from path: {pdf_path}" + Style.RESET_ALL)
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logging.info(Fore.GREEN + "PDF loaded successfully." + Style.RESET_ALL)
    except Exception as e:
        logging.error(Fore.RED + f"Error loading PDF: {e}" + Style.RESET_ALL)
        return []

    processed_hadith_chunks = []
    current_hadith_text = []
    current_hadith_metadata = {}
    hadith_counter = 0

    logging.info(Fore.BLUE + "Starting to process pages..." + Style.RESET_ALL)

    hadith_start_pattern = re.compile(
        r'Book (\d+), Number (\d+):',
        re.IGNORECASE | re.DOTALL
    )
    
    # A more general pattern for chapter/book titles if needed for context
    section_title_pattern = re.compile(
        r'^(SAHIH MUSLIM BOOK \d+: THE BOOK|Chapter \d+: Pertaining to the verse:).+$',
        re.IGNORECASE | re.MULTILINE
    )

    # Regex to find narrator at the beginning of a hadith text
    # This pattern looks for common phrases followed by a name or names (capitalized words)
    # It tries to be flexible by matching various introductory phrases.
    narrator_pattern = re.compile(
        r'(?:It is narrated (?:on the authority of|from)|reported (?:on the authority of|from)|On the authority of)\s+([A-Z][a-zA-Z\s\'-]+(?: and [A-Z][a-zA-Z\s\'-]+)*)(?:,| that| said| narrated| reported| observed| that he)',
        re.IGNORECASE
    )
    
    current_section_title = "Introduction" # Default section

    for page_idx, page in enumerate(pages, 1):
        content = page.page_content
        page_num = page.metadata.get('page', page_idx - 1) + 1 # Adjust for 0-indexed pages
        logging.info(Fore.YELLOW + f"Processing page {page_num}" + Style.RESET_ALL)

        lines = content.split('\n')
        for line_idx, line in enumerate(lines):
            # Check for section titles first to update context
            section_match = section_title_pattern.match(line.strip())
            if section_match:
                current_section_title = line.strip()
                logging.info(Fore.CYAN + f"Found section title: {current_section_title}" + Style.RESET_ALL)
                continue # Don't treat section titles as hadith starts

            hadith_match = hadith_start_pattern.match(line.strip())

            if hadith_match:
                # If we found a new hadith, and we were collecting a previous one,
                # save the previous one first.
                if current_hadith_text:
                    hadith_content = "\n".join(current_hadith_text).strip()
                    if hadith_content: # Ensure content is not empty
                        # Extract narrator from the collected hadith text
                        narrator = "Unknown"
                        narrator_match = narrator_pattern.search(hadith_content)
                        if narrator_match:
                            narrator = narrator_match.group(1).strip()
                        
                        current_hadith_metadata['section_title'] = current_section_title
                        current_hadith_metadata['source_page'] = current_hadith_metadata.get('source_page', 'Unknown') # Ensure page is recorded
                        current_hadith_metadata['narrator'] = narrator
                        current_hadith_metadata['book_name'] = "Sahih Muslim" # Explicitly set book name

                        processed_hadith_chunks.append(
                            Document(page_content=hadith_content, metadata=current_hadith_metadata)
                        )
                        hadith_counter += 1
                        logging.info(Fore.GREEN + f"Processed Hadith {hadith_counter}: {current_hadith_metadata['full_source']}" + Style.RESET_ALL)

                # Start collecting the new hadith
                book_num = hadith_match.group(1)
                hadith_num = hadith_match.group(2)
                current_hadith_text = [line.strip()] # Start with the current line
                current_hadith_metadata = {
                    'book_number': book_num,
                    'hadith_number': hadith_num,
                    'source_page': page_num,
                    'full_source': f"Book {book_num}, Hadith {hadith_num} (Page {page_num})"
                }
                logging.info(Fore.CYAN + f"Started new Hadith: {current_hadith_metadata['full_source']}" + Style.RESET_ALL)
            else:
                if current_hadith_text:
                    current_hadith_text.append(line.strip())

    # After iterating through all pages, check if there's a pending hadith to save
    if current_hadith_text:
        hadith_content = "\n".join(current_hadith_text).strip()
        if hadith_content:
            narrator = "Unknown"
            narrator_match = narrator_pattern.search(hadith_content)
            if narrator_match:
                narrator = narrator_match.group(1).strip()

            current_hadith_metadata['section_title'] = current_section_title
            current_hadith_metadata['source_page'] = current_hadith_metadata.get('source_page', 'Unknown')
            current_hadith_metadata['narrator'] = narrator
            current_hadith_metadata['book_name'] = "Sahih Muslim"

            processed_hadith_chunks.append(
                Document(page_content=hadith_content, metadata=current_hadith_metadata)
            )
            hadith_counter += 1
            logging.info(Fore.GREEN + f"Processed final Hadith {hadith_counter}: {current_hadith_metadata['full_source']}" + Style.RESET_ALL)

    logging.info(Fore.BLUE + f"Total Hadith Chunks Generated: {len(processed_hadith_chunks)}" + Style.RESET_ALL)
    return processed_hadith_chunks

pdf_path = "data/sahih_muslim.pdf" 
chunks = chunk_hadith_sahi_muslim_pdf(pdf_path)

if chunks:
    print(f"Total Hadith Chunks Generated: {len(chunks)}\n")
    print("First Hadith Chunk:")
    print(f"Content: {chunks[0].page_content}")
    print(f"Metadata: {chunks[0].metadata}\n")
    if len(chunks) > 1:
        print("Second Hadith Chunk:")
        print(f"Content: {chunks[1].page_content}")
        print(f"Metadata: {chunks[1].metadata}\n")
    if len(chunks) > 2:
        print("Third Hadith Chunk:")
        print(f"Content: {chunks[2].page_content}")
        print(f"Metadata: {chunks[2].metadata}\n")
    if len(chunks) > 0:
        print("Last Hadith Chunk:")
        print(f"Content: {chunks[-1].page_content}")
        print(f"Metadata: {chunks[-1].metadata}\n")
else:
    print("No hadith chunks were generated. Please check the PDF path and content patterns.")