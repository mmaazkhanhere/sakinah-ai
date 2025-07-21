from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import re
from colorama import init, Fore, Style

# Initialize colorama for colored console output

def chunk_hadith_sahi_muslim_pdf(pdf_path):
    """
    Load a PDF document, identify individual hadith entries, and chunk them
    along with their associated metadata (Book and Number).

    This function assumes a specific pattern for hadith entries,
    e.g., "Book XXX, Number YYYY:" at the beginning of a hadith.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: List of document chunks, where each chunk represents
                        a hadith with its text and extracted metadata.
    """
    print(Fore.YELLOW + f"\n{'='*50}")
    print(Fore.CYAN + Style.BRIGHT + "🚀 Starting Hadith PDF Processing Pipeline")
    print(Fore.YELLOW + f"{'='*50}")
    print(Fore.LIGHTBLUE_EX + f"📄 Source PDF: {pdf_path}")
    print(Fore.YELLOW + "❗ Note: This script assumes hadith are marked by 'Book XXX, Number YYYY:' pattern.")
    print(Fore.YELLOW + "❗ Adjust `hadith_start_pattern` if your PDF's format differs.\n")

    print(Fore.MAGENTA + "⏳ Loading PDF document...")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(Fore.GREEN + f"✅ Successfully loaded {len(pages)} pages\n")
    except Exception as e:
        print(Fore.RED + f"❌ Error loading PDF: {e}")
        return []

    processed_hadith_chunks = []
    current_hadith_text = []
    current_hadith_metadata = {}
    hadith_counter = 0


    hadith_start_pattern = re.compile(
        r'Book (\d+), Number (\d+):',
        re.IGNORECASE
    )
    # A more general pattern for chapter/book titles if needed for context
    section_title_pattern = re.compile(
        r'^(SAHIH MUSLIM BOOK \d+: THE BOOK|Chapter \d+: Pertaining to the verse:).+$',
        re.IGNORECASE | re.MULTILINE
    )
    current_section_title = "Introduction" # Default section

    print(Fore.MAGENTA + "🔍 Starting hadith extraction and chunking...\n")

    for page_idx, page in enumerate(pages, 1):
        content = page.page_content
        page_num = page.metadata.get('page', page_idx - 1) + 1 # Adjust for 0-indexed pages

        print(Fore.LIGHTBLUE_EX + f"\n📖 Processing Page {page_num}/{len(pages)}...")

        lines = content.split('\n')
        for line_idx, line in enumerate(lines):
            # Check for section titles first to update context
            section_match = section_title_pattern.match(line.strip())
            if section_match:
                current_section_title = line.strip()
                print(Fore.CYAN + f"  🆕 Section Title Detected: '{current_section_title}'")
                continue # Don't treat section titles as hadith starts

            hadith_match = hadith_start_pattern.match(line.strip())

            if hadith_match:
                # If we found a new hadith, and we were collecting a previous one,
                # save the previous one first.
                if current_hadith_text:
                    hadith_content = "\n".join(current_hadith_text).strip()
                    if hadith_content: # Ensure content is not empty
                        # Add current section title to hadith metadata
                        current_hadith_metadata['section_title'] = current_section_title
                        current_hadith_metadata['source_page'] = current_hadith_metadata.get('source_page', 'Unknown') # Ensure page is recorded
                        processed_hadith_chunks.append(
                            Document(page_content=hadith_content, metadata=current_hadith_metadata)
                        )
                        hadith_counter += 1
                        print(Fore.GREEN + f"    ✅ Hadith {current_hadith_metadata.get('book_number', '')}-{current_hadith_metadata.get('hadith_number', '')} extracted (Length: {len(hadith_content)})")

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
            else:

                if current_hadith_text:
                    current_hadith_text.append(line.strip())

    # After iterating through all pages, check if there's a pending hadith to save
    if current_hadith_text:
        hadith_content = "\n".join(current_hadith_text).strip()
        if hadith_content:
            current_hadith_metadata['section_title'] = current_section_title
            current_hadith_metadata['source_page'] = current_hadith_metadata.get('source_page', 'Unknown')
            processed_hadith_chunks.append(
                Document(page_content=hadith_content, metadata=current_hadith_metadata)
            )
            hadith_counter += 1
            print(Fore.GREEN + f"    ✅ Final Hadith {current_hadith_metadata.get('book_number', '')}-{current_hadith_metadata.get('hadith_number', '')} extracted (Length: {len(hadith_content)})")


    # Print final summary
    print(Fore.YELLOW + f"\n{'='*50}")
    print(Fore.CYAN + Style.BRIGHT + "🏁 Hadith Extraction Complete!")
    print(Fore.YELLOW + f"{'='*50}")
    print(Fore.GREEN + f"📊 Total Pages Processed: {len(pages)}")
    print(Fore.GREEN + f"📊 Total Hadith Chunks Generated: {hadith_counter}")
    print(Fore.LIGHTBLUE_EX + "\n✨ Hadith are ready for further processing!\n")

    return processed_hadith_chunks

pdf_path = "data/sahih_muslim.pdf" # Assuming the uploaded PDF is named this way
chunks = chunk_hadith_sahi_muslim_pdf(pdf_path)

if chunks:
    print(Fore.GREEN + f"\nFirst Hadith Chunk:\n{chunks[0].page_content}\nMetadata: {chunks[0].metadata}\n")
    if len(chunks) > 1:
        print(Fore.YELLOW + f"\nSecond Hadith Chunk:\n{chunks[1].page_content}\nMetadata: {chunks[1].metadata}\n")
    if len(chunks) > 2:
        print(Fore.MAGENTA + f"\nThird Hadith Chunk:\n{chunks[2].page_content}\nMetadata: {chunks[2].metadata}\n")
    if len(chunks) > 0:
        print(Fore.CYAN + f"\nLast Hadith Chunk:\n{chunks[-1].page_content}\nMetadata: {chunks[-1].metadata}\n")
else:
    print(Fore.RED + "No hadith chunks were generated. Please check the PDF path and content patterns.")

