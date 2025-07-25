import pandas as pd
import logging
from colorama import Fore, Style, init

# Initialize colorama for colored logging
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)
logger = logging.getLogger(__name__)

def create_ayah_chunks(file_path, prev_overlap=3, next_overlap=3):
    """
    Creates ayah-based text chunks with specified previous and next ayah overlap,
    and separates metadata from the content to be embedded.

    Args:
        file_path (str): Path to the CSV file containing Quranic verses.
        prev_overlap (int): Number of previous ayahs to include for context.
        next_overlap (int): Number of next ayahs to include for context.

    Returns:
        list: List of dictionaries, each containing 'text_content' for embedding
              and 'metadata' for storage.
    """
    try:
        logger.info(f"{Fore.CYAN}Reading CSV file: {file_path}{Style.RESET_ALL}")
        df = pd.read_csv(file_path)
        logger.info(f"{Fore.YELLOW}Found {len(df)} ayahs in the dataset{Style.RESET_ALL}")

        # Ensure the DataFrame is sorted for correct sequential processing
        df = df.sort_values(by=['surah_no', 'ayah_no_surah']).reset_index(drop=True)
        logger.info(f"{Fore.GREEN}Data sorted by surah and ayah numbers{Style.RESET_ALL}")
        logger.debug(f"{Fore.MAGENTA}First 5 rows of the dataframe:\n{df.head()}{Style.RESET_ALL}")

        chunks = []
        total_ayahs = len(df)

        for idx in range(total_ayahs):
            current_ayah_row = df.iloc[idx]

            # --- 1. Prepare Metadata for the CURRENT Ayah ---
            # This metadata describes the primary ayah this chunk is built around.
            metadata = {
                "surah_no": int(current_ayah_row['surah_no']),
                "surah_name_en": str(current_ayah_row['surah_name_en']),
                "ayah_no_surah": int(current_ayah_row['ayah_no_surah']),
                "ayah_ar": str(current_ayah_row['ayah_ar']),
                "ayah_en": str(current_ayah_row['ayah_en']),
            }

            # --- 2. Construct Text Content for Embedding with Overlap ---
            context_ayats = []

            # Helper function to format an ayah string for text_content
            def format_ayah_for_embedding(row_data):
                return (
                    f"Surah {row_data['surah_no']}:{row_data['ayah_no_surah']}: "
                    f"{row_data['ayah_ar']} - {row_data['ayah_en']}"
                )

            # Add previous ayahs for context (within the same Surah)
            for i in range(prev_overlap, 0, -1): # Iterate backwards from prev_overlap down to 1
                prev_idx = idx - i
                if prev_idx >= 0 and df.iloc[prev_idx]['surah_no'] == current_ayah_row['surah_no']:
                    context_ayats.append(format_ayah_for_embedding(df.iloc[prev_idx]))
                else:
                    break # Stop if we hit beginning of DataFrame or a new Surah

            # Add the current ayah
            context_ayats.append(format_ayah_for_embedding(current_ayah_row))

            # Add next ayahs for context (within the same Surah)
            for i in range(1, next_overlap + 1):
                next_idx = idx + i
                if next_idx < total_ayahs and df.iloc[next_idx]['surah_no'] == current_ayah_row['surah_no']:
                    context_ayats.append(format_ayah_for_embedding(df.iloc[next_idx]))
                else:
                    break # Stop if we hit end of DataFrame or a new Surah

            # Combine all collected ayahs into the single text content for embedding
            text_content_for_embedding = "\n".join(context_ayats)
            
            # --- 3. Store the Chunk ---
            chunks.append({
                "text_content": text_content_for_embedding,
                "metadata": metadata
            })

            # Log progress
            if (idx + 1) % 100 == 0 or (idx + 1) == total_ayahs:
                logger.info(f"{Fore.MAGENTA}Created {idx+1}/{total_ayahs} chunks ({round((idx+1)/total_ayahs*100, 1)}%){Style.RESET_ALL}")

        logger.info(f"{Fore.GREEN}Successfully created {len(chunks)} chunks!{Style.RESET_ALL}")
        return chunks

    except FileNotFoundError:
        logger.error(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
        return []
    except pd.errors.EmptyDataError:
        logger.error(f"{Fore.RED}The CSV file is empty{Style.RESET_ALL}")
        return []
    except KeyError as e:
        logger.error(f"{Fore.RED}Missing required column in CSV: {e}{Style.RESET_ALL}")
        return []
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error during chunking: {e}{Style.RESET_ALL}")
        return []

# Example Usage:
chunks = create_ayah_chunks('data/quran.csv')
print(chunks[0])
print("-----")
print("-----")
print("-----")
print(chunks[-2])