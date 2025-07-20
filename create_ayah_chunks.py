import pandas as pd
import logging
from colorama import Fore, Style, init

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.BLUE}%(asctime)s{Style.RESET_ALL} - {Fore.GREEN}%(levelname)s{Style.RESET_ALL} - %(message)s'
)
logger = logging.getLogger(__name__)

def create_ayah_chunks(file_path):
    """
    Creates ayah-based text chunks with overlap from a Quran CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        list: List of text chunks with metadata headers
    """
    try:
        logger.info(f"{Fore.CYAN}Reading CSV file: {file_path}{Style.RESET_ALL}")
        df = pd.read_csv(file_path)
        logger.info(f"{Fore.YELLOW}Found {len(df)} ayahs in the dataset{Style.RESET_ALL}")
        
        # Sort the data
        #df = df.sort_values(by=['surah_no', 'ayah_no_surah'])
        logger.info(f"{Fore.GREEN}Data sorted by surah and ayah numbers{Style.RESET_ALL}")

        logger.info(f"{Fore.MAGENTA}First 5 row of the dataframe: {df.head()}{Style.RESET_ALL}")
        
        chunks = []
        total_chunks = len(df)
        
        # Reset index for proper integer indexing
        df = df.reset_index(drop=True)
        
        for idx in range(total_chunks):
            row = df.iloc[idx]
            
            # Create metadata header for richer context
            header = (
                f"surah_no: {row['surah_no']}\n"
                f"surah_name_en: {row['surah_name_en']}\n"
                f"ayah_no_surah: {row['ayah_no_surah']}\n"
                f"ayah_ar: {row['ayah_ar']}\n"
                f"ayah_en: {row['ayah_en']}"
            )
            
            # Create content with current ayah
            current_content = f"{row['ayah_ar']} - {row['ayah_en']}"
            
            # Add previous ayah if exists and same surah
            if idx > 0:
                prev_row = df.iloc[idx-1]
                if prev_row['surah_no'] == row['surah_no']:
                    prev_content = f"{prev_row['ayah_ar']} - {prev_row['ayah_en']}"
                    content = f"{prev_content}\n{current_content}"
                    logger.debug(f"Chunk {idx+1}: Added ayah {prev_row['ayah_no_surah']} as overlap")
                else:
                    content = current_content
                    logger.debug(f"Chunk {idx+1}: New surah started ({row['surah_name_en']})")
            else:
                content = current_content
                logger.debug(f"First chunk created for surah {row['surah_name_en']}")
            
            # Combine header and content
            chunk = f"{header}\n\n{content}"
            chunks.append(chunk)
            
            # Log progress every 100 chunks
            if (idx + 1) % 100 == 0:
                logger.info(f"{Fore.MAGENTA}Created {idx+1}/{total_chunks} chunks ({round((idx+1)/total_chunks*100, 1)}%){Style.RESET_ALL}")
        
        logger.info(f"{Fore.GREEN}Successfully created {len(chunks)} chunks!{Style.RESET_ALL}")
        return chunks
    
    except FileNotFoundError:
        logger.error(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
        return []
    except pd.errors.EmptyDataError:
        logger.error(f"{Fore.RED}The CSV file is empty{Style.RESET_ALL}")
        return []
    except KeyError as e:
        logger.error(f"{Fore.RED}Missing required column: {e}{Style.RESET_ALL}")
        return []
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
        return []