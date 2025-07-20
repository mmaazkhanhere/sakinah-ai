import os
import time
import uuid
import pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from typing import List, Optional, Dict
from colorama import Fore, Style
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def embed_and_store(
    chunks: List[str], 
    index_name: str, 
    namespace: Optional[str] = None, 
    batch_size: int = 100,
    embedding_model: str = "text-embedding-3-small",
    dimension: int = 1536,
    text_key: str = "text"
) -> Dict[str, int]:
    """
    Embeds text chunks using OpenAI and stores them in Pinecone
    
    Args:
        chunks: List of text chunks to embed and store
        index_name: Pinecone index name
        namespace: Pinecone namespace (optional)
        batch_size: Batch size for upsert operations
        embedding_model: OpenAI embedding model name
        dimension: Dimension of the embedding vectors
        text_key: Metadata key for storing text content
    
    Returns:
        Dictionary containing storage statistics
    """
    # Validate environment variables
    required_envs = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
    for env in required_envs:
        if env not in os.environ:
            raise EnvironmentError(f"{Fore.RED}Missing required environment variable: {env}{Style.RESET_ALL}")

    # Initialize Pinecone
    try:
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        print(f"{Fore.GREEN}✓ Pinecone initialized successfully")
    except Exception as e:
        print(f"{Fore.RED}❌ Pinecone initialization failed: {e}{Style.RESET_ALL}")
        raise

    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        print(f"{Fore.YELLOW}⚠ Index '{index_name}' not found. Creating new index...")
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",  # Choose your cloud provider
                    region="us-east-1"  # Choose your region
                )
            )
            print(f"{Fore.GREEN}✓ Created new index: {index_name} (dimension={dimension})")
            
            # Wait for index initialization
            print(f"{Fore.BLUE}⏳ Waiting for index readiness...")
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print(f"{Fore.GREEN}✓ Index is ready")
        except Exception as e:
            print(f"{Fore.RED}❌ Index creation failed: {e}{Style.RESET_ALL}")
            raise
    else:
        print(f"{Fore.BLUE}✓ Found existing index: {index_name}")

    # Initialize embeddings
    try:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
        )
        print(f"{Fore.GREEN}✓ OpenAI embeddings initialized: {embedding_model} ({dimension}D)")
    except Exception as e:
        print(f"{Fore.RED}❌ Embedding initialization failed: {e}{Style.RESET_ALL}")
        raise

    # Embed and store documents with progress tracking
    stats = {
        "total_chunks": len(chunks),
        "batches_processed": 0,
        "vectors_stored": 0,
        "start_time": time.time(),
        "last_update": time.time()
    }

    print(f"{Fore.CYAN}🚀 Starting embedding process for {stats['total_chunks']} chunks...")
    print(f"{Fore.CYAN}├── Batch size: {batch_size}")
    print(f"{Fore.CYAN}├── Estimated batches: {(stats['total_chunks'] + batch_size - 1) // batch_size}")
    
    try:
        # Get the Pinecone index
        index = pc.Index(index_name)
        
        # Process in batches with progress tracking
        for i in range(0, stats["total_chunks"], batch_size):
            batch_chunks = chunks[i:i+batch_size]
            
            # Embed the batch
            batch_vectors = embeddings.embed_documents(batch_chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for j, (text, vector) in enumerate(zip(batch_chunks, batch_vectors)):
                metadata = {text_key: text}
                vector_id = f"{index_name}-{uuid.uuid4().hex}"
                vectors.append((vector_id, vector, metadata))
            
            # Upsert to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)
            
            # Update stats
            stats["batches_processed"] += 1
            stats["vectors_stored"] += len(vectors)
            
            # Progress reporting every 5 seconds or last batch
            current_time = time.time()
            if current_time - stats["last_update"] > 5 or (i + batch_size) >= stats["total_chunks"]:
                elapsed = current_time - stats["start_time"]
                chunks_per_sec = stats["vectors_stored"] / elapsed if elapsed > 0 else 0
                remaining_chunks = stats["total_chunks"] - stats["vectors_stored"]
                eta = remaining_chunks / chunks_per_sec if chunks_per_sec > 0 else 0
                
                print(f"{Fore.CYAN}├── Processed: {stats['vectors_stored']}/{stats['total_chunks']} chunks "
                      f"({stats['vectors_stored']/stats['total_chunks']:.1%})")
                print(f"{Fore.CYAN}├── Speed: {chunks_per_sec:.1f} chunks/sec")
                print(f"{Fore.CYAN}├── Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                stats["last_update"] = current_time
        
        # Final stats
        total_time = time.time() - stats["start_time"]
        chunks_per_sec = stats["vectors_stored"] / total_time if total_time > 0 else 0
        
        print(f"{Fore.GREEN}✅ Successfully stored {stats['vectors_stored']} vectors in Pinecone!")
        print(f"{Style.BRIGHT}├── Index: {Fore.MAGENTA}{index_name}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}├── Total time: {Fore.MAGENTA}{total_time:.2f} seconds{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}├── Throughput: {Fore.MAGENTA}{chunks_per_sec:.1f} chunks/sec{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}└── Batches processed: {Fore.MAGENTA}{stats['batches_processed']}{Style.RESET_ALL}")
        
        if namespace:
            print(f"{Style.BRIGHT}Namespace: {Fore.MAGENTA}{namespace}")
        
        return {
            "total_vectors": stats["vectors_stored"],
            "total_time": total_time,
            "chunks_per_sec": chunks_per_sec,
            "batches_processed": stats["batches_processed"]
        }
        
    except Exception as e:
        print(f"{Fore.RED}❌ Vector storage failed at batch {stats['batches_processed']}: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠ Successfully stored {stats['vectors_stored']} vectors before failure")
        raise