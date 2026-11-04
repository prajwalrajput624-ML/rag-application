import os
import nest_asyncio
import logging
import joblib

from dotenv import load_dotenv
from qdrant_client import QdrantClient,AsyncQdrantClient

from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings,StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.ingestion import IngestionCache,IngestionPipeline
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.llms.groq import Groq 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



nest_asyncio.apply()
load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

INGESTION_CACHE = "RAG_V1"
BATCH_SIZE = 32

try:
    Settings.llm = Groq(
        model = "openai/gpt-oss-120b",
        is_chat_model = True
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name = "BAAI/bge-base-en-v1.5"
    )
    print("✅ LLM And Embedding Model Loaded!")
except Exception as e:
    print(f"Models Loding Error: {e}")


try:
    client = QdrantClient(
        url = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
        timeout = 60
    )

    aclient = AsyncQdrantClient(
        url = os.getenv("QDRANT_URL"),
        api_key = os.getenv("QDRANT_API_KEY"),
        timeout = 60
    )

    vector_store = QdrantVectorStore(
        collection_name = "History_Rag_v1",
        client = client,
        aclient = aclient,
        batch_size = BATCH_SIZE
    )
    storage_context = StorageContext.from_defaults(
        vector_store = vector_store
    )
    print("✅ Qdrant Vector Database Is Ready!")
except Exception as e:
    print(f"Qdrant Connection Error: {e}")

try:
    redis_store = RedisKVStore(redis_uri = os.getenv("REDIS_URL"))
    cache = IngestionCache(
        collection = f"Rag:{INGESTION_CACHE}",
        cache = redis_store
    )
    print("✅ Cacheing Apply Complete!")
except Exception as e:
    print(f"Cacheing Error: {e}")

try:
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(
                chunk_size = 512,
                chunk_overlap = 50,
                include_metadata = True,
                include_prev_next_rel = True
            ),
            Settings.embed_model
        ],
        cache = cache,
        vector_store = vector_store
    )
    print("✅ Ingestion Pipeline Is Ready!")
except Exception as e:
    print(f"Ingestion Pipeline Error: {e}")

try:
    documents = SimpleDirectoryReader(input_dir = "data").load_data()
    print(f"Total Documents: {len(documents)}")
    print("✅ Documents Reading Complete!")
except Exception as e:
    print(f"Documents Reading Error: {e}")

try:
    nodes = pipeline.run(documents = documents)
    print(f"Total Nodes: {len(nodes)}")
    joblib.dump(nodes, "app/nodes.pkl")
    print("✅ Chunking Is Complete!")
except Exception as e:
    print(f"Chuking Error: {e}")

try:
    index = VectorStoreIndex.from_vector_store(
        vector_store = vector_store,
        storage_context = storage_context
    )
    print("✅ Indexing Is Complete!")
except Exception as e:
    print(f"Indexing Error: {e}")