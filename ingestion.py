# %%
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

from fastembed import TextEmbedding

import uuid
import os

# %%
DATA_PATH = "recipes"

# %%
def run_ingestion(collection_name="documents_collection"):
    # QDRANT SETUP
    collection_name = "documents_collection"
    qdrant = QdrantClient(":memory:")

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

    # CHUNKER SETUP
    chunker_tokenizer = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    chunker = HybridChunker(
        tokenizer=chunker_tokenizer,
        max_tokens=768,
        merge_peers=True
    )

    # EMBEDDING MODEL SETUP
    dense_model = TextEmbedding(model_name=chunker_tokenizer)

    # INGESTION PIPELINE
    DATA_PATH = "recipes"
    files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH)]

    converter = DocumentConverter()

    for f in files:
        dl_doc = converter.convert(f)

        chunks = chunker.chunk(dl_doc=dl_doc.document)

        points=[]
        for c in chunks:
            dense_embedding = list(dense_model.passage_embed(c.text))[0].tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=dense_embedding,
                payload={
                    "text": c.text
                }
            )
            points.append(point)

        qdrant.upload_points(
            collection_name=collection_name,
            points=points
        )

        return qdrant, dense_model