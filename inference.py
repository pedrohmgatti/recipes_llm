# %%
import os
import dotenv

dotenv.load_dotenv(".env")

from groq import Groq

# %%
def run_inference(query_raw, qdrant, dense_model, collection_name="documents_collection"):
    # QUERY EMBEDDING
    query_embedding = list(dense_model.passage_embed(query_raw))[0].tolist()

    # QDRANT SEARCH
    result = qdrant.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=3,
    )

    documents = "\n".join(
    [f"Documento {i+1}: {hit.payload['text']}" for i, hit in enumerate(result)]
    )

    # PROMPT
    prompt = f"""
    Com base nas seguintes receitas, responda a pergunta de forma simples.

    Documentos: {documents}

    Pergunta: {query_raw}
    """

    llm_client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )
    chat_completion = llm_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content