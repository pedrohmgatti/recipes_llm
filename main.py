from inference import run_inference
from ingestion import run_ingestion

def main():
    collection_name = "documents_collection"
    
    qdrant, dense_model = run_ingestion(collection_name=collection_name)
    
    while True:
        query = input("Enter your query: ")
        if query == "":
            break

        result = run_inference(query_raw=query, qdrant=qdrant, dense_model=dense_model, collection_name=collection_name)

        print(result)    

if __name__ == "__main__":
    main()