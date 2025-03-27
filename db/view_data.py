from pymilvus import Collection, utility, connections

connections.connect(alias="default", host="localhost", port="19530")

# Check connection status
print(connections.has_connection("default"))  # Should return True

def create_index_and_load(collection_name):
    # Get the collection
    collection = Collection(collection_name)
    
    # Check if index exists on the vector field
    index_param = {
        "metric_type": "L2",  # or "IP" for inner product
        "index_type": "IVF_FLAT",  # or "HNSW", "IVF_SQ8", etc.
        "params": {"nlist": 1024}  # adjust based on your data size
    }
    
    try:
        # Create index on the vector field if not exists
        collection.create_index(
            field_name="face_vector", 
            index_params=index_param
        )
        print(f"Index created on {collection_name}")
    except Exception as e:
        print(f"Index might already exist or error occurred: {e}")
    
    # Load the collection
    collection.load()
    print(f"Collection {collection_name} loaded")
    
    return collection

# Method 1: Basic query to retrieve all rows
def view_all_rows(collection_name):
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist.")
        return None
    
    collection = create_index_and_load(collection_name)
    
    # Define search parameters to retrieve all entities
    search_param = {
        "expr": "", # empty expression means no filtering
        "output_fields": ["id", "name", "age", "gender"],
        "limit" : 20,
    }
    
    # Perform the query
    results = collection.query(**search_param)
    
    # Print the results
    for row in results:
        print(row)
    
    print(f"Total number of rows: {len(results)}")

# Usage examples
view_all_rows("face_data")
# view_rows_with_limit("face_data", 10)
# count_total_entities("face_data")