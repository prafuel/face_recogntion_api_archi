from pymilvus import ( 
    Collection, CollectionSchema, FieldSchema, DataType, utility
)

def get_schema():
    # Check if collection exists
    if utility.has_collection("face_data"):
        # Load existing collection
        collection = Collection("face_data")

        # Print out current schema details
        print("Existing Collection Schema:")
        print("Fields:", collection.schema.fields)
        print("Primary Field:", collection.schema.primary_field)

        return collection
    
    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="age", dtype=DataType.INT64),
        FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="face_vector", dtype=DataType.FLOAT_VECTOR, dim=2048)  # Adjust dimension accordingly
    ]

    # Create schema
    schema = CollectionSchema(fields, description="Face Embeddings Storage")

    # Create collection
    collection = Collection(name="face_data", schema=schema)

    return collection