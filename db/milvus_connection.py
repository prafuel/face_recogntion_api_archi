import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from create_schema import get_schema

# Connect to Milvus running on localhost (default port: 19530)
connections.connect(alias="default", host="localhost", port="19530")

# Check connection status
print(connections.has_connection("default"))  # Should return True


def npy_data_in_milvus(npy_file_path: str, collection: Collection):
    # Load the NumPy data
    npy_database = np.load(npy_file_path, allow_pickle=True).item()
    
    # Extract data from the NumPy file
    face_vectors = npy_database['face_vectors'].tolist()
    labels = npy_database['labels'].tolist()
    user_age_data = [int(age) for age in npy_database['age'].tolist()]
    user_gender_data = npy_database['gender'].tolist()

    print("user_age_data: ", type(user_age_data[0]))
    
    # Prepare data for insertion
    # Note: We're adding a name field (which seems to be missing in your current data)
    # You might want to generate names or use labels as names
    insert_data = [
        labels,  # name field
        user_age_data,  # age field
        user_gender_data,  # gender field
        face_vectors  # face_vector field
    ]
    
    # Insert data into the collection
    collection.insert(insert_data)
    collection.flush()


if __name__ == "__main__":
    # utility.drop_collection("face_data")
    collection = get_schema()
    npy_data_in_milvus("./detailed_data.npy", collection)
