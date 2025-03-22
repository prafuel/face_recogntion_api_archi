import os
import numpy as np

local_db_path = os.path.join("db", "detailed_data.npy")

if not os.path.exists(local_db_path):
    print("Database is not exists...")
    print("Creating new database")
    np.save(local_db_path, {"face_vectors": np.array(), "labels" : np.array(), "age" : np.array(), "gender" : np.array()})
    print("Database created...")
