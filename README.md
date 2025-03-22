# Face Recognition API - Setup Guide

## Prerequisites
1. **Git** - To clone the repository.
2. **Python 3.10+** - Ensure Python is installed and accessible from the terminal.
3. **Virtual Environment** - Recommended for dependency management.

## Step 1: Clone the Repository
```
git clone https://github.com/prafuel/face_recogntion_api_archi.git
cd face_recogntion_api_archi
```

## Step 2: Setup the Environment
Run the setup script to create necessary folders, set up a virtual environment, activate it, and install all required libraries:
```
bash ./setup/setup.sh
```

### Important Note: Library Import Fix
Due to version differences, update the following lines in your virtual environment:

**File:** `venv/lib/python3.10/site-packages/keras_vggface/models.py`

Replace:
```python
from keras.utils import layer_utils
```
With:
```python
from tensorflow.python.keras.utils import layer_utils
```

Also, replace:
```python
from keras.engine.topology import get_source_inputs
```
With:
```python
from keras.utils.layer_utils import get_source_inputs
```

## Step 3: Running the APIs
Start each API separately as follows:
```
python3 api/recognition/recognition_api.py --reload
python3 api/age_prediction/age_prediction_api.py --reload
python3 api/drowsiness/drowsiness_api.py --reload
python3 api/emotion/emotion_api.py --reload
python3 api/gender_prediction/gender_prediction_api.py --reload
python3 api/liveliness/liveliness_api.py --reload
```

## Step 4: Run the Main API
Once all sub-APIs are running, start the main API:
```
python3 main_api.py --reload
```

## Step 5: Populate the Local Database
To insert data into the local database, run the Gradio app:
```
python3 setup/put_data_in_db_gradio.py
```

## Configuration
Update the configuration file to select active modules and settings:
**File:** `config/config.py`

Modify the desired options and save the file.

## Troubleshooting
- Ensure that the virtual environment is active before running any API.
- Check for version conflicts and resolve them by updating the environment as needed.
- Review logs and error messages for hints on configuration issues.

## Support
For any issues or questions, feel free to reach out through the project repository or contact the maintainers.

