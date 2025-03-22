

To Run Whole Project
here is steps of command, have to follow

clone dir
```git clone <url>```

setup
```bash ./setup/setup.sh```

this command will ensure creating neccessory folders, creating enviroments, activate it and install all required lib

You have to do some changes in library imports in virtual env due to different version issue

changes to do
```
path = "venv/lib/python3.10/site-packages/keras_vggface/models.py"

from this : from keras.utils import layer_utils
to this : from tensorflow.python.keras.utils import layer_utils

from this : from keras.engine.topology import get_source_inputs
to this : from keras.utils.layer_utils import get_source_inputs

```

Run all apis
``` python3 api/recognition/recognition_api.py --reload ```
``` python3 api/age_prediction/age_prediction_api.py --reload ```
``` python3 api/drowsiness/drowsiness_api.py --reload ```
``` python3 api/emotion/emotion_api.py --reload ```
``` python3 api/gender_prediction/gender_prediction_api.py --reload ```
``` python3 api/liveliness/liveliness_api.py --reload ```

then Run main api
``` python3 main_api.py --reload ```


To put data in local database
run following gradio app
``` python3 setup/put_data_in_db_gradio.py ```

Update config file path : ``` config/config.py ```
here are option to select which option you want to activate
# face_recogntion_api_archi
