mkdir db
python3 ./setup/create_local_database.py
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt