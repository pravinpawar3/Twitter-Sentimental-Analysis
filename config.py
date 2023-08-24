# config.py
import urllib.parse
MONGO_CONNECTION_URL = "clustername"
DATABASE_NAME = "test"
USERNAME = "username"
PASSWORD = "password"
MONGO_CONNECTION_URL = f"mongodb+srv://{urllib.parse.quote_plus(USERNAME)}:{urllib.parse.quote_plus(PASSWORD)}@{MONGO_CONNECTION_URL}/{DATABASE_NAME}?retryWrites=true&w=majority"
