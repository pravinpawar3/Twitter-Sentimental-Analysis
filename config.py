# config.py
import urllib.parse
MONGO_CONNECTION_URL = "cluster0.cyoe45m.mongodb.net"
DATABASE_NAME = "test"
USERNAME = "prav3"
PASSWORD = "Zfo5Q7wu16FsRkn7"
MONGO_CONNECTION_URL = f"mongodb+srv://{urllib.parse.quote_plus(USERNAME)}:{urllib.parse.quote_plus(PASSWORD)}@{MONGO_CONNECTION_URL}/{DATABASE_NAME}?retryWrites=true&w=majority"
