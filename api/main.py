# mongo.py
import os
from flask import Flask
from flask import jsonify
from flask import request
from dotenv import load_dotenv
from flask_mongoengine import MongoEngine

load_dotenv()

mongo_uri = os.environ.get("MONGO_URI")

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
    'host': mongo_uri
}

db = MongoEngine()
db.init_app(app)

class User(db.Document):
    name = db.StringField()
    email = db.StringField()


@app.route('/', methods=['GET'])
def get_test():
    return jsonify("this is a test")


if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0")