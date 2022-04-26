import json
from flask_cors import CORS
from pymongo import MongoClient
from flask import Flask, Response, request

from inference import *

app = Flask(__name__)
CORS(app)

try:
    client = MongoClient(db_url, connect=False)
    db = client[database]
    client.server_info()
    print("Successfully Data Base Accessed !")

except Exception as e:
    print("#############################################################")
    print(e)
    print("#############################################################")

SMapp = IntegratedInference()
SMapp.load_inference_models_objects()

@app.route("/suicide", methods=["GET", "POST"])
def suicide():
    try:
        message = request.get_json(force=True)
        text = message["text"]
        user_name = message["user_name"]
        suicidal_response = SMapp.SuicidalInference(text, user_name)

        _ = db[suicidal_collection].insert_one(suicidal_response)

        return Response(
                        response=json.dumps({
                                    "status": "success",
                                    "suicidal_response" : "{}".format(suicidal_response)
                                    }), 
                        status=200, 
                        mimetype="application/json"
                        )

    except Exception as error:
        return Response(
                        response=json.dumps({
                                    "status": "failed",
                                    "error" : error
                                    }), 
                        status=400, 
                        mimetype="application/json"
                        )

@app.route("/bot", methods=["GET", "POST"])
def bot():
    try:
        message = request.get_json(force=True)
        chat = message["chat"]
        user_name = message["user_name"]
        bot_response = SMapp.BotInference(chat, user_name)

        _ = db[chat_bot_collection].insert_one(bot_response)

        return Response(
                        response=json.dumps({
                                        "status": "success", 
                                        "bot_response" : "{}".format(bot_response)
                                           }), 
                        status=200, 
                        mimetype="application/json"
                        )

    except Exception as error:
        return Response(
                        response=json.dumps({
                                    "status": "failed",
                                    "error" : error
                                    }), 
                        status=400, 
                        mimetype="application/json"
                        )

if __name__ =='__main__':
    app.run(
        debug=True, 
        host=aws_url, 
        port=aws_port
        )