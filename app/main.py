from flask import Flask, request, jsonify, abort
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from datetime import datetime
import re
import pandas as pd
import numpy as np
import random
import gspread
import requests
import json
import os
import tensorflow as tf
from pythainlp.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import logging
import pickle

Channel_access_token = "aq1Ap78l8iwsg8gfqKBGrhg9HRroQs07TeJv2KEBQnlUcgotB2xd/WLx2xUMHmwMXiXC6rx+1zMY7DrbW1+jFYTKz+uoVM2J+UV9cOQvdLt4vKSSkH8f/GVCpZWXCXKPMZ+6EAGWL2hVctOZl12VSwdB04t89/1O/w1cDnyilFU="


# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
# Function to tokenize and preprocess text

credentials = {
    "type": "service_account",
    "project_id": "dynamic-radar-401417",
    "private_key_id": "398cae454d74e4d1ca8de3e259a21a159e7c31ee",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDNOZiPA+ecsns3\n6pFwQFfil3mLGBBYsSeWOcwK9t1BUDEhWWjh6rbvPXcH7FTQNomkdn4E4xydHzX1\n8pPoeu/29K0D4M+neoBU648xfxCQxiOAx9z7AzrCyVsW4tLliH1LJil8eL/XthSy\nZe9lJKhgfIUCAKAoJB8bAi/j2ny8xM4/6i9oq1lgqs2IsgYoFh72FbcFo3onmduL\n/LfsZy3ydQIL6FVhSX5XE1F3ez9+6iORTw5KzYdhL9WZ7cZoPTEZhS5AQ26Oe2KG\npLqfoMSGPonR+sf6h30tsytfLMB9LCwJK8ZeIPFJMYb/qywR9z1rLScFC56wOQWn\njwe/CONjAgMBAAECggEAV+kPrlesW+XivYGoOm79lq1vXcN5oRyGCiaI/rtf7O32\nlcQQlxHtug7Y0daNQEHUdVRiyCInbDDl8wyuCjy7VUzbXllramEh3v6m8Ltu861E\noRI8WXQ5NB5//A4+7B8rGMlopQ6uky9Gr2LWCTwKOoasjT6KXJeYkX++1vqNS2i4\nMqH+JFFIA6GrnSlZKCiogJ3l0itrgQKcis5fsOYuQ3G9JIZLA+Wj4QNriGQf+Lnk\nwS8sVlbBw/0M+v/bFhvyivNZxfn9GKgxJ/FrAznadVlifkM6suhaxqh5N12O6Rbp\nB8gKqEetTXmHfTD5lGPTfladoYAjtnCnXmSHqznYiQKBgQDuKu27j1Md/TDmyxCc\n7dlh5unci0Ee8ZoF6z3VbQAbrX+nZM/mTYo3ptzuC9NSM7mRXpwcVw8mq8kbHByZ\nFgD0KD4ZStBjKhNdKkWQeohATZDyOMp3WTToRgBEoMEMF5TVsaVlNUwWWaF4d0fU\nFpy+DMzOaVVaz+HJ0+9C32M3OQKBgQDclz0y7Iy/13jCXMK0tGjgjoSXF9tMh7VY\nmozTk52qM/9IqJ+acL/nHuLUzpFEhagrowf0wGO9Q24ddpUtpltmjGoNBuMOW+17\ni5yTju02r5lvrdF5VyBB2Y7DozBqXjvcF4cWOMqEprQ+20Gu6xYR+gIJ3YVQFfmT\nNCdB5EUzewKBgQCJAz08d1lzmSK2wv9NqC80eSj0bVALQyY+XXR3AXvccFVNsVtA\nvD7VgTL75uWtFoCctnbMfvECaDULWYLCNrxZuYTv+/Ah/CSjphva4ALeK0FFd5JS\nUolzTkH9ORWVpUNEJCxoKt8YxHt11/kEc/W8B5US8dneolTcTQYJzIVBUQKBgB3q\nhOo7GnuKBV6WpfFL5k1OBr1XBu7CGN4DV8X55xAGLUD1XW/ciqoVjj2+JgVc+wmv\n+ow+60fntS9ZvbGLNioaMOfDX1e7L+HpdTqtz43zEqQKtrX2EvHbR3lQ0Ggcj5Gx\nvyhMW2rSEO/VXHsUdtHJTi14VXQfBtHn6MsO4jOxAoGBAONx1EtyQvoGv6c0X2na\nAR9jwQPFGcGi/K3X9jUcdk0yaUVesrgjB/7+Gwt7WFSBIuzyrfOHe7iI+OCt7C6t\n3RTPztbcf282izYUFnxwKtojnMiFmuXABYLcX/o3hwtFSSipM9Nuso45Y93jUdPe\nA1Lr3f1bCUOxJnKyUsNbaLVR\n-----END PRIVATE KEY-----\n",
    "client_email": "test-chatbot@dynamic-radar-401417.iam.gserviceaccount.com",
    "client_id": "111123689561375064130",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test-chatbot%40dynamic-radar-401417.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# question
gc = gspread.service_account_from_dict(credentials)
sht2 = gc.open_by_url(
    'https://docs.google.com/spreadsheets/d/1GLkhEngVAxDoyKVa0aSMIM5YS4cPGWv7LTMmWbhNi-g/edit?usp=sharing')
# Select a specific sheet by name
worksheet = sht2.worksheet('Resource')
df = pd.DataFrame(worksheet.get_all_records())

# answer
answer = gc.open_by_url(
    'https://docs.google.com/spreadsheets/d/1GLkhEngVAxDoyKVa0aSMIM5YS4cPGWv7LTMmWbhNi-g/edit?usp=sharing')
worksheet_2 = answer.worksheet('Resource-ans')
Answer_sheet = pd.DataFrame(worksheet_2.get_all_records())



# def load_keras_model(model_path):
#     global loaded_model
#     loaded_model = load_model(model_path)


model_path = 'app/static/model_train-chat.h5'
loaded_model = load_model(model_path)

with open('app/static/model_tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
# load_keras_model(model_path)


def clean_text(text):
    # Remove emojis
    if isinstance(text, str):

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # alchemical symbols
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U0001FB00-\U0001FBFF"  # Symbols for Legacy Computing
                                   u"\U0001F004-\U0001F0CF"  # Miscellaneous Symbols and Pictographs
                                   u"\U0001F170-\U0001F19A"  # Enclosed Alphanumeric Supplement
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Strip leading and trailing whitespace and convert to lowercase
        text = text.strip().lower()

    return text


# Maximum sequence length (you should set this to match the model's input size)
maxlen = 17  # Adjust this based on your model's requirements


# Define the webhook endpoint
@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    if request.method == 'POST':
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        print(payload)
        logging.debug(Reply_token)

        
        # Extract the user's text message
        message = payload['events'][0]['message']['text']
        logging.debug("hi " + message)

        message = clean_text(message)

        # Create an empty list to store multiple reply messages
        # reply_messages = []

        # Add the first reply message
        if "คาเฟ่แนวธรรมชาติ" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][0]
            # reply_messages.append({"type": "text", "text": "CAFE"})
            # reply_messages.append({"type": "text", "text": "NATURE"})
        elif "คาเฟ่สายอาร์ต" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][1]

        elif "คาเฟ่สัตว์เลี้ยง" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][2]
        
        elif "คาเฟ่แนวโฮมมี่" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][3]
            # reply_messages.append({"type": "text", "text": "CAFE"})
            # reply_messages.append({"type": "text", "text": Answer_sheet["answer"][3]})
        elif "คาเฟ่แฟนตาซี" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][4]
        
        elif "คาเฟ่วินเทจ" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][5]
        
        elif "คาเฟ่มินิมอล" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][6]

        elif "คาเฟ่หมา" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][7]

        elif "คาเฟ่แมว" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][8]
        elif "คาเฟ่อ่านหนังสือ" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][9]
        elif "คาเฟ่การ์ตูน" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][10]
        elif "คาเฟ่ญี่ปุ่น" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][11]
        elif "คาเฟ่เกาหลี" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][12]
          
        elif "คาเฟ่ดอกไม้" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][13]
        
        elif "คาเฟ่น่ารัก" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][14]
        
        elif "คาเฟ่บอร์ดเกม" in message:
            Reply_message = "CAFE\n" + Answer_sheet["answer"][15] 

            # reply_messages.append({"type": "text", "text": "CAFE"})
            # reply_messages.append({"type": "text", "text": Answer_sheet['answer'][1]})
        elif "สวัสดี" in message:
            Reply_message = "สวัสดีค่ะ"

            # reply_messages.append({"type": "text", "text": "สวัสดีค่ะ"})
        elif "type" in message:
            flexType(Reply_token, Channel_access_token )
            return request.json, 200


            # reply_messages.append({"type": "text", "text": "Cafe: \n -แนวธรรมชาติ \n -แนวอาร์ต\n"})

        elif "developer" in message:
            Reply_message = "cream pleang boom >0<"

            # reply_messages.append({"type": "text", "text": "cream pleang boom >0<"})
        else:
            confidence_threshold = 0.66

            dt = datetime.now()
            data_df = pd.DataFrame.from_records(df)

            new_sentence_tokenized = [word_tokenize(message)]
            new_sentence_sequences = tokenizer.texts_to_sequences(new_sentence_tokenized)
            new_sentence_padded = pad_sequences(new_sentence_sequences, maxlen=17, padding="post")
            logit = loaded_model.predict(new_sentence_padded)
            predicted_class = np.argmax(logit)
            confidence = logit[0][predicted_class]

            if confidence > confidence_threshold:
                Reply_message = str(confidence) + "\n" + Answer_sheet["answer"][predicted_class] 

            else:
                Reply_message = "ขอโทษค่ะ เราไม่เข้าใจคำถามคุณ" + str(confidence) + Answer_sheet['class'][predicted_class] 

               
    
        print("hi " + Reply_message ,flush=True)

        ReplyMessage(Reply_token,Reply_message, Channel_access_token)
        return request.json, 200



        # return jsonify(response)
    elif request.method == 'GET':
        return "this is method GET!!!", 200
    else:
        abort(400)



def flexType(Reply_token,  Line_Access_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply/'

    Authorization = 'Bearer {}'.format(Line_Access_Token)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': Authorization
    }

    flex = '''
    {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
            {
                "type": "text",
                "text": "Type of cafe",
                "size": "xxl",
                "weight": "bold",
                "align": "center"
            }
        ]
    },
    "body": {
        "type": "box",
        "layout": "vertical",
        "contents": [
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่แนวธรรมชาติ",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่สายอาร์ต",
                "align": "center"
            }
        ],
            "margin": "xs"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่สัตว์เลี้ยง",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่โฮมมี่",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่แฟนตาซี",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่วินเทจ",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่แนวมินิมอล",
                "gravity": "top",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่หมา",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่แมว",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่อ่านหนังสือ",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่การ์ตูน",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่แนวญี่ปุ่น",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่แนวเกาหลี",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่ดอกไม้",
                "align": "center"
            }
        ],
            "margin": "md"
        },
        {
            "type": "box",
            "layout": "horizontal",
            "contents": [
            {
                "type": "text",
                "text": "คาเฟ่น่ารัก ๆ ชมพู",
                "align": "center"
            },
            {
                "type": "text",
                "text": "คาเฟ่บอร์ดเกม",
                "align": "center"
            }
        ],
            "margin": "md"
        }
        ],
            "backgroundColor": "#F1F0E8"
            },
            "styles": {
            "header": {
            "backgroundColor": "#ADC4CE"
            }
        }
    } '''

    flexType = json.loads(flex)

    data = {
        "replyToken": Reply_token,
        "messages": [
            {
                "type": "flex",
                "altText": "Type of cafe options",  # Specify the altText
                "contents": flex
            }
        ]
    }


    data = json.dumps(data)  # Convert to JSON
    r = requests.post(LINE_API, headers=headers, data=data)

    # Log the response
    logging.info(f"Response from LINE API: {r.status_code} - {r.text}")
    return 200


def ReplyMessage(Reply_token, TextMessage, Line_Acces_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply/'

    Authorization = 'Bearer {}'.format(Line_Acces_Token)
    headers = {
        'Content-Type': 'application/json; char=UTF-8',
        'Authorization': Authorization
    }

    data = {
        "replyToken": Reply_token,
        "messages": [{
            "type": "text",
            "text": TextMessage
        }
        ]
    }
    data = json.dumps(data)  # Convert to JSON

    # Log the message being sent
    logging.info(f"Sending message: {TextMessage}")

    # Send the HTTP POST request to the LINE API
    r = requests.post(LINE_API, headers=headers, data=data)

    # Log the response
    logging.info(f"Response from LINE API: {r.status_code} - {r.text}")

    return 200

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        filename='myapp.log',  # Specify the log file
        # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        level=logging.DEBUG,
        # Define log message format
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'  # Define date and time format
    )

    # Your application code here
    app.run(debug=True)
