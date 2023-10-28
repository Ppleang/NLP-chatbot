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

Channel_access_token = ""


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

answer2 = gc.open_by_url(
    'https://docs.google.com/spreadsheets/d/1GLkhEngVAxDoyKVa0aSMIM5YS4cPGWv7LTMmWbhNi-g/edit?usp=sharing')
worksheet_3 = answer2.worksheet('Class & Answer')
Answer_flex = pd.DataFrame(worksheet_3.get_all_records())


# def load_keras_model(model_path):
#     global loaded_model
#     loaded_model = load_model(model_path)


model_path = 'app/static/model_train-chat.h5'
loaded_model = load_model(model_path)

with open('app/static/model_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
# load_keras_model(model_path)


# clean emoji & space & text to lower 
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
        # msgType = payload['events'][0]['message']['type']  # Extract the msgType

    # if (msgType == "text") :

        # Extract the user's text message
        message = payload['events'][0]['message']['text']
        logging.debug("hi " + message)

        message = clean_text(message)

        # Create an empty list to store multiple reply messages
        # reply_messages = []

        # Add the first reply message
        if "คาเฟ่แนวธรรมชาติ" in message:
            loopBubble(Reply_token, "คาเฟ่แนวธรรมชาติ", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่สายอาร์ต" in message:
            loopBubble(Reply_token, "คาเฟ่สายอาร์ต", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่สัตว์เลี้ยงนำสัตว์เลี้ยงไปได้" in message:
            loopBubble(Reply_token, "คาเฟ่นำสัตว์เลี้ยงไปได้", Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่แนวโฮมมี่" in message:
            loopBubble(Reply_token, "คาเฟ่แนวโฮมมี่", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่แนวแฟนตาซี" in message:
            loopBubble(Reply_token, "คาเฟ่แนวแฟนตาซี", Channel_access_token)
            return request.json, 200
        elif "คาเฟ่แนววินเทจ" in message:
            loopBubble(Reply_token, "คาเฟ่แนววินเทจ", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่แนวมินิมอล" in message:
            loopBubble(Reply_token, "คาเฟ่แนวมินิมอล", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่หมา" in message:
            loopBubble(Reply_token, "คาเฟ่หมา", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่แมว" in message:
            loopBubble(Reply_token, "คาเฟ่แมว", Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่อ่านหนังสือ" in message:
            loopBubble(Reply_token, "คาเฟ่อ่านหนังสือ", Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่การ์ตูน" in message:
            loopBubble(Reply_token, "คาเฟ่การ์ตูน", Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่แนวญี่ปุ่น" in message:
            loopBubble(Reply_token, "คาเฟ่แนวญี่ปุ่น", Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่แนวเกาหลี" in message:
            loopBubble(Reply_token, "คาเฟ่แนวเกาหลี", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่ดอกไม้" in message:
            loopBubble(Reply_token, "คาเฟ่ดอกไม้", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่น่ารัก" in message:
            loopBubble(Reply_token, "คาเฟ่น่ารัก", Channel_access_token)
            return request.json, 200

        elif "คาเฟ่บอร์ดเกม" in message:
            loopBubble(Reply_token, "คาเฟ่บอร์ดเกม", Channel_access_token)
            return request.json, 200

            # reply_messages.append({"type": "text", "text": "CAFE"})
            # reply_messages.append({"type": "text", "text": Answer_sheet['answer'][1]})
        elif "สวัสดี" in message:
            Reply_message = "สวัสดีค่ะ"
        
        elif "hi" in message:
            Reply_message = "สวัสดีค่ะ"

        elif "hello" in message:
            Reply_message = "สวัสดีค่ะ"

        elif "หวัดดี" in message:
            Reply_message = "สวัสดีค่ะ"
        
        elif "ฮาย" in message:
            Reply_message = "สวัสดีค่ะ"

            # reply_messages.append({"type": "text", "text": "สวัสดีค่ะ"})
        elif "ประเภทคาเฟ่" in message:
            flexType(Reply_token, Channel_access_token)
            return request.json, 200
        
        elif "คู่มือการใช้งาน" in message:
            sendManual(Reply_token, Channel_access_token)
            return request.json, 200
        
        elif "คาเฟ่แนะนำ" in message:
            loopBubble(Reply_token, "คาเฟ่แนะนำ", Channel_access_token)
            return request.json, 200
        
        elif "สุนัข" in message:
            loopBubble(Reply_token, "คาเฟ่หมา", Channel_access_token)
            return request.json, 200

        elif "developer" in message:
            Reply_message = "cream pleang boom p'nam >0<"

            # reply_messages.append({"type": "text", "text": "cream pleang boom >0<"})
        else:
            confidence_threshold = 0.66

            dt = datetime.now()
            data_df = pd.DataFrame.from_records(df)

            new_sentence_tokenized = [word_tokenize(message)]
            new_sentence_sequences = tokenizer.texts_to_sequences(
                new_sentence_tokenized)
            new_sentence_padded = pad_sequences(
                new_sentence_sequences, maxlen=17, padding="post")
            logit = loaded_model.predict(new_sentence_padded)
            predicted_class = np.argmax(logit)
            confidence = logit[0][predicted_class]

            if confidence > confidence_threshold:
                # flexReplyClass(Reply_token,  Answer_sheet["class"][predicted_class], Channel_access_token )
                loopBubble(Reply_token, Answer_sheet["class"][predicted_class], Channel_access_token)
                return request.json, 200

            else:
                Reply_message = "ขอโทษค่ะ เราไม่เข้าใจคำถามคุณ"

        print("hi " + Reply_message, flush=True)

        ReplyMessage(Reply_token, Reply_message, Channel_access_token)
        return request.json, 200

        # else :
        #     randSticker(Reply_token , Channel_access_token)
        #     return request.json, 200

        # return jsonify(response)
    elif request.method == 'GET':
        return "this is method GET!!!", 200
    else:
        abort(400)



def sendManual(Reply_token,  Line_Access_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply/'

    Authorization = 'Bearer {}'.format(Line_Access_Token)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': Authorization
    }
    data = {
        "replyToken": Reply_token,
        "messages": [{
            "type": "image",
            "originalContentUrl": "https://i.imgur.com/NDy2s7o.jpg",
            "previewImageUrl": "https://i.imgur.com/NDy2s7o.jpg"
        }
        ]
    }

    data = json.dumps(data)
    r = requests.post(LINE_API, headers=headers, data=data)
    logging.info(f"Response from LINE API: {r.status_code} - {r.text}")

    return 200


def loopBubble(Reply_token, train_class, Line_Access_Token):
    LINE_API = 'https://api.line.me/v2/bot/message/reply/'

    Authorization = 'Bearer {}'.format(Line_Access_Token)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': Authorization
    }

    res = Answer_flex[Answer_flex['Class'] == train_class]

    # Assuming 'res' is a DataFrame with the data you mentioned

    # Extract data from the DataFrame
    imgUrls = res['Imgurl'].values
    names = res['Name'].values
    stations = res['Station'].values
    contacts = res['Contact'].values
    times = res['Time'].values
    maps = res['Map'].values

    # Initialize an empty list to store bubble data
    bubbles = []

    # Loop through the data and create a bubble for each item
    for i in range(len(res)):
        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": imgUrls[i],
                "aspectMode": "cover",
                "size": "full",
                "aspectRatio": "20:13"
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": names[i],
                        "weight": "bold",
                        "size": "xl"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "lg",
                        "spacing": "sm",
                        "contents": [
                            {
                                "type": "box",
                                "layout": "baseline",
                                "spacing": "sm",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": "Station",
                                        "color": "#7D7C7C",
                                        "size": "sm",
                                        "flex": 0
                                    },
                                    {
                                        "type": "text",
                                        "text": stations[i],
                                        "wrap": True,
                                        "color": "#666666",
                                        "size": "sm",
                                        "flex": 2,
                                        "weight": "bold",
                                        "action": {
                                            "type": "message",
                                            "label": "action",
                                            "text": "station"
                                        }
                                    }
                                ]
                            },
                            {
                                "type": "box",
                                "layout": "baseline",
                                "spacing": "sm",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": "Time",
                                        "color": "#7D7C7C",
                                        "size": "sm",
                                        "flex": 1
                                    },
                                    {
                                        "type": "text",
                                        "text": times[i],
                                        "wrap": True,
                                        "color": "#666666",
                                        "size": "sm",
                                        "flex": 5,
                                        "weight": "bold",
                                        "action": {
                                            "type": "message",
                                            "label": "action",
                                            "text": "hello"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "spacing": "none",
                "offsetTop": "none"
            },
            "footer": {
                "type": "box",
                "layout": "horizontal",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "action": {
                            "type": "uri",
                            "label": "Contact",
                            "uri": contacts[i]
                        },
                        "color": "#C5DFF8"
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "height": "sm",
                        "action": {
                            "type": "uri",
                            "label": "Google Map",
                            "uri": maps[i]
                        },
                        "color": "#CCEEBC"
                    }
                ],
                "flex": 0
            },
            "size": "mega",
            "styles": {
                "body": {
                    "backgroundColor": "#FEFCF3"
                },
                "footer": {
                    "backgroundColor": "#FEFCF3"
                }
            }
        }

        # Add the bubble to the list of bubbles
        bubbles.append(bubble)

    flex = {
        "type": "carousel",
        "contents": bubbles
    }

    flex_str = json.dumps(flex)

    # Create the JSON structure with the list of bubbles

    flexClass = json.loads(flex_str)

    data = {
        "replyToken": Reply_token,
        "messages": [
            {
                "type": "flex",
                "altText": "Classify cafe",  # Specify the altText
                "contents": flexClass
            }
        ]
    }

    data = json.dumps(data)  # Convert to JSON
    r = requests.post(LINE_API, headers=headers, data=data)

    # Log the response
    logging.info(f"Response from LINE API: {r.status_code} - {r.text}")
    return 200


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
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวธรรมชาติ",
                "text": "คาเฟ่แนวธรรมชาติ"
                },
                "style": "secondary"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่สายอาร์ต",
                "text": "คาเฟ่สายอาร์ต"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่นำสัตว์เลี้ยงไปได้",
                "text": "คาเฟ่นำสัตว์เลี้ยงไปได้"
                },
                "style": "secondary",
                "color": "#F2D8D8"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวโฮมมี่",
                "text": "คาเฟ่แนวโฮมมี่"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวแฟนตาซี",
                "text": "คาเฟ่แนวแฟนตาซี"
                },
                "style": "secondary"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนววินเทจ",
                "text": "คาเฟ่แนววินเทจ"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวมินิมอล",
                "text": "คาเฟ่แนวมินิมอล"
                },
                "style": "secondary",
                "color": "#F2D8D8"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่หมา",
                "text": "คาเฟ่หมา"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แมว",
                "text": "คาเฟ่แมว"
                },
                "style": "secondary"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่อ่านหนังสือ",
                "text": "คาเฟ่อ่านหนังสือ"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่การ์ตูน",
                "text": "คาเฟ่การ์ตูน"
                },
                "style": "secondary",
                "color": "#F2D8D8"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวญี่ปุ่น",
                "text": "คาเฟ่แนวญี่ปุ่น"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่แนวเกาหลี",
                "text": "คาเฟ่แนวเกาหลี"
                },
                "style": "secondary"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่ดอกไม้",
                "text": "คาเฟ่ดอกไม้"
                }
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่น่ารัก ๆ",
                "text": "คาเฟ่น่ารัก"
                },
                "color": "#F2D8D8",
                "style": "secondary"
            },
            {
                "type": "button",
                "action": {
                "type": "message",
                "label": "คาเฟ่บอร์ดเกม",
                "text": "คาเฟ่บอร์ดเกม"
                }
            }
            ],
            "backgroundColor": "#F1F0E8"
        },
        "size": "mega",
        "styles": {
            "header": {
            "backgroundColor": "#F5F0BB"
            }
        }
    }'''

    flexType = json.loads(flex)

    data = {
        "replyToken": Reply_token,
        "messages": [
            {
                "type": "flex",
                "altText": "Type of cafe options",  # Specify the altText
                "contents": flexType
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
