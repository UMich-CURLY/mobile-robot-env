import numpy as np
import ast
import jsonschema
from jsonschema import validate
from utils.json_validate import schema
import time
import requests
import json
from arguments import get_args


gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo-0125',
            'gpt-4o',
            'gpt-4o-mini'
        ]           

args = get_args()

import openai
from openai.error import OpenAIError
openai.api_key = args.api
api_key = args.api



def chat_with_gpt(chat_history):
    retries = 10    
    while retries > 0:  
        try: 
            response = openai.ChatCompletion.create(
                model='gpt-4o', 
                response_format = { "type": "json_object" },
                messages=chat_history,
                temperature=0.5,
            )

            response_message = response.choices[0].message['content']
            # print(response_message)
            print('gpt-4o' + " response: ")
            try:
                ground_json = ast.literal_eval(response_message)
                break
                
            except (SyntaxError, ValueError) as e:
                print(e)
        except OpenAIError as e:
            if e:
                print(e)
                print('Timeout error, retrying...')    
                retries -= 1
                time.sleep(5)
            else:
                raise e
            
    print(response_message)
    return response_message

def chat_with_gpt4v(chat_history, gpt_type = args.gpt_type):
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    retries = 10    
    while retries > 0:  
        try: 
            payload = {
                "model": gpt_name[gpt_type],
                "response_format": { "type": "json_object" },
                "temperature": 1.1,
                "messages": chat_history,
                "max_tokens": 100
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response_message = response.json()["choices"][0]["message"]["content"]
            # print(response_message)
            print(gpt_name[gpt_type] + " response: ")
            try:
                ground_json = ast.literal_eval(response_message)
                break
                
            except (SyntaxError, ValueError) as e:
                print(e)
        except OpenAIError as e:
            if e:
                print(e)
                print('Timeout error, retrying...')    
                retries -= 1
                time.sleep(5)
            else:
                raise e
            
    print(response_message)
    return response_message

def chat_with_llama3(user_prompt, chat_history):

    chat_history.append({"role": "user", "content": user_prompt})

    retries = 10    
    while retries > 0:  
        
        url = 'http://localhost:11434/api/chat'
        data = {
            "model": "llama3",
            "messages": chat_history,
            "stream": False,
            "format": "json",
            "options": {
                "seed": 101,
                "temperature": 0
            }
        }
        
        response = requests.post(url, json=data)
        decoded_string = response.content.decode('utf-8')
        response_message = json.loads(decoded_string)["message"]["content"]
        ground_json = ast.literal_eval(response_message)
                
        try: 
            validate(instance = ground_json, schema = schema)
        except jsonschema.exceptions.ValidationError as err:
            print("JSON data is invalid", err)
            print('Json format error, retrying...')    
            retries -= 1

            
    chat_history.append({"role": "assistant", "content": response_message})
    print(response_message)
    return response_message

