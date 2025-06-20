import ollama
import re
import json
import requests
import os
import cv2
import base64
from typing import List, Dict, Any
from utils.vln_logger import vln_logger
from textblob import TextBlob
import numpy as np
# from openai import OpenAI

def load_json(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    return data

def save_json(data, filename):
    """
    Saves a Python dictionary (or any serializable object) to a JSON file.

    Args:
    - data: The data to save (should be a dictionary or list).
    - filename: The file path (including filename) where the data should be saved.
    """
    try:
        # Open the file in write mode and dump the JSON data into it
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        # print(f"Data has been successfully saved to {filename}")
    
    except Exception as e:
        print(f"Error saving the data to {filename}: {e}")

def construct_query(phase, **placeholders):
    """
    Construct a natural language query for the LLM.
    
    Args:
        phase (dict): Dictionary containing the phase prompts.
        **placeholders: Arbitrary keyword arguments representing the dynamic placeholders.

    Returns:
        str: The formatted query string.
    """
    phase_prompt = "\n\n".join(phase['phase_prompt'])
    content = phase_prompt.format(**placeholders)
    return content

def construct_target_query(candidate_object, goal_class, image_history):
    '''
    Input:
        candidate_object: dict, store all candidate object info, from the object list
        goal_class: str, the goal class name
    Output:
        prompt: str, the prompt for the LLM
        image: np.ndarray, the image with xyxy bbox of the candidate object
    '''
    prompt = f"You are a verification agent that helps a robot in a 3D environment looking for objects in the scene. Please verify if the object in the image enclosed by the red bounding box is a '{goal_class}' or not. Answer with 'yes' or 'no' with no additional text." 
    
    # if the image with highest confidence is not the same as the goal class, then it is not a valid candidate
    max_conf_index = np.argmax(candidate_object['conf'])
    xyxy = candidate_object['xyxy'][max_conf_index]
    image_idx = candidate_object['image_idx'][max_conf_index]
    image = np.array(image_history[image_idx])
    # draw red bounding box on the image
    image = cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
    
    return prompt, image
    
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_llm(prompt, llm_name='gpt-4o-mini', image=None):
    """
    Query an LLM (like GPT-4) to predict what exists at the frontier location.
    """
    if 'gpt' in llm_name:
        llm_label = 'gpt'
    elif 'llama' in llm_name:
        llm_label = 'llama'
    elif 'qwen' in llm_name:
        llm_label = 'qwen'
    else:
        llm_label = 'other'
    vln_logger.increase_function_call_count('query_llm', args={'llm_name': llm_label})

    if image is None:
        if llm_name == 'gpt':
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}
                ]}] 
        else:
            messages = [{'role': 'user', 'content': prompt}]
    else:
        if isinstance(image, str) and os.path.exists(image):
            base64_image = encode_image(image)
        elif isinstance(image, np.ndarray):
            retval, buffer = cv2.imencode('.png', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError(f"Invalid image type: {type(image)}")
        if llm_name == 'gpt':
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ]}] 
        else:
            messages = [{
                'role': 'user',
                'content': prompt,
                'images': [base64_image]
            }]
    if 'gpt' in llm_name:
        url = "http://localhost:8888/v1/chat/completions"
        max_tokens_map = {
            "gpt-4o": 16384,
            "gpt-4o-2024-11-20": 16384,
            "o1-preview": 32768,
            "gpt-4o-mini": 16384,
            "gpt-4.1-nano": 16384
        }
        if llm_name == "gpt":
            model = "gpt-4o-mini" # "gpt-4.1-nano"
        else:
            model = llm_name
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens_map[model]
        })
        api_key = os.environ["GPT_API_KEY"]
        headers = {
            'Authorization': api_key,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            # 'Host': '47.88.65.188:8405',
            'Connection': 'keep-alive'
            }
        response = requests.request("POST", url, headers=headers, data=payload)
        try:
            response = response.json()['choices'][0]['message']['content']
        except:
            print("[query_llm] ERROR")
            print(response.json())

        return response
    else:
        client = ollama.Client(
            host=os.getenv('OLLAMA_HOST')
        )
        response = client.chat(model=llm_name, messages=messages)
        return response.message.content.lower()

def format_graph(response):
    regex = r"(.+?)\n```.*?\n(.*?)```"
    matches = re.finditer(regex, response, re.DOTALL)
    graph = None
    for match in matches:
        graph = match.group(2)
        try:
            graph = json.loads(graph)
            break
        except:
            graph = None
            pass
    return graph

# def format_response(response):
#     response = response.lower().strip()

#     # Regex to match **answer:**, # " or a direct answer
#     match = re.search(r"\*\*answer:\*\* (.+)|#\s*answer:\s*(.+)|\*answer:\* (.+)|\n\nanswer:\n\n (.+)|# \"(.+)\"|# \s*(\w[\w\s]*)?|^\s*(\w[\w\s]*)$", response, re.MULTILINE)

#     if match:
#         answer = match.group(1) or match.group(2) or match.group(3) or match.group(4) or match.group(5) or match.group(6) or match.group(7)
#         answer = answer.strip()
#     else:
#         answer = response
#     return answer

def clean_markdown(text):
    return re.sub(r'^[#\*]+|[#\*]+$', '', text).strip()

def correct_text(text):
    return str(TextBlob(text).correct())

def format_response(response):
    try:
        response = response.lower().strip()

        # Updated pattern with more formats added
        pattern = re.compile(
            r"\*\*answer:\*\*\s*(.+)"            # **Answer:** value
            r"|\*answer:\*\s*(.+)"               # *Answer:* value
            r"|#\s*answer:\s*(.+)"               # # Answer: value
            r"|\n\nanswer:\n\n\s*(.+)"           # Answer in a block
            r"|>\s*answer:\s*(.+)"               # > Answer: value (blockquote style)
            r"|###\s*answer\s*[:\-]?\s*(.+)"     # ### Answer or ### Answer:
            r"|#\s*\"(.+?)\""                    # # "answer"
            r"|\*\*(.+?)\*\*"                    # **answer**
            r"|\*(.+?)\*"                        # *answer*
            r"|^\s*([\w\s\-]+)$",                # plain answer
            re.MULTILINE
        )

        match = pattern.search(response)

        if match:
            # Get the first non-None group
            answer = next(group for group in match.groups() if group)
            answer = clean_markdown(answer)
            answer = correct_text(answer)
        else:
            answer = clean_markdown(response)
            answer = correct_text(answer)

        return answer.strip()
    except:
        print("[format_response] ERROR")
        print(response)
        return ""


def text2value(text):
    try:
        value = float(text)
    except:
        value = 0
    return value

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query
    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    return query

class EnvironmentHistory:
    def __init__(self, base_query: str, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['prediction', 'observation', 'summary']
        self._history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            s += item['value']
            if i != len(self._history) - 1:
                s += '\n'
        return s

def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            # reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            # reflection: str = query_llm(reflection_query) # type: ignore
            # env_configs[i]['memory'] += [reflection]
                
    return env_configs