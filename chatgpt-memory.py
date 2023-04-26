import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# OPTIONS ####################################################################################################################################
verbosity = False
gpt = "GPT-4"        # GPT-3.5-Turbo-0301 | GPT-4
max_tokens = 200     # Max tokens that OpenAI can return
max_conv_length = 5  # Max length of conversation buffer
k = 5                # Number of results to return from semantic search
system_prompt = "You are a friendly AI assistant conversing with a Human. The user will type messages to you, and you will respond back in text. A semantic search will be run on the entire chat history to source relevant information as necessary. If you cannot answer a topic to the best of your ability, please truthfully answer that you do not know." 


# Variables ###################################################################################################################################
chatGPT_message_buffer = [] # Message buffer used for context i.e. ChatGPT's short term memory
entire_message_history = [] # Entire chat history, in non-vector form
chat_history_embeddings = np.empty((0, 384), dtype='float32') # Entire history, embedded in vector form


# Functions #######################################################################################################################
def verbose_print(text):
    """ Print only if verbosity is set to True """
    if (verbosity):
        print(text)


def load_history():
    """ Loads entire chat history from CSV file, properly populates vector array in memory, and initializes ChatGPT message buffer """
    global chat_history_embeddings
    global chatGPT_message_buffer
    if not os.path.isfile('history.csv'):
        print("No history file")
        return
    with open('history.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        loaded_tensor_stack = []
        for row in csvreader:
            entire_message_history.append(row[0].strip('"'))
            loaded_tensor = np.array(row[1:], dtype='float32')
            loaded_tensor_stack.append(loaded_tensor)
        chat_history_embeddings = np.vstack(loaded_tensor_stack)
        old_messages_str = entire_message_history[-max_conv_length:]
        old_messages_json = []
        for message in old_messages_str:
            role, content = message.split(": ", 1)
            json_message = {'role': role.strip(), 'content': content.strip()}
            old_messages_json.append(json_message)
        chatGPT_message_buffer = old_messages_json


def save_history():
    """ Saves entire chat history and vector array to CSV file """
    sanitized_data = []
    for i in range(len(entire_message_history)):
        sanitized_row = [f'"{entire_message_history[i]}"'] + [str(val) for val in chat_history_embeddings[i]]
        sanitized_data.append(sanitized_row)
        row = [entire_message_history[i]] + chat_history_embeddings[i].tolist()
    with open('history.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(sanitized_data)


def append_message(message):
    """ Appends a message json object to both ChatGPT message buffer and chat history """
    chatGPT_message_buffer.append(message)
    messageContent = message['content']
    messageContent = messageContent[:min(len(messageContent), 512)]
    message_as_str = f'{message["role"]}: {message["content"]}'
    entire_message_history.append(message_as_str)
    embedOne(message_as_str)


def embedAll():
    """ Vector embeds entire chat history """
    global chat_history_embeddings
    start_time = time.perf_counter()
    chat_history_embeddings = embedding_model.encode(entire_message_history)
    end_time = time.perf_counter()
    verbose_print(f"Encoded {len(entire_message_history)} dataset entries in {end_time - start_time:.4f} seconds")


def embedOne(message):
    """ Vector embeds one message and adds it to chat history embeddings """
    global chat_history_embeddings
    start_time = time.perf_counter()
    message_as_vector = embedding_model.encode([message])
    end_time = time.perf_counter()
    verbose_print(f"Encoded single message in {end_time - start_time:.4f} seconds")
    chat_history_embeddings = np.vstack((chat_history_embeddings, message_as_vector[0]))
    pass

def chatgpt_req(text):
    """ Sends text to OpenAI, gets the response, and puts it into the chatbox """
    # Query chat history
    hitsAsString = f"user: {text}" #if no embeddings present, return a string with just the user's message alone
    if (len(entire_message_history)):
        query_embedding = embedding_model.encode(f'user: {text}')
        start_time = time.perf_counter()
        hits = semantic_search(query_embedding, chat_history_embeddings, top_k=k)
        end_time = time.perf_counter()
        verbose_print(f"Semantic search took {end_time - start_time} seconds")
        hitsAsString = '\n'.join([entire_message_history[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
    # Add user's message to the chat buffer
    append_message({"role": "user", "content": text})
    while len(chatGPT_message_buffer) > max_conv_length:  # Trim down chat buffer if it gets too long
        chatGPT_message_buffer.pop(0)
    # Init system prompt with date and add it persistently to top of chat buffer
    # Also inject top results from semantic search 
    systemPromptObject = [{"role": "system", "content":
                           system_prompt
                           + f' The current date and time is {datetime.now().strftime("%A %B %d %Y, %I:%M:%S %p")} Eastern Standard Time.'
                           + f' You are using {gpt} from OpenAI.'
                           + f' Based on the semantic search preformed on chat history, the top {min(len(entire_message_history),k)} results that came up most relevant to what the user typed were as follows:\n'
                           + hitsAsString
                        }]
    # create object with system prompt and chat history to send to OpenAI for generation
    messagePlusSystem = systemPromptObject + chatGPT_message_buffer
    err = None
    try:
        start_time = time.time()
        completion = openai.ChatCompletion.create(
            model=gpt.lower(),
            messages=messagePlusSystem,
            max_tokens=max_tokens,
            temperature=0,
            frequency_penalty=0.2,
            presence_penalty=0.5,
            logit_bias={'1722': -100, '292': -100, '281': -100, '20185': -100, '9552': -100, '3303': -100, '2746': -100, '19849': -100, '41599': -100, '7926': -100}
            # 'As', 'as', ' an', 'AI', ' AI', ' language', ' model', 'model', 'sorry', ' sorry'
            )
        end_time = time.time()
        verbose_print(f'--OpenAI API took {end_time - start_time:.3f}s')
        result = completion.choices[0].message.content
        append_message({"role": "assistant", "content": result})
        save_history()
        print(f"assistant: {result}\n")
    except openai.APIError as e:
        err = e
        print(f"!!Got API error from OpenAI: {e}")
    except openai.InvalidRequestError as e:
        err = e
        print(f"!!Invalid Request: {e}")
    except openai.OpenAIError as e:
        err = e
        print(f"!!Got OpenAI Error from OpenAI: {e}")
    except Exception as e:
        err = e
        print(f"!!Other Exception: {e}")
    finally:
        if err is not None: print(f'⚠ {err}')


if __name__ == '__main__':
    load_history()
    while True:
        text = input('user: ')
        chatgpt_req(text)