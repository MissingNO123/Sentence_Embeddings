import time
full_start_time = time.time()
from datetime import datetime
from dotenv import load_dotenv
import openai
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# OPTIONS ####################################################################################################################################
verbosity = False
gpt = "GPT-4"       # GPT-3.5-Turbo-0301 | GPT-4
max_tokens = 200    # Max tokens that openai will return
max_conv_length = 5 # Max length of conversation buffer
k = 5
systemPrompt = "You are a friendly AI assistant conversing with a Human. The user will type messages to you, and you will respond back in text. A semantic search will be run on the entire chat history to source relevant information as necessary. If you cannot answer a topic to the best of your ability, please truthfully answer that you do not know." 


# Variables ###################################################################################################################################
chatGPTMessageBuffer = [] # List of messages sent back and forth between ChatGPT / User, initialized with example messages
entireMessageHistory = []
chathistory_embeddings = embedding_model.encode(entireMessageHistory)


# Functions #######################################################################################################################
def verbose_print(text):
    if (verbosity):
        print(text)


def append_message(message):
    global chathistory_embeddings
    chatGPTMessageBuffer.append(message)
    messageContent = message['content']
    messageContent = messageContent[:min(len(messageContent), 512)]
    entireMessageHistory.append(f"{message['role']}: {messageContent}")
    start_time = time.perf_counter()
    chathistory_embeddings = embedding_model.encode(entireMessageHistory)
    end_time = time.perf_counter()
    verbose_print(f"Encoded {len(entireMessageHistory)} dataset entries in {end_time - start_time:.4f} seconds")


def chatgpt_req(text):
    """ Sends text to OpenAI, gets the response, and puts it into the chatbox """
    if len(chatGPTMessageBuffer) > max_conv_length:  # Trim down chat buffer if it gets too long
        chatGPTMessageBuffer.pop(0)
    # Query chat history
    hitsAsString = f"user: {text}" #if no embeddings present, return a string with just the user's message alone
    if (len(entireMessageHistory)):
        query_embedding = embedding_model.encode(f'user: {text}')
        start_time = time.perf_counter()
        hits = semantic_search(query_embedding, chathistory_embeddings, top_k=k)
        end_time = time.perf_counter()
        verbose_print(f"Semantic search took {end_time - start_time} seconds")
        hitsAsString = '\n'.join([entireMessageHistory[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
    # Add user's message to the chat buffer
    append_message({"role": "user", "content": text})
    # Init system prompt with date and add it persistently to top of chat buffer
    systemPromptObject = [{"role": "system", "content":
                           systemPrompt
                           + f' The current date and time is {datetime.now().strftime("%A %B %d %Y, %I:%M:%S %p")} Eastern Standard Time.'
                           + f' You are using {gpt} from OpenAI.'
                           + f' Based on the semantic search preformed on chat history, the top {min(len(entireMessageHistory),k)} results that came up most relevant to what the user typed were as follows:\n'
                           + hitsAsString
                        }]
    # create object with system prompt and chat history to send to OpenAI for generation
    messagePlusSystem = systemPromptObject + chatGPTMessageBuffer
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
        print(f"\nassistant: {result}")
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
    while True:
        text = input('user: ')
        chatgpt_req(text)