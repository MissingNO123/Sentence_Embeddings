# OpenAI GPT-4 with Sentence Embeddings

This is a proof of concept for using sentence embeddings to effectively give ChatGPT a form of long-term memory

## But how does it work?

The entire chat history is being stored on a message-by-message basis, i.e. each message is being transformed into a 384-component vector.
Before a message is sent to the API to generate, a semantic search is performed on the vectors and the top 6 most relevant chat messages are returned, which are then inserted into the generation prompt.
Because everything is a vector, you can use algorithms like cosine similarity, Euclidean distance or even dot product to determine similarity between the user's message and messages in chat history.
The sentence_transformers library processes up to 1 million vectors by default in chunks of ~100k, and is extremely fast.

## Usage

Take your OpenAI API key and put it in an environment variable, then run `python .\chatgpt-memory.py`, and start chatting.
Set `verbosity = True` in the file to enable verbose logging.

## Requirements

Python 3.8 or higher with pip, and about 200MB of free disk space.

Required libraries:

- [python-dotenv](https://pypi.org/project/python-dotenv/) if on Windows
- [openai](https://github.com/openai/openai-python)
- [sentence-transformers](https://huggingface.co/sentence-transformers)

Most likely requires an [NVidia GPU](https://new.reddit.com/r/nvidia/comments/yc6g3u/rtx_4090_adapter_burned/). Not tested with AMD, but I doubt it will work. In that case, edit the file to remove the CUDA requirement.

## Copyright

Copyright (c) 2023 MissingNO123. All rights reserved.

The contents of this repository, including all code, documentation, and other materials, unless otherwise specified, are the exclusive property of MissingNO123 and are protected by copyright law. Unauthorized reproduction, distribution, or disclosure of the contents of this repository, in whole or in part, without the express written permission of MissingNO123 is strictly prohibited.

The original version of the Software was authored on the 25th of April, 2023.
