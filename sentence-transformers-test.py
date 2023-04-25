import time
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
# import torch
# from datasets import load_dataset

print('Hi mom!')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# faqs_embeddings = load_dataset('ITESM/embedded_faqs_medicare')
# dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

dataset = [
    'Bot: This is an example conversation',
    'Human: That\'s crazy, how does that work?',
    'Bot: Because you typed this manually into the Python document',
    'Human: Oh yeah, lmao',
    'Bot: Among Us',
    'Human: I\'m 69 years old by the way, please remember this',
    'Bot: Ok, I\'ll try to!',
    'Human: You\'d better remember it!',
    'Bot: As an AI language model, your mom\'s a ho',
    'Human: You\'d better also remember that I live in the same country as you',
    'Bot: Got it, I\'ll keep that in mind.',
    'Human: How old are you?',
    'Bot: I\'m an AI language model and don\'t have any concept of an age.'
    ]
queries = ['Do you remember how old I am?']

start_time = time.perf_counter()
dataset_embeddings = embedding_model.encode(dataset)
end_time = time.perf_counter()
print(f"Encoded {len(dataset)} dataset entries in {end_time - start_time:.2f} seconds")

start_time = time.perf_counter()
query_embeddings = embedding_model.encode(queries)
end_time = time.perf_counter()
print(f"Encoded {len(queries)} queries in {end_time - start_time:.2f} seconds")

#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")

print(" --- ")

# print(hits)
k=5
start_time = time.perf_counter()
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=k)
end_time = time.perf_counter()
print(f"Semantic search took {end_time - start_time} seconds")
print(f"Query: {queries[0]}")
print(f"Top {k} matches:")
for i in range(len(hits[0])):
    print( f"id: {hits[0][i]['corpus_id']:2} [{hits[0][i]['score']*100:.2f}%]  >  {dataset[hits[0][i]['corpus_id']]}" )