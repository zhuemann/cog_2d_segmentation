import torch
from transformers import LlamaModel, AutoTokenizer
import os


def precomputed_language_embeddings():

    dir_base = "/UserData/"
    model_name = os.path.join(dir_base, 'Zach_Analysis/models/llama3.1/')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(model_name)

    texts = ["Text sample 1", "Text sample 2"]

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # Example: using CLS token
            embeddings.append(text_embeddings)

    # Save embeddings to file
    torch.save(embeddings, 'precomputed_embeddings.pt')