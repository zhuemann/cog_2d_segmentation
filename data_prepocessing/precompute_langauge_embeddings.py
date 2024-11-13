import torch
from transformers import LlamaModel, AutoTokenizer, RobertaModel, BertModel, AutoModel
import os
import json
from pathlib import Path



def precomputed_language_embeddings():

    dir_base = "/UserData/"
    model_name = os.path.join(dir_base, 'Zach_Analysis/models/llama3.1/')

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = LlamaModel.from_pretrained(model_name)
    lang_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    lang_path = os.path.join(dir_base, 'Zach_Analysis/models/rad_bert/')
    lang_path = os.path.join(dir_base, 'Zach_Analysis/language_models/bert/')

    tokenizer = AutoTokenizer.from_pretrained(lang_path)

    model = AutoModel.from_pretrained(lang_path, output_hidden_states=True)
    # Load JSON data
    with open("/UserData/Zach_Analysis/uw_lymphoma_pet_3d/final_training_testing_v6.json", "r") as file:
        data = json.load(file)

    # Base directory to store embeddings
    embedding_base_dir = Path("embeddings")
    embedding_base_dir.mkdir(exist_ok=True)

    # Process each sample and save embeddings with <label_name>_embedding.pt naming
    for subset in ['training', 'testing']:
        for sample in data[subset]:
            report = sample['report']
            label_name = sample['label_name']  # Assuming each sample has a 'label_name' field

            # Tokenize and get embeddings for every token in the report
            inputs = tokenizer(report, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                # Get embeddings for every token (full last hidden state)
                token_embeddings = outputs.last_hidden_state.squeeze()  # Shape: (sequence_length, hidden_size)
                print(f"token size: {token_embeddings.size()}")

            # Define the path based on label_name
            embedding_path = embedding_base_dir / f"{label_name}_embedding.pt"

            # Save the full token embeddings to the specified path
            torch.save(token_embeddings, embedding_path)

            # Update the JSON with the path to the embedding file
            sample['embedding_path'] = str(embedding_path)

    # Save the updated JSON
    with open("final_training_testing_v6_with_embeddings.json", "w") as file:
        json.dump(data, file, indent=4)