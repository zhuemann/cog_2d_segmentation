from transformers import AutoModelForCausalLM, AutoTokenizer

def run_mixstal():



    device = "cuda"  # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    prompt = "My favourite condiment is"

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    print(tokenizer.batch_decode(generated_ids)[0])