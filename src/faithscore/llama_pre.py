from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import transformers
import torch
from tqdm import tqdm
import re
import os
path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")


def load_llama(llama_path, BS=32):
    tokenizer = AutoTokenizer.from_pretrained(llama_path, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(llama_path, quantization_config=quantization_config)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        batch_size=BS,
    )
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    pipeline.tokenizer.padding_side = "left"
    pipeline.tokenizer.truncation_side = "left"
    return pipeline, tokenizer

def stage1_llama(pipeline, tokenizer, answer):
    BS = pipeline._batch_size
    with open(os.path.join(cur_path, "prompts/prompt_llama.txt"), "r") as f:
        prompt_label_des_ana = f.read() + "\n\n"
    # print(answer)
    all_subs = [[s.replace("\n", " ").replace('"', '') for s in re.split(r'[,.]', ans) if len(s.replace("\n", " ").replace('"', '').strip()) > 1] for ans in answer]
    lens = [len(subs) for subs in all_subs]
    flattened_subs = [s for subs in all_subs for s in subs]
    result = ["" for _ in range(len(flattened_subs))]
    pts = [prompt_label_des_ana + subs.replace("\n", " ") + "\n" + "This sub-sentence is " for subs in flattened_subs]
    with torch.no_grad():
        sequences =[]
        print("Stage 1: LLaMA")
        for i in tqdm(range(0, len(pts), BS)):
            sequences.extend([s[0]["generated_text"].split("\n")[0] for s in pipeline(
                pts[i:i+BS],
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=10,
                return_full_text=False,
                batch_size=BS
            )])
    # print("sequences: ", sequences)
    #unflattening the sequences
    result =  [result[i]+ seq for i,seq in enumerate(sequences)]
    # print("responces: ", response)
    # print()
    for i,seq in enumerate(sequences):
        if "analytical" in seq.lower():
            result[i] = flattened_subs[i] + " [A] "
        else:
            result[i] =  flattened_subs[i] + " [D] "
    final_res = []
    prev = 0
    for length in lens:
        final_res.append(result[prev:prev+length])
        prev+=length
    return final_res

def stage2_llama(pipeline, tokenizer, pts):
    # print(pts)
    BS = pipeline._batch_size
    with torch.no_grad():
        result =[]
        for i in tqdm(range(0, len(pts), BS)):
            result.extend([s[0] for s in pipeline(
                pts[i:i+BS],
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                return_full_text=False,
                
            )])
        # print("sequences: ", sequences)
        response = [res['generated_text'].split("\n\n")[0] for res in result]
        # print("responces: ", response)
        # print()
    return response
