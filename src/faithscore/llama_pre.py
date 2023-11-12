from transformers import AutoTokenizer
import transformers
import torch
import re
import os
path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")

def load_llama(llava_path):
    tokenizer = AutoTokenizer.from_pretrained(llava_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=llava_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline, tokenizer

def stage1_llama(pipeline, tokenizer, answer):
    with open(os.path.join(cur_path, "prompts/prompt_llama.txt"), "r") as f:
        prompt_label_des_ana = f.read() + "\n\n"
    # print(answer)
    result = ""
    for subs in re.split(r'[,.]', answer):
        if len(subs.replace(" ", "")) > 0:
            pts = prompt_label_des_ana + subs.replace("\n", " ") + "\n" + " This sub-sentence is "
            # print(pts)
            with torch.no_grad():
                sequences = pipeline(
                   pts,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=1000,
                )
                # print("sequences: ", sequences)
                response = sequences[0]['generated_text'][len(pts):len(pts)+15]
                # print("responces: ", response)
                # print()
            if "analytical" in response.lower():
                result += subs + " [A] "
            else:
                result += subs + " [D] "
    return result
