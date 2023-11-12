import os.path
from framework import FaithScore
import os
# from faithscore import FaithScore
# img_path = "/home/lxj220018/data/coco/val2014"
# llava_llava = []
# with open("/home/lxj220018/evaluate/answer/llava_llava/answer_llava_llava.jsonl") as f:
#     for line in f:
#         line = eval(line.strip())
#         line['image'] = os.path.join(img_path, "COCO_val2014_" + line["question_id"] + ".jpg")
#         llava_llava.append(line)
# print(llava_llava[45])
# images = [llava_llava[i]['image'] for i in range(10)]
# answers = [llava_llava[i]['text'] for i in range(10)]

images = ["./COCO_val2014_000000164255.jpg"]
answers = ["The main object in the image is a colorful beach umbrella."]

vemodel = "damo/ofa_visual-entailment_snli-ve_large_en"
score = FaithScore(vem_type="llava", model_path=vemodel, api_key="xxx",
                   llava_path="/home/lxj220018/llava15/llava/eval/checkpoints/llava-v1.5-13b", use_llama=False,
                   llama_path="/home/data/llama2/llama-2-7b-hf")

print(score.faithscore(answers, images))
