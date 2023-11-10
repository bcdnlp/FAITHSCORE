# FaithScore
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arxiv](https://img.shields.io/badge/arXiv-2311.01477-b31b1b.svg)](https://arxiv.org/abs/2311.01477)
<!-- [![PyPI version factscore](https://badge.fury.io/py/factscore.svg)](https://pypi.python.org/pypi/factscore/) -->
<!-- [![Downloads](https://pepy.tech/badge/factscore)](https://pepy.tech/project/factscore) -->

This is the official release accompanying our paper, [FAITHSCORE: Evaluating Hallucinations in Large Vision-Language Models](https://arxiv.org/abs/2311.01477). FAITHSCORE is available as a PIP package as well.

If you find FAITHSCORE useful, please cite:
```
@misc{faithscore,
      title={FAITHSCORE: Evaluating Hallucinations in Large Vision-Language Models}, 
      author={Liqiang Jing and Ruosen Li and Yunmo Chen and Mengzhao Jia and Xinya Du},
      year={2023},
      eprint={2311.01477},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Install

1. Install [LLaVA 1.5](https://github.com/haotian-liu/LLaVA) 
2. Install [modelscope](https://modelscope.cn/home);
   ```python
   pip install modelscope
   pip install "modelscope[multi-modal]" 
   ```
3. Install our package.
    ```python
    pip install -i https://test.pypi.org/simple/ 
    ```
   
## Evaluate Answers Generated by Large Vision-Language Models
You can evaluate answers generated by large vision-language models via our metric. 

```python
from faithscore.framework import FaithScore

images = ["./COCO_val2014_000000164255.jpg"]
answers = ["The main object in the image is a colorful beach umbrella."]

vemodel = "damo/ofa_visual-entailment_snli-ve_large_en"
scorer = FaithScore(vem_type="llava", model_path=vemodel, api_key="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",
                   llava_path="/home/lxj220018/llava15/llava/eval/checkpoints/llava-v1.5-13b", use_llama=False,
                   llama_path="/home/data/llama2/llama-2-7b-hf", tokenzier_path="/home/data/llama2/tokenizer.model")
score, sentence_score = scorer.faithscore(answers, images

```

## Data
The data is given in a json format file. For example, 
```python
{"id": "000000525439", "answer": "The skateboard is positioned on a ramp, with the skateboarder standing on it.", "stage 1": {"The skateboard is positioned on a ramp": 1, " with the skateboarder standing on it": 1}, "stage 2": {"There is a skateboard.": 1, "There is a ramp.": 0, "There is a skateboarder.": 1, "The skateboarder is standing on a skateboard.": 0}}
```
In the stage 1, "1" denotes descriptive sub-sentence, "0" denotes analytical sub-sentence.

In the stage 2, "1" denotes the content is hallucination, "0" denotes the content is not hallucination.

You can download our [annotation dataset](https://github.com/bcdnlp/FAITHSCORE/blob/main/annotation.jsonl).
