
from modelscope.outputs import OutputKeys


def llava15(image_path, prompt, model):
    return model.eval(image_path, prompt)

def blip_2(image_path, prompt, model, vis_processors):
    raw_image = Image.open(image_path).convert(
        "RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    ans = model.generate({"image": image, "prompt": prompt+" Answer:"})
    # print(ans)
    return ans[0]

def ofa(entail, model, prompt, image):
    if isinstance(image, list) and len(image) == 1:
        image = image[0]
    elif isinstance(image[0], list):
        image = [img[0] for img in image]
    BS = len(prompt)
        
    input = [{'image': im,
             'text': p} for im,p in zip(image, prompt)]

    # print(input)
    output = [model(inp) for inp in input]
    if entail:
        ans = [out[OutputKeys.LABELS][0] if isinstance(out[OutputKeys.LABELS], list) else out[OutputKeys.LABELS] for out in output]
    else:
        ans = [out[OutputKeys.TEXT][0] if isinstance(out[OutputKeys.TEXT], list) else out[OutputKeys.TEXT] for out in output]
    return ans

def mplug(image, prompt, mplug_model):
    input = {'image': image,
             'text': prompt}

    output = mplug_model(input)
    # ans = output[OutputKeys.TEXT][0]
    ans = output['text']
    return ans

