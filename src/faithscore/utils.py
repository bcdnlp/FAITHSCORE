
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
    input = {'image': image,
             'text': prompt}

    # print(input)
    output = model(input)
    if entail:
        ans = output[OutputKeys.LABELS][0]
    else:
        ans = output[OutputKeys.TEXT][0]
    return ans

def mplug(image, prompt, mplug_model):
    input = {'image': image,
             'text': prompt}

    output = mplug_model(input)
    # ans = output[OutputKeys.TEXT][0]
    ans = output['text']
    return ans

