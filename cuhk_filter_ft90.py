import json
import random
import torch
import os
from PIL import Image
from lavis.models import load_model_and_preprocess

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
##################### step1 处理cuhk的caption
# 读取 JSON 文件获得prompt
cuhk_prompt_list = []

prompt_json = '/xxx/data/finetune/cuhk_train.json'
with open(prompt_json, 'r') as file:
    data = json.load(file)
    for item in data:
        prompt = item['caption']
        cuhk_prompt_list.append(prompt)

cuhk_caption_list = cuhk_prompt_list   # prompt list

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                  model_type="pretrain", is_eval=True,
                                                                  device=device)
feature_cuhk_list = []
for c in range(0,10000):
    k = random.randint(0, len(cuhk_caption_list) - 1)
    text_input_cuhk = txt_processors["eval"](cuhk_caption_list[k])
    sample = {"text_input": [text_input_cuhk]}
    # features_multimodal = model.extract_features(sample)
    # print(features_multimodal.multimodal_embeds.shape)
    # # torch.Size([1, 32, 768]), 32 is the number of queries
    features_text_cuhk = model.extract_features(sample, mode="text")
    feature_cuhk_list.append(features_text_cuhk)
print(feature_cuhk_list)

######################## 提取文件内的caption
# 指定 JSON 文件的路径
gen_json = "/xxx/data/finetune/cuhk_train.json"
gen_data_dict = {}
# 打开并读取 JSON 文件
with open(gen_json, "r") as json_file:
    json_dict = json.load(json_file)

    # 遍历 JSON 文件中的每个条目
    for item in json_dict:
        image_path = item["image"]  # 直接使用完整的路径
        caption = item["caption"]  # 获取图片的描述

        # 检查完整的图片路径是否已经存在于字典中
        if image_path not in gen_data_dict:
            # 如果不存在，初始化一个列表来存储这个图片路径对应的所有描述
            gen_data_dict[image_path] = []

        # 将描述添加到图片路径对应的列表中
        gen_data_dict[image_path].append(caption)

exceed_threshold_image_captions = []

# 遍历每个图像及其描述
for image_relative_path, captions in gen_data_dict.items():
    # 构建完整图像路径
    image_full_path = os.path.join("/xxx/images/CUHK-PEDES/", image_relative_path)

    try:
        raw_image = Image.open(image_full_path).convert("RGB")
    except FileNotFoundError:
        print(f"File not found: {image_full_path}")
        continue

    for caption in captions:
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}

        features_image = model.extract_features(sample, mode="image")
        features_text = model.extract_features(sample, mode="text")
        similarity_self = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:, 0, :].t()).max().item()

        # 与其他10000个描述计算相似度
        similarities = [similarity_self]
        for feature_cuhk in feature_cuhk_list[:10000]:
            similarity = (features_image.image_embeds_proj @ feature_cuhk.text_embeds_proj[:, 0, :].t()).max().item()
            similarities.append(similarity)

        # 判断是否超过阈值
        sort_result = sorted(similarities, reverse=True)
        if similarity_self <= sort_result[1800]:
            exceed_threshold_image_captions.append((image_relative_path, caption))
            # break  # 如果任何描述超过阈值，停止处理该图像的其他描述

# 打印和保存结果
for image, caption in exceed_threshold_image_captions:
    print(f"Image: {image}, Caption: '{caption}'")

output_file_path = "/xxx/output/exceed_threshold_image_captions_1800.txt"

with open(output_file_path, "w") as output_file:
    for image, caption in exceed_threshold_image_captions:
        output_file.write(f"Image: {image}, Caption: '{caption}'\n")