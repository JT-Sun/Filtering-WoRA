import json
import random
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


##################### step1 处理cuhk的caption
# 定义JSON文件路径
cuhk_json = "/xxx/data/finetune/cuhk_train.json"

cuhk_data_dict = {}
# i = 0
with open(cuhk_json, "r") as json_file:
    jason_dict = json.load(json_file)
    for item in jason_dict:
        image_path = item["image"]
        caption = item["caption"]
        cuhk_data_dict[image_path] = caption

cuhk_caption_list = list(cuhk_data_dict.values())

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
# 定义JSON文件路径
gen_json = "/xxx/data/finetune/gene_attrs/g_c_g_a_attrs.json"

gen_data_dict = {}
# i = 0
with open(gen_json, "r") as json_file:
    jason_dict = json.load(json_file)
    for item in jason_dict:
        image_path = item["image"]
        caption = item["caption"]
        gen_data_dict[image_path] = caption

gen_caption_list = list(gen_data_dict.values())


# result = []
image_name = []
for i in range(len(gen_caption_list)):
    ## step 1
    result = []
    raw_image = Image.open("/xxx/images/CUHK-PEDES/gene_crop/c_g_a/"+ str(i) +".jpg").convert("RGB")
    caption= gen_caption_list[i]

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    sample = {"image": image, "text_input": [text_input]}

    features_multimodal = model.extract_features(sample)
    # print(features_multimodal.multimodal_embeds.shape)
    # # torch.Size([1, 32, 768]), 32 is the number of queries

    features_image = model.extract_features(sample, mode="image")
    features_text = model.extract_features(sample, mode="text")
    similarity_self = (features_image.image_embeds_proj @ features_text.text_embeds_proj[:, 0, :].t()).max()
    similarity_self_value = similarity_self.item()
    result.append(float(similarity_self_value))

    ## step 2
    for z in range(0, 10000):

        similarity = (features_image.image_embeds_proj @ feature_cuhk_list[z].text_embeds_proj[:, 0, :].t()).max()
        similarity_value = similarity.item()
        result.append(float(similarity_value))


## step 3

    sort_result = sorted(result, reverse=True)
    if float(similarity_self_value) <= sort_result[49]:
        image_name.append(i)
# print("gene_crop/c_g_a/"+ str(image_name) +".jpg")
print(image_name)

output_file_path = "/xxx/output/cga.txt"

with open(output_file_path, "w") as output_file:
    for item in image_name:
        output_file.write(f"{item}\n")