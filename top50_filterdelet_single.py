import os
import json

# 读取txt文件内容并转为list(can change for 4x or cga)
input_file_path = "/xxx/output/iga.txt"

image_names_to_delete = []
with open(input_file_path, "r") as input_file:
    for line in input_file:
        # 去除换行符并将内容添加到列表中
        image_names_to_delete.append(int(line.strip()))

# 图片文件夹路径
image_folder_path = "/xxx/images/CUHK-PEDES/gene_crop/i_g_a/"

# 逐一删除图片文件夹内对应名称的图片
for name in image_names_to_delete:
    image_path = os.path.join(image_folder_path, f"{name}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted: {image_path}")
    else:
        print(f"File not found: {image_path}")

#删除JSON文件内对应名称的image_name和caption
json_file_path = "/xxx/data/finetune/gene_attrs/g_i_g_a_attrs.json"


# 假设要删除的索引是 index_list_to_delete
index_list_to_delete = image_names_to_delete  # 用你想要删除的具体索引值列表替换

with open(json_file_path, "r") as json_file:
    json_data = json.load(json_file)   # list

# 删除指定索引位置的数据
indices_to_keep = [index for index, _ in enumerate(json_data) if index not in index_list_to_delete]
filtered_json_data = [json_data[index] for index in indices_to_keep]

# 保存更新后的JSON文件
with open(json_file_path, "w") as json_file:
    json.dump(filtered_json_data, json_file, indent=4)

print("Updated JSON file.")

