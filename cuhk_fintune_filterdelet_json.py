import json


# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 读取TXT文件，并转换为适合的格式
def read_txt_file(file_path):
    text_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(", Caption: '")
            if len(parts) == 2:
                image = parts[0].split("Image: ")[1]
                caption = parts[1].rstrip("'")
                text_data.append({"image": image, "caption": caption})
    return text_data


# 删除JSON数据中匹配的项
def remove_matching_entries(json_data, text_data):
    # 将text_data转换为一个包含(image, caption)元组的集合，用于快速比较
    text_data_set = {(item['image'], item['caption']) for item in text_data}

    # 过滤掉那些在text_data_set中找到匹配的image、caption的JSON项
    filtered_json_data = [item for item in json_data if (item['image'], item['caption']) not in text_data_set]

    return filtered_json_data


# 将更新后的JSON数据保存到新文件中
def save_to_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# 定义文件路径
json_file_path = '/xxx/data/finetune/cuhk_train.json'  # JSON文件路径
txt_file_path = '/xxx/output/exceed_threshold_image_captions_1800.txt'  # TXT文件路径
new_json_file_path = '/xxx/data/finetune/cuhk_train_filter_90percent.json'  # 更新后的JSON文件路径

# 执行步骤
json_data = read_json_file(json_file_path)  # 读取JSON文件
text_data = read_txt_file(txt_file_path)  # 读取TXT文件
updated_json_data = remove_matching_entries(json_data, text_data)  # 删除匹配的项
save_to_json(new_json_file_path, updated_json_data)  # 保存更新后的JSON数据

print("已完成更新并保存新的JSON文件。")
