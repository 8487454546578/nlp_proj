import json

import os

# 更新后的输出路径
input_path = "./data/test.jsonl"
output_path = "./data/test_filter.jsonl"

# 读取并筛选数据：保留整行数据
filtered_data = []
if os.path.exists(input_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if data.get("is_response_0_safe") != data.get("is_response_1_safe"):
                filtered_data.append(data)

    # 写入筛选后的完整行数据
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    len(filtered_data)
else:
    len(filtered_data)  # 为 0，因为文件不存在


# 更新后的输出路径
input_path = "./data/train.jsonl"
output_path = "./data/train_filter.jsonl"

# 读取并筛选数据：保留整行数据
filtered_data = []
if os.path.exists(input_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            if data.get("is_response_0_safe") != data.get("is_response_1_safe"):
                filtered_data.append(data)

    # 写入筛选后的完整行数据
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    len(filtered_data)
else:
    len(filtered_data)  # 为 0，因为文件不存在
