import os
import json
from transformers import BertTokenizer


def process_json_files(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 初始化BERT英文分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    index = 0
    for root, dirs, files in os.walk(input_folder_path):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 使用BERT分词器进行分词
                words = tokenizer.tokenize(data['text'])

                # 将分词结果存储到 _s.txt 文件
                s_file_name = os.path.join(output_folder_path, f"{index}_s.txt")
                with open(s_file_name, 'w', encoding='utf-8') as s_file:
                    s_file.write("\t".join(words))

                # 将图片 ID 存储到 _p.txt 文件
                p_file_name = os.path.join(output_folder_path, f"{index}_p.txt")
                with open(p_file_name, 'w', encoding='utf-8') as p_file:
                    p_file.write(f"{data['index']}")

                # 将标签 O 存储到 _l.txt 文件
                l_file_name = os.path.join(output_folder_path, f"{index}_l.txt")
                with open(l_file_name, 'w', encoding='utf-8') as l_file:
                    l_file.write("\t".join(['O'] * len(words)))

                # 更新索引
                index += 1


# 指定输入文件夹路径和输出文件夹路径
input_folder_path = './data/classified_data'
output_folder_path = './data/to_test/test'
process_json_files(input_folder_path, output_folder_path)
