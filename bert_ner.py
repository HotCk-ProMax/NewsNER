import os
import json
import random
from transformers import pipeline

# 初始化指定的NER模型和分词器
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",  # 使用简单聚合策略，合并连续标记
    device=0
)

# 定义新闻和NER结果的主目录路径
classified_dir = './data/classified_data/'
ner_dir = './data/ner_data/'
os.makedirs(ner_dir, exist_ok=True)  # 确保NER结果主目录存在

# 设置拆分比例
dev_ratio = 0.1
test_ratio = 0.2
train_ratio = 1 - dev_ratio - test_ratio


# 合并子词并生成BIO格式标签
def generate_bio_tags(tokens, entities):
    bio_tags = ["O"] * len(tokens)  # 初始化所有标记为 "O"

    token_idx = 0  # 用于跟踪tokens中的位置
    for entity in entities:
        entity_tokens = entity['word'].split()  # 获取实体的子词
        label = entity['entity_group']

        # 找到实体的起始位置
        while token_idx < len(tokens) and not tokens[token_idx].startswith(entity_tokens[0]):
            token_idx += 1

        if token_idx < len(tokens):  # 确保索引不超出范围
            bio_tags[token_idx] = f"B-{label}"  # 实体开头标记为 "B-"
            for i in range(1, len(entity_tokens)):
                if token_idx + i < len(tokens):
                    bio_tags[token_idx + i] = f"I-{label}"  # 实体内部标记为 "I-"
            token_idx += len(entity_tokens)  # 更新索引

    return bio_tags


# 遍历分类后的文件夹
for category in os.listdir(classified_dir):
    category_path = os.path.join(classified_dir, category)

    if os.path.isdir(category_path):
        ner_category_dir = os.path.join(ner_dir, category)
        os.makedirs(ner_category_dir, exist_ok=True)

        all_data = []

        news_files = sorted(os.listdir(category_path),
                            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

        for news_file in news_files:
            news_path = os.path.join(category_path, news_file)
            with open(news_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)

            text = news_data['text']
            entities = ner_pipeline(text)

            # 获取原始文本的token
            tokens = text.split()

            # 生成BIO标签，包括O标签
            bio_tags = generate_bio_tags(tokens, entities)

            # 将结果存入all_data列表
            indexed_result = [f"IMGID:{os.path.splitext(news_file)[0]}"]  # 添加索引行
            for token, label in zip(tokens, bio_tags):
                indexed_result.append(f"{token}\t{label}")
            indexed_result.append("")  # 在每篇新闻的结尾添加一个空行，符合CoNLL格式
            all_data.append("\n".join(indexed_result))

        # 打乱数据集
        random.shuffle(all_data)

        # 计算每个集合的大小
        total = len(all_data)
        dev_size = int(total * dev_ratio)
        test_size = int(total * test_ratio)
        train_size = total - dev_size - test_size

        # 拆分数据
        dev_data = all_data[:dev_size]
        test_data = all_data[dev_size:dev_size + test_size]
        train_data = all_data[dev_size + test_size:]

        # 保存到对应的文件
        with open(os.path.join(ner_category_dir, "dev.txt"), 'w', encoding='utf-8') as dev_file:
            dev_file.write("\n\n".join(dev_data))

        with open(os.path.join(ner_category_dir, "test.txt"), 'w', encoding='utf-8') as test_file:
            test_file.write("\n\n".join(test_data))

        with open(os.path.join(ner_category_dir, "train.txt"), 'w', encoding='utf-8') as train_file:
            train_file.write("\n\n".join(train_data))

        print(f"Processed category {category} and saved NER results to dev.txt, test.txt, and train.txt.")

print("NER processing completed for all news files.")
