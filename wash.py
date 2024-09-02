import os
import json

# 定义保存路径
save_dir = './data/news_data/'
picture_dir = './data/pictures/'

# 输入要删除的索引序列（例如，输入“1,2,5-7”表示删除索引为1,2,5,6,7的文章和图片）
indexes_to_delete = input("请输入要删除的索引序列（例如1,2,5-7）：")


# 解析输入的索引序列
def parse_indexes(indexes_str):
    indexes = set()
    parts = indexes_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            indexes.update(range(int(start), int(end) + 1))
        else:
            indexes.add(int(part))
    return indexes


# 删除对应索引的文章和图片
def delete_articles_and_images(indexes):
    for filename in os.listdir(save_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(save_dir, filename)
            with open(file_path, 'r') as f:
                articles = json.load(f)

            # 保留不在删除索引中的文章
            updated_articles = [article for article in articles if article['index'] not in indexes]

            # 删除与索引对应的图片文件
            for article in articles:
                if article['index'] in indexes:
                    image_path = os.path.join(picture_dir, article['image'])
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"Deleted image: {image_path}")

            # 如果有剩余文章则更新文件，否则删除文件
            if updated_articles:
                with open(file_path, 'w') as f:
                    json.dump(updated_articles, f, indent=4)
                print(f"Updated file: {file_path}")
            else:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


# 运行删除操作
indexes = parse_indexes(indexes_to_delete)
delete_articles_and_images(indexes)

print("删除完成！")
