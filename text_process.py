import os

# 定义输入数据的根目录和输出目录
input_root_dir = './data/ner_data'
output_root_dir = './data/ner_txt'

# 创建输出根目录
os.makedirs(output_root_dir, exist_ok=True)

# 定义要处理的子目录名称
subdirs = ['business', 'entertainment', 'politics', 'sport', 'tech']

# 处理每个子目录
for subdir in subdirs:
    input_dir = os.path.join(input_root_dir, subdir)
    subdir_output_root = os.path.join(output_root_dir, subdir)
    os.makedirs(subdir_output_root, exist_ok=True)

    # 文件名列表
    file_types = ['dev', 'test', 'train']

    for file_type in file_types:
        input_file_path = os.path.join(input_dir, f'{file_type}.txt')

        # 输出文件夹名称
        if file_type == 'dev':
            output_file_type = 'valid'  # 将 dev 重命名为 valid
        else:
            output_file_type = file_type

        output_file_dir = os.path.join(subdir_output_root, output_file_type)
        os.makedirs(output_file_dir, exist_ok=True)

        current_imgid = None
        file_counter = 0  # 文件序号计数器
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            s_lines, l_lines = [], []
            for line in input_file:
                if line.startswith('IMGID'):
                    # 如果已经有内容，先保存之前的文件
                    if current_imgid is not None:
                        s_file_path = os.path.join(output_file_dir, f'{file_counter}_s.txt')
                        l_file_path = os.path.join(output_file_dir, f'{file_counter}_l.txt')
                        p_file_path = os.path.join(output_file_dir, f'{file_counter}_p.txt')

                        # 保存当前文件
                        with open(s_file_path, 'w', encoding='utf-8') as s_file:
                            s_file.write(''.join(s_lines))
                        with open(l_file_path, 'w', encoding='utf-8') as l_file:
                            l_file.write(''.join(l_lines))
                        with open(p_file_path, 'w', encoding='utf-8') as p_file:
                            p_file.write(f"{current_imgid}\n")

                        # 增加文件序号
                        file_counter += 1

                    # 读取新 IMGID
                    current_imgid = line.strip().split(':')[1]
                    s_lines, l_lines = [], []

                elif line.strip():  # 非空行
                    word, label = line.strip().split('\t')
                    # 将 MISC 标签替换为 OTHER
                    label = label.replace('MISC', 'OTHER')
                    s_lines.append(f"{word}\t")
                    l_lines.append(f"{label}\t")

            # 保存最后一个 IMGID 的内容
            if current_imgid is not None:
                s_file_path = os.path.join(output_file_dir, f'{file_counter}_s.txt')
                l_file_path = os.path.join(output_file_dir, f'{file_counter}_l.txt')
                p_file_path = os.path.join(output_file_dir, f'{file_counter}_p.txt')

                with open(s_file_path, 'w', encoding='utf-8') as s_file:
                    s_file.write(''.join(s_lines))
                with open(l_file_path, 'w', encoding='utf-8') as l_file:
                    l_file.write(''.join(l_lines))
                with open(p_file_path, 'w', encoding='utf-8') as p_file:
                    p_file.write(f"{current_imgid}\n")

        print(f"Processed {file_type}.txt for {subdir} and saved in {output_file_dir}")

print("All files processed successfully!")
