import subprocess
import os
import sys

# 定义要依次运行的脚本路径列表
scripts = [
    "spider.py",
    "classify.py",
    "bert_ner.py",
    "./img_detect/my_grounding.py",
    "check.py",
    # 添加更多脚本路径
]

def run_script(script_path):
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: Script {script_path} failed with return code {e.returncode}")
        return False

def main():
    for script in scripts:
        script_path = os.path.abspath(script)
        print(f"Running {script_path}...")
        if not run_script(script_path):
            print("Stopping execution due to script failure.")
            break
        print(f"{script_path} completed successfully.")

if __name__ == "__main__":
    main()
