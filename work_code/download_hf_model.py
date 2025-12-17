from huggingface_hub import snapshot_download
import os

def download_hf_model(model_name: str, save_dir: str, token: str = None):
    """
    从 Hugging Face 下载模型到本地指定文件夹

    参数:
    - model_name: str, 模型名称，如 "meta-llama/Llama-3-8B-Instruct"
    - save_dir: str, 保存模型的本地路径，如 "/home/user/models/llama3"
    - token: str, 可选参数，用于访问私有模型的 Hugging Face token
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 开始下载模型: {model_name}")
    print(f"📁 保存路径: {save_dir}")

    # 调用 huggingface_hub 下载整个仓库（包括模型权重、配置、分词器等）
    snapshot_download(
        repo_id=model_name,
        local_dir=save_dir,
        token=token,  # 如果是公开模型可不填
        local_dir_use_symlinks=False  # 复制文件而不是符号链接
    )

    print("✅ 模型下载完成！")

# 示例使用
if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 模型名称（可替换）
    save_dir = "/Users/liyihan/coding/models/Qwen2.5-0.5B-Instruct"  # 本地存放路径
    # token = "hf_xxx"  # 如果需要访问私有模型，请在此处填写token
    download_hf_model(model_name, save_dir)
