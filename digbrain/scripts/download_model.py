#!/usr/bin/env python3
"""
模型下载脚本
下载Qwen3.5-0.8B模型
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(model_name: str, save_path: str, use_auth_token: str = None):
    """
    下载模型
    
    Args:
        model_name: 模型名称
        save_path: 保存路径
        use_auth_token: HuggingFace token
    """
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Save path: {save_path}")
        
        # 创建目录
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 下载模型
        logger.info("Downloading model files...")
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            token=use_auth_token
        )
        
        logger.info(f"Model downloaded to: {local_dir}")
        
        # 验证模型
        logger.info("Verifying model...")
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(save_path)
        
        logger.info("Model verification successful!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def download_qwen_model():
    """下载Qwen模型"""
    # 使用Qwen2.5-0.5B-Instruct（更小，更容易下载）
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    save_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
    
    return download_model(model_name, save_path)


def main():
    parser = argparse.ArgumentParser(description='Download Qwen Model')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model name on HuggingFace')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Save path')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace token')
    
    args = parser.parse_args()
    
    save_path = args.save_path or os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}"
    )
    
    success = download_model(args.model, save_path, args.token)
    
    if success:
        logger.info("Model download completed successfully!")
    else:
        logger.error("Model download failed!")
        exit(1)


if __name__ == '__main__':
    main()
