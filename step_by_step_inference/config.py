"""
VLM Block Builder 配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """配置类，包含所有系统参数"""
    
    # API配置
    api_key: str = "your-api-key-here"
    url: str = "https://api.openai.com/v1"
    use_api: bool = True
    
    # 模型配置
    model_name: str = "gpt-4-vision-preview"
    max_token_length: int = 8192
    
    # 环境配置
    max_steps: int = 20
    cube_num: int = 5
    block_path: str = "./data/blocks"
    
    # 数据路径
    data_path: str = "./data/scenes"
    results_dir: str = "./results"
    save_img_dir: str = "./block_imgs"
    save_img: bool = True
    
    # 提示词路径
    prompt_dir: str = "./prompts/block_build"
    
    # 规划方法配置
    method_type: str = "step_memory_cot_multimodalscore"
    memory_thresh: float = 4.0
    load_memory: bool = False
    majority_vote: bool = True
    sample_num: int = 3
    
    # 评估配置
    euler_constraint: bool = False
    max_len: int = 0  # 0表示处理所有数据
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.save_img_dir, exist_ok=True)
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.block_path, exist_ok=True)

# 默认配置实例
default_config = Config()

def get_config_from_args(args):
    """从命令行参数创建配置"""
    config = Config()
    
    # 更新配置
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config 