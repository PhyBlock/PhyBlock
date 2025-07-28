"""
VLM Block Planner - 视觉语言模型积木规划器
"""
import os
import re
import json
import numpy as np
import torch
import tiktoken
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils import (
    read_txt, load_json, save_json, image_to_base64,
    extract_move_action, validate_response, calculate_token_count,
    create_image_message, create_text_message
)
from config import Config

# 提示词文件名常量
ACT_PROMPT = "env_action"
EXAMPLE_PROMPT = "examples_step"
SYSTEM_PROMPT = "system"
TASK_PROMPT = "task"
OUTPUT_PROMPT = "step_output"
RULE_PROMPT = "rules"

class VLMPlanner:
    """视觉语言模型积木规划器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.use_api = config.use_api
        self.prompt_dir = config.prompt_dir
        self.model_name = config.model_name
        self.max_token_length = config.max_token_length
        self.api_key = config.api_key
        self.url = config.url
        
        # 初始化客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
        
        # 状态变量
        self.history = []
        self.response = None
        self.full_response = None
        self.episode_memory = "Plans:\n"
        self.memory = []
        self.selected_index = None
        self.sorted_pairs = []
        
        # Token编码器
        self.encoding = tiktoken.encoding_for_model("gpt-4-0314")
        
        # 图像路径
        self.goal_image_path = []
        self.obj_image_path = ""
        
    def reset(self, goal_image_path: List[str], obj_image_path: str):
        """重置规划器状态"""
        self.history = []
        self.response = None
        self.episode_memory = "Plans:\n"
        self.goal_image_path = goal_image_path
        self.obj_image_path = obj_image_path
        self.selected_index = None
        self.sorted_pairs = []
        
    def calculate_token(self, messages: List[Dict[str, Any]]) -> int:
        """计算消息的token数量"""
        token_count = 0
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "text":
                        token_count += calculate_token_count(content["text"], self.encoding)
                    elif content["type"] == "image_url":
                        token_count += 300  # 估计图片token
            else:
                token_count += calculate_token_count(message["content"], self.encoding)
        return token_count
    
    def get_prompt(self, image_dict: List[Image.Image]) -> List[Dict[str, Any]]:
        """构建提示词"""
        # 读取提示词模板
        system_prompt = read_txt(os.path.join(self.prompt_dir, f"{SYSTEM_PROMPT}.txt"))
        example_prompt = read_txt(os.path.join(self.prompt_dir, f"{EXAMPLE_PROMPT}.txt"))
        act_prompt = read_txt(os.path.join(self.prompt_dir, f"{ACT_PROMPT}.txt"))
        output_prompt = read_txt(os.path.join(self.prompt_dir, f"{OUTPUT_PROMPT}.txt")).format(
            HISTORY="".join(self.history) if len(self.history) > 0 else ""
        )
        task_prompt = read_txt(os.path.join(self.prompt_dir, f"{TASK_PROMPT}.txt"))
        
        # 根据方法类型选择规则
        if 'cot' in self.config.method_type:
            rule_prompt = read_txt(os.path.join(self.prompt_dir, "cot_rules.txt"))
        else:
            rule_prompt = read_txt(os.path.join(self.prompt_dir, f"{RULE_PROMPT}.txt"))
        
        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                create_image_message(image_dict[1]),  # 目标图像
                create_image_message(image_dict[4]),  # 当前状态
                create_image_message(image_dict[3]),  # 对象图像
                create_text_message(output_prompt)
            ]}
        ]
        
        return messages
    
    def plan(self, obs: Dict[str, Any]) -> str:
        """生成规划动作"""
        image = obs["image"]
        
        if 'cot' in self.config.method_type and self.config.sample_num > 1:
            return self._plan_with_sampling(image)
        else:
            return self._plan_single(image)
    
    def _plan_single(self, image: np.ndarray) -> str:
        """单次规划"""
        # 准备图像
        img_dict = []
        for path in self.goal_image_path:
            img_dict.append(Image.open(path))
        img_dict.append(Image.fromarray(image))
        img_dict.append(Image.open(self.obj_image_path))
        
        # 构建消息
        self.messages = self.get_prompt(img_dict)
        
        # 检查token限制
        if self.calculate_token(self.messages) > self.max_token_length:
            self.history = self.history[2:]
            self.messages = self.get_prompt(img_dict)
        
        # 设置参数
        max_tokens = 400 if 'cot' in self.config.method_type else 25
        temperature = 0.7 if 'cot' in self.config.method_type else 0.05
        
        # 调用API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            full_output = response.choices[0].message.content
            
            # 处理输出
            if 'cot' in self.config.method_type:
                print(f"Full response: {full_output}")
                try:
                    output = full_output[full_output.index('['):].lower().strip()
                except:
                    output = full_output
            else:
                output = full_output
            
            # 提取动作
            match = re.search(r'Next plan:\s*(.*?)(?=\n|$)', output)
            if match:
                self.response = match.group(1).strip()
            else:
                self.response = output.strip()
            
            print(f"Action: {self.response}")
            self.full_response = full_output
            
            # 更新历史
            self.history.append(self.response + "\n")
            if 'memory' in self.config.method_type:
                self.episode_memory += self.response + '\n'
            
            return self.response
            
        except Exception as e:
            print(f"API调用错误: {e}")
            return ""
    
    def _plan_with_sampling(self, image: np.ndarray) -> str:
        """多次采样规划"""
        # 准备图像
        img_dict = []
        for path in self.goal_image_path:
            img_dict.append(Image.open(path))
        img_dict.append(Image.fromarray(image))
        img_dict.append(Image.open(self.obj_image_path))
        
        # 构建消息
        self.messages = self.get_prompt(img_dict)
        
        # 检查token限制
        if self.calculate_token(self.messages) > self.max_token_length:
            self.history = self.history[2:]
            self.messages = self.get_prompt(img_dict)
        
        # 多次采样
        rewards = []
        responses = []
        full_responses = []
        
        for i in range(self.config.sample_num):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    max_tokens=400,
                    temperature=0.7,
                    top_p=0.9
                )
                
                full_output = response.choices[0].message.content
                print(f"Sample {i+1}: {full_output}")
                
                try:
                    output = full_output[full_output.index('['):].lower().strip()
                except:
                    output = full_output
                
                match = re.search(r'Next plan:\s*(.*?)(?=\n|$)', output)
                if match:
                    response_text = match.group(1).strip()
                else:
                    response_text = output.strip()
                
                # 评估响应
                reward, _ = self._evaluate_response(img_dict, response_text, full_output)
                
                rewards.append(reward)
                responses.append(response_text)
                full_responses.append(full_output)
                
            except Exception as e:
                print(f"Sample {i+1} failed: {e}")
                rewards.append(0.0)
                responses.append("")
                full_responses.append("")
        
        # 排序并选择最佳响应
        self.sorted_pairs = sorted(zip(rewards, responses, full_responses), reverse=True)
        
        if self.config.majority_vote:
            responses_counter = Counter(responses)
            self.response = responses_counter.most_common()[0][0]
            self.selected_index = next(i for i, (_, resp, _) in enumerate(self.sorted_pairs) if resp == self.response)
        else:
            self.response = self.sorted_pairs[0][1]
        
        # 更新历史
        self.history.append(self.response + "\n")
        if 'memory' in self.config.method_type:
            self.episode_memory += self.response + '\n'
        
        return self.response
    
    def _evaluate_response(self, image_dict: List[Image.Image], response: str, full_response: str) -> Tuple[float, str]:
        """评估响应质量"""
        # 这里可以实现更复杂的评估逻辑
        # 目前使用简单的启发式评估
        score = 0.0
        
        # 检查响应格式
        if validate_response(response):
            score += 1.0
        
        # 检查是否包含有效动作
        if extract_move_action(response):
            score += 2.0
        
        # 检查响应长度
        if len(response) > 10:
            score += 0.5
        
        return score, full_response
    
    def update_history(self, success: bool):
        """更新执行历史"""
        if self.history:
            status = "**Execution successful**" if success else "**Execution failed**"
            self.history[-1] = self.history[-1].strip() + f" {status}\n"
    
    def add_memory(self, image: np.ndarray):
        """添加记忆"""
        img_dict = []
        for path in self.goal_image_path:
            img_dict.append(Image.open(path))
        img_dict.append(Image.fromarray(image))
        img_dict.append(Image.open(self.obj_image_path))
        
        score, _ = self._evaluate_response(img_dict, self.response, self.full_response)
        
        if score >= self.config.memory_thresh:
            self.memory.append({
                'instruction': self.goal_image_path[1],
                'candidate': self.obj_image_path,
                'plans': self.episode_memory
            })
    
    def save_memory(self, file_path: str):
        """保存记忆"""
        save_json(self.memory, file_path)
    
    def load_memory(self, file_path: str):
        """加载记忆"""
        self.memory = load_json(file_path) 