import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Process
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------------- Path Config ----------------------
"""
# Qwen-Type-Models:
# MODEL_DIR = "/data1/vincia/models/VeBrain"
# MODEL_DIR = "/data1/vincia/models/SpaceR"
# MODEL_DIR = "/data1/vincia/models/Embodied-R1-3B-Stage1"
# MODEL_DIR = "/data1/vincia/models/Cosmos-Reason1-7B"
"""
# MODEL_DIR = "/data1/vincia/models/Embodied-R1-3B-Stage1"
MODEL_DIR = r"/data1/vincia/models/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT_PATH = r"/data1/vincia/PIPELINE_FULL_DPO/system_prompt.txt"
IMAGES_DIR = r"/data1/vincia/data/SCENEs_400_Goal_and_Cand_Imgs_resized"
OUTPUT_DIR = r"/data1/vincia/Supplementary/Embodied-R1/Raw-Infer-400"

# ---------------------- Helpers ----------------------
def read_system_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def run_inference_on_gpu(gpu_index, index_range, cuda_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ids[gpu_index])
    torch.cuda.set_device(0)  # 映射到本地第一块卡（即实际 cuda_ids[gpu_index]）

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype="auto", device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
        padding_side="right",
    )

    system_prompt = read_system_prompt()

    for idx in tqdm(index_range, desc=f"[GPU {cuda_ids[gpu_index]}]"):
        image_path = os.path.join(IMAGES_DIR, f"{idx:03d}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")
        if os.path.exists(output_path):
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": "Please analyze the above picture and output the detailed steps for assembling the building block structure.",
                    },
                ],
            },
        ]

        try:
            # 构建输入
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # 生成输出
            generated_ids = model.generate(**inputs, max_new_tokens=768)
            output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            # 保存结果
            json.dump({"image_path": image_path, "text": output_text},
                      open(output_path, "w", encoding="utf-8"),
                      indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[GPU {cuda_ids[gpu_index]}] Error processing {idx:03d}.png: {e}")

def parallel_infer(cuda_ids, total_samples=400):
    num_gpus = len(cuda_ids)
    per_gpu = total_samples // num_gpus
    processes = []

    for i in range(num_gpus):
        start = i * per_gpu
        end = total_samples if i == num_gpus - 1 else (i + 1) * per_gpu
        p = Process(target=run_inference_on_gpu, args=(i, range(start, end), cuda_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cuda_ids = [5, 6, 7]
    parallel_infer(cuda_ids=cuda_ids, total_samples=400)