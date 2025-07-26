import os
import io
import json
import torch
import base64
from PIL import Image
from tqdm import tqdm
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from multiprocessing import Process

# Global path configuration
SYSTEM_PROMPT_PATH = "/data1/vincia/PIPELINE_FULL_DPO/system_prompt.txt"
GOAL_IMAGES_DIR = "/data1/vincia/data/SCENEs_400_Goal_Imgs"
CAND_IMAGES_DIR = "/data1/vincia/data/SCENEs_400_Cand_Imgs"
OUTPUT_DIR = "/data1/vincia/PIPELINE_Supplementary/SpatialVLM/4096-224/Raw-Infer-400"
MODEL_PATH = "/data1/vincia/SpatialVLM/SpaceLLaVA/SpaceLLaVA/ggml-model-q4_0.gguf"
CLIP_PATH = "/data1/vincia/SpatialVLM/SpaceLLaVA/SpaceLLaVA/mmproj-model-f16.gguf"


def image_to_base64_data_uri(image_input):
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("Unsupported input type. Must be file path or PIL.Image.Image.")
    return f"data:image/png;base64,{base64_data}"


def image512_to_base64_data_uri(image_input, target_size=(224, 224)):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported input type. Must be file path or PIL.Image.Image.")
    image = image.resize(target_size, Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_data}"


def read_system_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def run_inference_on_gpu(gpu_id, index_range):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    chat_handler = Llava15ChatHandler(clip_model_path=CLIP_PATH, verbose=True)
    model = Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_ctx=4096,
        logits_all=True,
        n_gpu_layers=-1
    )

    system_prompt = read_system_prompt()

    for idx in tqdm(index_range, desc=f"GPU {gpu_id}"):
        goal_image_path = os.path.join(GOAL_IMAGES_DIR, f"{idx:03d}_1.png")
        cand_image_path = os.path.join(CAND_IMAGES_DIR, f"{idx:03d}_16_cand.png")

        goal_img_uri = image512_to_base64_data_uri(goal_image_path)
        cand_img_uri = image512_to_base64_data_uri(cand_image_path)

        messages = [
            {
                "role": "system",
                "content": "You are an expert in visual assembly tasks. Given a target structure and a dictionary of building blocks in fixed orientations, you analyze the structure, identify required blocks, and provide a step-by-step assembly sequence using only the available blocks without rotation."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": goal_img_uri}},
                    {"type": "image_url", "image_url": {"url": cand_img_uri}},
                    {"type": "text", "text": system_prompt}
                ]
            }
        ]

        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")
        if os.path.exists(output_path):
            continue

        try:
            result = model.create_chat_completion(messages=messages, max_tokens=2048)
            json_data = {
                "image_path": goal_image_path,
                "text": result["choices"][0]["message"]["content"]
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {idx:03d}: {e}")


def parallel_infer(num_gpus=4, total_samples=400):
    samples_per_gpu = total_samples // num_gpus
    processes = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        end_idx = total_samples if i == num_gpus - 1 else (i + 1) * samples_per_gpu
        process = Process(target=run_inference_on_gpu, args=(i + 4, range(start_idx, end_idx)))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parallel_infer(num_gpus=4)
