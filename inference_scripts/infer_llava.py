import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Process
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info  # assuming shared image/video processing logic

# ---------------------- Global Configuration ----------------------
MODEL_DIR = "/data1/vincia/models/LLaVA-7B-FT"
SYSTEM_PROMPT_PATH = "/data1/vincia/PIPELINE_FULL_DPO/system_prompt.txt"
IMAGES_DIR = "/data1/vincia/data/SCENEs_400_Goal_and_Cand_Imgs_resized"
OUTPUT_DIR = "/data1/vincia/PIPELINE_Supplementary/Embodied-R1/Raw-Infer-LLaVA"

# ---------------------- Helper Functions ----------------------
def read_system_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def run_inference_on_gpu(gpu_index, index_range, cuda_ids):
    """Run inference on a single GPU over a subset of image indices."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ids[gpu_index])
    torch.cuda.set_device(0)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    system_prompt = read_system_prompt()

    for idx in tqdm(index_range, desc=f"[GPU {cuda_ids[gpu_index]}]"):
        image_path = os.path.join(IMAGES_DIR, f"{idx:03d}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")

        if os.path.exists(output_path):
            continue  # Skip if already processed

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": "Please analyze the image and describe the steps to assemble the block structure."
                    },
                ],
            },
        ]

        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=768)
            output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            json_output = {
                "image_path": image_path,
                "text": output_text
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[GPU {cuda_ids[gpu_index]}] Error processing {idx:03d}.png: {e}")

def parallel_infer(cuda_ids, total_samples=400):
    """Launch multiple processes for parallel inference across GPUs."""
    num_gpus = len(cuda_ids)
    samples_per_gpu = total_samples // num_gpus
    processes = []

    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = total_samples if i == num_gpus - 1 else (i + 1) * samples_per_gpu
        p = Process(target=run_inference_on_gpu, args=(i, range(start, end), cuda_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# ---------------------- Main Entry ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cuda_ids = [5, 6, 7]  # Change as needed
    parallel_infer(cuda_ids=cuda_ids, total_samples=400)