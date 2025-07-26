import os
import sys
import json
import torch
from tqdm import tqdm
from multiprocessing import Process, set_start_method

# Set CUDA devices globally before spawning processes
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"

# Add model path to sys.path for import
sys.path.append("/data1/vincia/models/RoboBrain2.0")

from qwen_vl_utils import process_vision_info
from inference import SimpleInference


"""
# RoboBrain-Type-Models:
# MODEL_DIR = "/data1/vincia/models/RoboBrain"
# MODEL_DIR = "/data1/vincia/models/RoboBrain2.0-7B"
# MODEL_DIR = "/data1/vincia/models/RoboBrain2.0-32B"
"""



# === Path configuration === #
MODEL_DIR = "/data1/vincia/models/RoboBrain2.0-32B"
SYSTEM_PROMPT_PATH = "/data1/vincia/PIPELINE_FULL_DPO/system_prompt.txt"
IMAGES_DIR = "/data1/vincia/data/SCENEs_400_Goal_and_Cand_Imgs_resized"
OUTPUT_DIR = "/data1/vincia/PIPELINE_Supplementary/RoboBrain32B/Raw-Infer-400"


def load_system_prompt():
    """Load system-level prompt for inference."""
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def run_inference_on_gpu(gpu_id: int, index_range: range):
    """
    Perform inference on a specified GPU for a given range of image indices.

    Args:
        gpu_id (int): GPU index to use.
        index_range (range): Range of image indices to process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    model = SimpleInference(MODEL_DIR)
    system_prompt = load_system_prompt()

    for idx in tqdm(index_range, desc=f"GPU {gpu_id}"):
        image_path = os.path.join(IMAGES_DIR, f"{idx:03d}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")

        if os.path.exists(output_path):
            continue  # Skip if already processed

        try:
            prediction = model.inference(
                system_prompt,
                images=[image_path],
                task="general",
                enable_thinking=True,
                do_sample=True
            )

            result = {
                "image_path": image_path,
                "text": f"{prediction['thinking']}  {prediction['answer']}"
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing image {idx:03d}.png: {e}")


def parallel_infer(num_gpus: int = 3, total_samples: int = 400):
    """
    Launch parallel inference across multiple GPUs.

    Args:
        num_gpus (int): Number of GPUs to use.
        total_samples (int): Total number of samples to process.
    """
    samples_per_gpu = total_samples // num_gpus
    processes = []

    for i in range(num_gpus):
        start = i * samples_per_gpu
        end = total_samples if i == num_gpus - 1 else (i + 1) * samples_per_gpu
        gpu_id = i + 4  # Adjust GPU offset if needed
        process = Process(target=run_inference_on_gpu, args=(gpu_id, range(start, end)))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parallel_infer(num_gpus=1)
