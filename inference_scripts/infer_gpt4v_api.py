import os
import json
import base64
import openai
from tqdm import tqdm
from multiprocessing import Process
from PIL import Image
from io import BytesIO

# ---------------------- Global Configuration ----------------------
API_KEY = os.getenv("OPENAI_API_KEY")  # or replace with your key: "sk-..."
openai.api_key = API_KEY

MODEL_NAME = "gpt-4-vision-preview"
IMAGE_DIR = "/data1/vincia/data/SCENEs_400_Goal_and_Cand_Imgs_resized"
OUTPUT_DIR = "/data1/vincia/PIPELINE_Supplementary/Embodied-R1/Raw-Infer-GPT4V"
SYSTEM_PROMPT = "You are an expert in spatial reasoning and block construction tasks."

TOTAL_SAMPLES = 400
NUM_PROCESSES = 4


# ---------------------- Utility Functions ----------------------
def encode_image_to_base64(image_path):
    """Load image and encode to base64 for OpenAI API."""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def construct_messages(image_path):
    """Format messages as required by GPT-4V API."""
    base64_image = encode_image_to_base64(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {
                    "type": "text",
                    "text": "Please analyze the image and describe the detailed steps to assemble the block structure.",
                },
            ],
        },
    ]


def run_inference(index_range):
    for idx in tqdm(index_range, desc="GPT-4V"):
        image_path = os.path.join(IMAGE_DIR, f"{idx:03d}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")

        if os.path.exists(output_path):
            continue

        try:
            messages = construct_messages(image_path)

            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )

            output_text = response.choices[0].message["content"]

            json_output = {
                "image_path": image_path,
                "text": output_text,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[Error] Failed at index {idx:03d}: {e}")


def parallel_infer(num_processes=4, total_samples=400):
    """Run inference in parallel using multiprocessing (CPU only)."""
    samples_per_proc = total_samples // num_processes
    processes = []

    for i in range(num_processes):
        start = i * samples_per_proc
        end = total_samples if i == num_processes - 1 else (i + 1) * samples_per_proc
        p = Process(target=run_inference, args=(range(start, end),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parallel_infer(num_processes=NUM_PROCESSES, total_samples=TOTAL_SAMPLES)
