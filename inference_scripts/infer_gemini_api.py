import os
import json
from tqdm import tqdm
from multiprocessing import Process
from PIL import Image
import google.generativeai as genai

# Gemini configuration
GOOGLE_API_KEY = "your-api-key"  # Replace with your actual Google API Key
MODEL_NAME = "models/gemini-pro-vision"
genai.configure(api_key=GOOGLE_API_KEY)

# Path configuration
SYSTEM_PROMPT_PATH = "/data1/vincia/PIPELINE_FULL_DPO/system_prompt.txt"
IMAGES_DIR = "/data1/vincia/data/SCENEs_400_Goal_and_Cand_Imgs_resized"
OUTPUT_DIR = "/data1/vincia/PIPELINE_Supplementary/Gemini-2.5-Pro/Raw-Infer-400"

def read_system_prompt():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def run_inference(index_range):
    system_prompt = read_system_prompt()
    model = genai.GenerativeModel(MODEL_NAME)

    for idx in tqdm(index_range, desc="Gemini Inference"):
        image_path = os.path.join(IMAGES_DIR, f"{idx:03d}.png")
        output_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.json")

        if os.path.exists(output_path):
            continue  # Skip if already processed

        try:
            image = load_image(image_path)
            response = model.generate_content(
                [
                    system_prompt,
                    image,
                    "Please analyze the above picture and output the detailed steps for assembling the building block structure."
                ],
                generation_config={"max_output_tokens": 4096},
            )
            output_text = response.text

            if len(output_text.strip()) > 0:
                json_data = {"image_path": image_path, "text": output_text}
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[Gemini] Error processing image {idx:03d}.png: {e}")

def parallel_infer(num_procs=4, total_samples=400):
    per_proc = total_samples // num_procs
    processes = []

    for i in range(num_procs):
        start = i * per_proc
        end = total_samples if i == num_procs - 1 else (i + 1) * per_proc
        p = Process(target=run_inference, args=(range(start, end),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def single_image_debug():
    system_prompt = read_system_prompt()
    model = genai.GenerativeModel(MODEL_NAME)
    image_path = os.path.join(IMAGES_DIR, "000.png")
    image = load_image(image_path)

    response = model.generate_content(
        [
            system_prompt,
            image,
            "Please analyze the above picture and output the detailed steps for assembling the building block structure."
        ],
        generation_config={"max_output_tokens": 4096},
    )

    print(response.text)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Uncomment one of the following as needed:
    # parallel_infer(num_procs=4)
    # single_image_debug()
