# 📦 Vision-Language Model Inference Scripts

This directory contains inference pipelines for various Vision-Language Models (VLMs), including Qwen2.5-VL, LLaVA, GPT-4V, and others. Each script is designed to produce structured JSON outputs for downstream evaluation.

## ✅ Available Scripts

| Script                | Model          | Description                                           |
|-----------------------|----------------|-------------------------------------------------------|
| `infer_gemini_api.py` | Gemini 2.5 Pro | Image-conditioned building block assembly inference using Google Gemini API |
| `infer_gpt4v_api.py`  | GPT-4V         | Image-conditioned building block assembly inference via OpenAI GPT-4 Vision API |
| `infer_llava.py`      | LLaVA          | Vision-language building block assembly generation with LLaVA model |
| `infer_qwen2_5_vl.py` | Qwen2.5-VL     | Vision-language building block assembly generation using Qwen2.5-VL model |
| `infer_robobrain.py`  | RoboBrain 32B  | Large-scale vision-language building block assembly inference with RoboBrain model |
| `infer_spatialvlm.py` | SpatialVLM     | Vision-language building block assembly generation with SpatialVLM model |


## 📁 Output Format

Each script generates JSON files like:

```json
{
  "image_path": "xxx.png",
  "text": "The predicted assembly steps are..."
}