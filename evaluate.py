import os
import json
import re
import csv
import glob
import argparse
from pathlib import Path



def extract_message_content(raw_data):
    try:
        # OpenAI GPT
        return raw_data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        pass

    try:
        # Claude normal
        return raw_data['content'][0]['text']
    except (KeyError, IndexError, TypeError):
        pass

    try:
        # Claude thinking
        return raw_data['content'][1]['text']
    except (KeyError, IndexError, TypeError):
        pass

    try:
        # Gemini
        return raw_data['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError, TypeError):
        pass

    try:
        # InternVL
        return raw_data['predict']
    except (KeyError, TypeError):
        pass

    try:
        # Qwen-VL-Max
        return raw_data['text']
    except (KeyError, TypeError):
        pass

    return ""



def extract_blocks_sequence(raw_outputs_json_dir, matching_json_dir, extracted_json_dir):
    """
    Extract block sequences from raw model output and map them to full block descriptions.

    Args:
        raw_outputs_json_dir (str): Directory containing raw model output JSONs.
        matching_json_dir (str): Directory containing candidate block JSONs.
        extracted_json_dir (str): Directory to save the extracted block sequence JSONs.
    """
    os.makedirs(extracted_json_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(raw_outputs_json_dir, "*.json")))
    total_files = len(raw_files)

    for file_idx, json_path in enumerate(raw_files):
        scene_name = os.path.basename(json_path)[:3]

        # Load raw model output
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            message_content = extract_message_content(raw_data)
            pattern = r'block with index (\d+)'
            blocks_indexes = re.findall(pattern, message_content)

        # Load candidate block information
        match_filename = f"{scene_name}_16_cand_blocks.json"
        match_path = os.path.join(matching_json_dir, match_filename)
        with open(match_path, 'r') as f:
            match_data = json.load(f)
            max_index = len(match_data)

            # Filter invalid indices and retrieve corresponding blocks
            blocks_indexes = [int(i) for i in blocks_indexes if 0 < int(i) <= max_index]
            blocks_sequence = [match_data[i - 1] for i in blocks_indexes]

        # Save the extracted block sequence
        save_data = {
            "shape_name": scene_name,
            "blocks": blocks_sequence
        }
        save_path = os.path.join(extracted_json_dir, f"{scene_name}_blocks_sequence.json")
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=4)

        print(f"[Extracted]: {scene_name} [{file_idx + 1}/{total_files}]")


def are_blocks_equal(block1, block2):
    """
    Check if two blocks are equal based on specific keys ('type', 'color', 'euler').

    Args:
        block1 (dict): The first block to compare.
        block2 (dict): The second block to compare.

    Returns:
        bool: True if the blocks are equal based on the selected keys, False otherwise.
    """
    comparison_keys = ['type', 'color', 'euler']
    return all(block1[key] == block2[key] for key in comparison_keys)


def is_place_legal(block, all_blocks):
    """
    Check if the placement of the given block is legal based on its dependencies.

    Args:
        block (dict): The block being checked for placement legality.
        all_blocks (list): A list of all blocks, where each block is a dictionary.

    Returns:
        bool: True if the block can be legally placed (i.e., all its dependencies are placed), False otherwise.
    """
    dependency_indices = block.get('depend', [])
    
    if dependency_indices == [0]:
        return True
    
    return all(all_blocks[i-1].get('placed', False) for i in dependency_indices)


def validate_block_placement(gt_path, pred_path):
    """
    Compare predicted block sequence with ground truth block sequence.

    Args:
        gt_path (str): Path to the ground truth JSON file.
        pred_path (str): Path to the predicted sequence JSON file.

    Returns:
        tuple: (level, TP, FP, FN, precision, recall, F1)
    """
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    gt_blocks = [
        {**b, 'placed': False}
        for b in ground_truth.get('blocks', [])
    ]
    gt_level = ground_truth.get('level', 0)

    with open(pred_path, 'r') as f:
        predicted = json.load(f)
    pred_blocks = [
        {**b, 'flag': 0}
        for b in predicted.get('blocks', [])
    ]

    # Match predicted blocks to ground truth blocks
    for pred_block in pred_blocks:
        for gt_block in gt_blocks:
            if not gt_block['placed'] and \
               is_place_legal(gt_block, gt_blocks) and \
               are_blocks_equal(pred_block, gt_block):
                gt_block['placed'] = True
                pred_block['flag'] = gt_block['order']
                break

    TP = sum(1 for b in pred_blocks if b['flag'] != 0)
    FP = sum(1 for b in pred_blocks if b['flag'] == 0)
    FN = len(gt_blocks) - TP

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return gt_level, TP, FP, FN, precision, recall, F1


def validate_predicted_scenes(gt_dir, pred_dir, result_csv_path):
    """
    Evaluate all predicted block sequences against ground truth and save metrics to CSV.

    Args:
        gt_dir (str): Directory containing ground truth JSONs.
        pred_dir (str): Directory containing predicted sequence JSONs.
        result_csv_path (str): Path to save the result CSV.
    """
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    header = ['Scene', 'Level', 'Blocks_num', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score']

    with open(result_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))
        for idx, pred_path in enumerate(pred_files):
            scene_name = os.path.basename(pred_path)[:3]
            gt_path = os.path.join(gt_dir, f"{scene_name}.json")

            if not os.path.exists(gt_path):
                print(f"[Warning]: Ground truth for {scene_name} not found.")
                continue

            level, TP, FP, FN, precision, recall, F1 = validate_block_placement(gt_path, pred_path)
            result = [
                scene_name, level, TP + FN,
                TP, FP, FN,
                round(precision, 3),
                round(recall, 3),
                round(F1, 3)
            ]
            writer.writerow(result)
            print(f"[Validated]: {scene_name} [{idx + 1}/{len(pred_files)}]")


def main():
    parser = argparse.ArgumentParser(description="Extract predicted blocks and evaluate them against ground truth.")
    parser.add_argument("--raw_outputs_json_dir", default=r'', 
                        help="Directory with raw model output JSONs.")
    parser.add_argument("--matching_json_dir", default=r'./data/SCENEs_400_Cand_Imgs', \
                        help="Directory with candidate block JSONs.")
    parser.add_argument("--extracted_json_dir", default=r'', \
                        help="Directory to save extracted block sequences.")
    parser.add_argument("--groundtruth_json_dir", default=r'./data/SCENEs_400_Goal_Jsons', \
                        help="Directory with ground truth goal JSONs.")
    parser.add_argument("--result_csv_path", default=r'', \
                        help="Output CSV path for evaluation metrics.")

    args = parser.parse_args()

    print("\n=== Step 1: Extracting predicted blocks ===")
    extract_blocks_sequence(args.raw_outputs_json_dir, args.matching_json_dir, args.extracted_json_dir)

    print("\n=== Step 2: Evaluating predictions ===")
    validate_predicted_scenes(args.groundtruth_json_dir, args.extracted_json_dir, args.result_csv_path)

    print(f"\n=== All done. Results saved to: {args.result_csv_path} ===\n")


if __name__ == '__main__':
    main()
