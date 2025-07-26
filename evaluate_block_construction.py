import os
import json
import re
import csv
import glob
import argparse
from pathlib import Path
import pandas as pd
import glob


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

    try:
        # Qwen-VL-Max
        return raw_data['response']
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


def are_blocks_equal(block1, block2, evaluate_mode='A'):
    """
    Check if two blocks are equal based on specific keys ('type', 'color', 'euler').

    Args:
        block1 (dict): The first block to compare.
        block2 (dict): The second block to compare.
        evaluate_mode(str, 'A'): Pose-Constrained Evaluation (with Euler Pose Consideration)
        evaluate_mode(str, 'B'): Topology-Oriented Evaluation (without Euler Pose Consideration)

    Returns:
        bool: True if the blocks are equal based on the selected keys, False otherwise.
    """
    comparison_keys = ['type', 'color', 'euler'] if evaluate_mode == 'A' else ['type', 'color']
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
               are_blocks_equal(pred_block, gt_block, evaluate_mode='A'):
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

def validate_block_placement_level1(ground_truth_path: Path, predicted_path: Path):
    """Match predicted blocks to ground truth and compute precision/recall/F1."""

    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    gt_blocks = [
        {**block, 'placed': False}
        for block in ground_truth.get('blocks', [])
    ]
    gt_level = ground_truth.get('level', 0)
    gt_block_count = len(gt_blocks)

    with open(predicted_path, 'r') as f:
        pred_blocks = [
            {**block, 'flag': 0}
            for block in json.load(f).get('blocks', [])
        ]

    for pred_block in pred_blocks:
        for gt_block in gt_blocks:
            if not gt_block['placed'] and are_blocks_equal(pred_block, gt_block, evaluate_mode='B'):
                gt_block['placed'] = True
                pred_block['flag'] = gt_block.get('order', 0)
                break

    TP = sum(block['flag'] > 0 for block in pred_blocks)
    FP = sum(block['flag'] == 0 for block in pred_blocks)
    FN = gt_block_count - TP

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return gt_level, gt_block_count, TP, FP, FN, precision, recall, f1


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



def validate_predicted_scenes_level1(groundtruth_json_dir, extracted_json_dir, result_csv_path):

    header = ['Scene', 'Level', 'Blocks_num', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score']
    with open(result_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for idx, predicted_path in enumerate(sorted(glob.glob(os.path.join(extracted_json_dir, "*.json")))):

            scene_name = os.path.basename(predicted_path)[:3]
            groundtruth_path = os.path.join(groundtruth_json_dir, scene_name + '.json')
            Level, num, TP, FP, FN, precision, recall, F1 = validate_block_placement_level1(groundtruth_path, predicted_path)

            print(f"[Successfully validated]: {scene_name}......{idx}/{400}")

            if Level == 2 and num < 7:
                pre_res_data = [scene_name, Level,
                                int(TP+FN), TP, FP, FN,
                                round(precision, 3), 
                                round(recall, 3), 
                                round(F1, 3)]
                writer.writerow(pre_res_data)
                csvfile.flush()


def compute_micro_metrics_per_csv(csv_path: str, output_path: str) -> tuple[dict, dict]:
    """
    Compute level-wise and overall micro-averaged precision, recall, and F1 from a single CSV file.
    
    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the result TXT file.

    Returns:
        tuple: A tuple of (level_metrics_dict, overall_metrics_dict).
    """
    df = pd.read_csv(csv_path)

    # Level-wise metrics
    level_metrics = {}
    for level, group in df.groupby("Level"):
        tp, fp, fn = group["TP"].sum(), group["FP"].sum(), group["FN"].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        level_metrics[level] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    # Overall metrics
    tp_total, fp_total, fn_total = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    precision_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_total = (2 * precision_total * recall_total) / (precision_total + recall_total) if (precision_total + recall_total) > 0 else 0

    overall_metrics = {
        "Precision": precision_total,
        "Recall": recall_total,
        "F1": f1_total
    }

    # Write results to TXT
    with open(output_path, "w") as f:
        f.write("Level-wise Micro-Averaging Metrics:\n")
        for level, metrics in sorted(level_metrics.items()):
            f.write(f"Level {level}: Precision={metrics['Precision']*100:.4f}, "
                    f"Recall={metrics['Recall']*100:.4f}, F1={metrics['F1']*100:.4f}\n")

        f.write("\nOverall Micro-Averaging Metrics:\n")
        f.write(f"Precision={precision_total*100:.4f}, "
                f"Recall={recall_total*100:.4f}, F1={f1_total*100:.4f}\n")

    return level_metrics, overall_metrics


def batch_process_csv_metrics(csv_dir: str, output_dir: str) -> None:
    """
    Batch process all CSV files in a directory to compute and export micro-averaging metrics.

    Args:
        csv_dir (str): Directory containing CSV files with TP, FP, FN columns.
        output_dir (str): Directory to save TXT result files.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    total_files = len(csv_files)

    if total_files == 0:
        print(f"[Warning] No CSV files found in: {csv_dir}")
        return

    for idx, csv_path in enumerate(csv_files, start=1):
        file_stem = os.path.splitext(os.path.basename(csv_path))[0]
        txt_path = os.path.join(output_dir, f"{file_stem}.txt")

        try:
            level_metrics, overall_metrics = compute_micro_metrics_per_csv(csv_path, txt_path)
            print(f"[{idx}/{total_files}] Processed: {file_stem} ✅")
        except Exception as e:
            print(f"[{idx}/{total_files}] Failed: {file_stem} ❌ — {e}")


def insert_micro_metrics_to_txt(level1_result_csv_path: str, txt_filename_suffix: str = ".txt") -> None:
    """
    Compute micro-averaged precision, recall, and F1 from a CSV file,
    and insert the formatted result into the second line of the corresponding TXT file.

    Args:
        level1_result_csv_path (str): Path to the Level-1 CSV result file.
        txt_filename_suffix (str): File suffix to match TXT file (default: ".txt").
    """
    csv_path = Path(level1_result_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Step 1: Load CSV and compute Micro Precision/Recall/F1
    df = pd.read_csv(csv_path)
    tp_total = df["TP"].sum()
    fp_total = df["FP"].sum()
    fn_total = df["FN"].sum()

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    formatted_metrics = (
        f"Level 1: Precision={precision * 100:.4f}, "
        f"Recall={recall * 100:.4f}, F1={f1 * 100:.4f}\n"
    )

    # Step 2: Locate the corresponding .txt file in the same directory
    txt_candidates = list(csv_path.parent.glob(f"*{txt_filename_suffix}"))
    if len(txt_candidates) == 0:
        raise FileNotFoundError(f"No TXT file with suffix '{txt_filename_suffix}' found in {csv_path.parent}")
    elif len(txt_candidates) > 1:
        print(f"[Warning] Multiple TXT files found, using the first one: {txt_candidates[0].name}")

    txt_path = txt_candidates[0]

    # Step 3: Read original TXT lines, insert metric into 2nd line
    with txt_path.open("r", encoding="utf-8") as f:
        original_lines = f.readlines()

    updated_lines = []
    if len(original_lines) >= 1:
        updated_lines.append(original_lines[0])  # first line
        updated_lines.append(formatted_metrics)  # insert micro metric line
        updated_lines.extend(original_lines[1:])  # rest
    else:
        updated_lines = [formatted_metrics]

    # Step 4: Overwrite TXT file
    with txt_path.open("w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    # Step 5: Rename the file to "Results_Levels.txt"
    new_txt_path = txt_path.parent / "Results_Levels.txt"
    txt_path.rename(new_txt_path)

    print(f"[✓] Inserted micro metrics into {txt_path.name} (second line).")




def main():
    parser = argparse.ArgumentParser(description="Extract predicted blocks and evaluate them against ground truth.")
    parser.add_argument("--raw_outputs_json_dir", default=r'/data1/vincia/PIPELINE_Supplementary/Embodied-R1/Raw-Infer-400', 
                        help="Directory with raw model output JSONs.")
    parser.add_argument("--matching_json_dir", default=r'/data1/vincia/data/SCENEs_400_Cand_Jsons_250305', \
                        help="Directory with candidate block JSONs.")
    parser.add_argument("--extracted_json_dir", default=r'/data1/vincia/PIPELINE_Supplementary/Embodied-R1/Evaluat_test/extracted-Infer-400', \
                        help="Directory to save extracted block sequences.")
    parser.add_argument("--groundtruth_json_dir", default=r'/data1/vincia/data/SCENEs_400_Goal_jsons_250306', \
                        help="Directory with ground truth goal JSONs.")
    parser.add_argument("--results_dir", default=r'/data1/vincia/PIPELINE_Supplementary/Embodied-R1/Evaluat_test/Results/Results.csv', \
                        help="Output CSV path for evaluation metrics.")

    args = parser.parse_args()

    print("\n=== Step 1: Extracting predicted blocks ===")
    extract_blocks_sequence(args.raw_outputs_json_dir, args.matching_json_dir, args.extracted_json_dir)

    print("\n=== Step 2: Evaluating predictions ===")
    result_csv_path = os.path.join(args.results_dir, 'Results_level234.csv')
    validate_predicted_scenes(args.groundtruth_json_dir, args.extracted_json_dir, result_csv_path)

    level1_result_csv_path = os.path.join(args.results_dir, 'Results_level1.csv')
    validate_predicted_scenes_level1(args.groundtruth_json_dir, args.extracted_json_dir, level1_result_csv_path)

    print("\n=== Step 3: Computing overall metrics ===")
    batch_process_csv_metrics(args.results_dir, args.results_dir)
    insert_micro_metrics_to_txt(level1_result_csv_path)


    print(f"\n=== All done. Results saved to: {args.results_dir} ===\n")


if __name__ == '__main__':
    main()
