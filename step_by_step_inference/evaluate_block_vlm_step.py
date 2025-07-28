import os
import sys
import json
import logging
import argparse
import threading
from block_env import BlockEnv
from vlm_planner_block_step import VLM_Planner
from tqdm import tqdm
from config import Config
import random
from PIL import Image
from utils import *

extract = "extracted_json_dir"
match = "maching_json_dir"
raw_outputs = "raw_outputs_json_dir"
result_csv_path_1_str = "result_csv_path_1.csv"
# result_csv_path_2_str = "result_csv_path_2.csv"

def load_task(task_file):
    with open(task_file,"r") as f:
        task = json.load(f) 
    
    return task 

def check_response(response, last_response, succ):
    if response is None:
        return False
    elif len(response) == 0:
        return False
    elif response==last_response and succ==False:
        return False
    else:
        return True

def save_plan(save_dir, file_name, steps, split):
    if not os.path.exists(os.path.join(save_dir,split)):
        os.makedirs(os.path.join(save_dir,split))
    with open(os.path.join(save_dir,split,file_name+".json"),"w") as f:
        plan = {"steps":steps}
        json.dump(plan, f, indent=4)

def check_execution_status(logs, max_failures=6):
    fail_count = sum(1 for log in logs if "Execution failed" in log)
    done = fail_count > max_failures
    return done


def Processor(config):
    fail={}    
    env = BlockEnv(config)
    task_planner = VLM_Planner(config)
    dir_path = config.data_path
    goal_image_path = config.data_path.replace("cand_outline_jsons","imgs")
    obj_image_path = config.data_path.replace("cand_outline_jsons","cand_imgs") 
    # ground_truth_path = config.data_path.replace("cand_json","w_depend_json")
    ground_truth_path = config.data_path.replace("cand_outline_jsons","jsons")
    total_len = len(os.listdir(dir_path))
    # dir_list = random.sample(os.listdir(dir_path),config.max_len) if config.max_len<=total_len and config.max_len!=0 else os.listdir(dir_path)
    dir_list = [f for f in os.listdir(dir_path) if f.endswith("_cand.json")]
    initial_states = []
    final_states = []
    exec_per_task = []
    test_tasks = []
    final_states_GT = []
    max_check_error = 3
    num_trial = 1
    extract_path = os.path.join(config.results_dir,extract)
    match_path =  os.path.join(config.results_dir,match)
    raw_outputs_path = os.path.join(config.results_dir,raw_outputs)
    result_csv_path_1 = os.path.join(config.results_dir, result_csv_path_1_str)
    # result_csv_path_2 = os.path.join(config.results_dir, result_csv_path_2_str)
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    if not os.path.exists(match_path):
        os.makedirs(match_path)
    if not os.path.exists(raw_outputs_path):
        os.makedirs(raw_outputs_path)

    for trial in range(num_trial):
        if trial != 0 and 'memory' in config.method_type:
            retrieve_memory = True
        elif config.load_memory:
            with open(os.path.join(config.results_dir, f'memory_trial{trial}.json'), 'r') as f:
                task_planner.memory = json.load(f)
            retrieve_memory = True
        else:
            retrieve_memory = False

        for task_file in tqdm(dir_list):
            task = load_task(os.path.join(dir_path,task_file))
            obs={}
            done = False
            fail_info = []
            obs["image"] = env.reset(task)
            # Image.fromarray(obs["image"]).save("/data1/linmin/zero-shot-planner/test.png")
            final_states_GT.append(task["blocks"])
            test_tasks.append(task_file)
            last_response = None
            print(task_file)
            print("\n")
            steps = []
            goal_path = []
            for idx in range(3):
                goal_path.append(os.path.join(goal_image_path, task_file.split("_")[0]+"_"+str(idx)+".png"))
            obj_path = os.path.join(obj_image_path, task_file.replace(".json",".png"))
            task_planner.reset(goal_path, obj_path, retrieve_memory)
            check_error = 0
            succ = False
            while not done:
                response = task_planner.plan(obs)
                print("VLM Output:",response)
                if check_response(response, last_response, succ):
                    last_response = response
                    step = response.replace("\n","")
                    steps.append(step)
                    if "Done" in step:
                        done = True
                    else:
                        script = step
          
                        obs["image"], succ, message, feedback, done, step = env.step(script, return_act=True)
                        print(succ, message)

                        task_planner.history[-1] = step + "\n"
                        task_planner.updata_history(succ)
                        if not succ:
                    
                            fail_info.append(str(script) + " : " + str(message))
                            # done = True

                else:
                    fail_info.append(response)
                    # env.total_steps += 1
                    step = step.replace("\n","")
                    idx = env.extract_id(step)
                    step = "Move({})".format(idx)
                    obs["image"], succ, message, feedback, done = env.step(step=[], return_act=False)
                    task_planner.history[-1] = step + "\n"
                    task_planner.updata_history(success=False)
                    done = check_execution_status(task_planner.history)
                    check_error += 1
                    if check_error >= max_check_error:
                        done = True

                if done:
                    final_state=env.get_block_state()
                    with open(os.path.join(extract_path, task["shape_name"]+"_16_cand_blocks"+".json"),"w")as f:
                        json.dump(final_state, f, indent=4)
                        
                    if env.total_steps==0:
                        exec_per_task.append(0)
                    else:
                        exec_per_task.append(env.executable_steps/env.total_steps)
                    
            validate_predicted_scenes(
            groundtruth_json_dir=ground_truth_path,
            extracted_json_dir=extract_path,
            result_csv_path=result_csv_path_1,
            # euler_constraint=config.euler_constraint
        )
            
if __name__=="__main__":
    config = Config()
    Processor(config)
    