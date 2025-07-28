import sys
import os
import json
import numpy as np
import genesis as gs
import os
import time
import copy
from PIL import Image
import re
from scipy.spatial.transform import Rotation as R
import random
gs.init(backend=gs.gpu)

cube_type=[
    "arch_red",
    "cube_blue",
    "cube_green",
    "cube_orange",
    "cube_red",
    "cube_yellow",
    "cuboid1_blue",
    "cuboid2_green",
    "cuboid3_yellow",
    "cylinder_orange",
    "semi_cylinder_yellow",
    "triangle_orange"
]
class BlockEnv():
    def __init__(self, config):
       
        self.args = config
        self.max_ids = {}
        self.save_img_dir = config.save_img_dir
        self.save_img = config.save_img
        self.executable_steps = 0
        self.total_steps = 0
        self.max_steps = config.max_steps
        self.max_try = 1
        self.plan_error = 0
        self.block_path = config.block_path
        self.cube_num = config.cube_num
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (3, 0, 1.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 30,
                max_FPS       = 60,
            ),
            vis_options= gs.options.VisOptions(
                show_world_frame = False,
            ),
            sim_options = gs.options.SimOptions(
                dt = 0.01,
                substeps = 4, # 为了更稳定的抓取接触
            ),
            show_viewer = False,
            renderer = gs.renderers.Rasterizer(), # 使用光栅化渲染器
        )
        self.plane  = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        ### modify camera position
        lookat_pos = (0, 0, 0.025)
        side_pos = (lookat_pos[0]+1, lookat_pos[1]+1, lookat_pos[2]+0.5)
        self.camera = self.scene.add_camera(res=(1280, 1280), pos=side_pos, fov=30, lookat=lookat_pos, GUI=False)
        self.task = {}
        self.task_progress = {}
        self.layer_progress = 1
        self.cube = {}
        self.cube_type = cube_type
        self.cube_idx = {k:0 for k in self.cube_type}  
        self.cube_dict = {k:[] for k in self.cube_type}
        self.cube_random_pos = (0,50,0) #
        self.init_scene()
    

    def init_scene(self):
        ori_x, ori_y, ori_z =  self.cube_random_pos
        for cube in self.cube_type:
            for _ in range(self.cube_num):
                rand_x = ori_x + random.uniform(-10.0, 10.0)  # 随机在原点附近的x坐标
                rand_y = ori_y + random.uniform(-10.0, 10.0)  # 随机在原点附近的y坐标
                rand_z = ori_z  # 固定z坐标
                self.cube_dict[cube].append(
                    self.scene.add_entity(
                    gs.morphs.Mesh(
                        file = os.path.join(self.block_path, cube+".obj"),
                        pos = (rand_x, rand_y, rand_z),
                    )
                )
                )
        self.scene.build()
            

    def move_cube_to(self, ori_cube_id, goal_cube_id, relative_type="up", relative_dis=0.05):
        """
        relative_type:
            "up"
            "front"
            "back"
            "right"
            "left"
        """
        
        # 验证 relative_type 是否合法
        valid_relative_types = ["up", "front", "back", "right", "left"]
        assert relative_type in valid_relative_types, f"Invalid relative_type: {relative_type}. Must be one of {valid_relative_types}"
        assert relative_dis > 0, "relative_dis must be a positive number"

        try:
            ori_cube = self.scene.entities[ori_cube_id]
            goal_cube = self.scene.entities[goal_cube_id]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid cube ID: {e}")

        # 获取目标立方体的位置
        goal_cube_pos = goal_cube.get_pos().cpu().numpy()  # -> array([x, y, z])

        # 根据 relative_type 计算新的位置
        if relative_type == "up":
            new_pos = goal_cube_pos + np.array([0, 0, relative_dis])
        elif relative_type == "front":
            new_pos = goal_cube_pos + np.array([relative_dis, 0, 0])
        elif relative_type == "back":
            new_pos = goal_cube_pos - np.array([relative_dis, 0, 0])
        elif relative_type == "left":
            new_pos = goal_cube_pos + np.array([0, relative_dis, 0])
        elif relative_type == "right":
            new_pos = goal_cube_pos - np.array([0, relative_dis, 0])

        # 更新原立方体的位置，仅需step()一次
        ori_cube.set_pos(new_pos)
        self.scene.step()
    

    def update_layer(self):
        update=True
        for block in self.task["blocks"]:
            if block["layer"]==self.layer_progress:
                update = False
        if update:
            self.layer_progress+=1


    def move_cube(self, cube_id):
        cube_name = None
        success = False
        remove_idx = None
        for idx in range(len(self.task["blocks"])):
            i = self.task["blocks"][idx]
            if i["candidate_id"][0] == cube_id and i["layer"]==self.layer_progress:
                self.task_progress["blocks"].append(i)
                cube_name = i["type"]+"_"+i["color"]
                pos = i["position"]
                euler = i["euler"]
                remove_idx = idx

        
        if cube_name is not None and remove_idx is not None:
            self.task["blocks"].pop(remove_idx)
            self.update_layer()
            # euler = [euler[0], euler[1], euler[2]]
            euler = np.radians(euler)
            r = R.from_euler('yzx', euler)
            quat = r.as_quat()
            # print(cube_name, euler, quat)
            self.cube_dict[cube_name][self.cube_idx[cube_name]].set_pos(pos)
            self.cube_dict[cube_name][self.cube_idx[cube_name]].set_quat(quat)
            # print(self.cube_dict[cube_name][self.cube_idx[cube_name]].get_quat())
            self.cube_idx[cube_name]+=1
            success = True

        for _ in range(10):
            self.scene.step()

        return success
        
    def reset(self, task):
        self.scene.reset()
        self.task = task
        self.executable_steps = 0
        self.total_steps = 0
        self.task_progress = {"blocks":[]}
        self.layer_progress = 1
        self.lookat_pos = task["lookat_pos"]
        self.blocks = task["blocks"]
        self.cube_idx = {i:0 for i in self.cube_type}  
        self.pos = (self.lookat_pos[0]+1, self.lookat_pos[1]+1, self.lookat_pos[2]+0.5)
        lookat_pos = (0, 0, 0.025)
        side_pos = (lookat_pos[0]+1, lookat_pos[1]+1, lookat_pos[2]+0.5)
        self.camera.set_pose(pos=side_pos, lookat=lookat_pos)
        obs, _, _, _ = self.camera.render()
        image = Image.fromarray(obs)
        if self.save_img:
            self.save_image_dir = os.path.join(self.save_img_dir, "shape"+self.task["shape_name"])
            if not os.path.exists(self.save_image_dir):
                os.makedirs(self.save_image_dir)
            save_img_path = os.path.join(self.save_image_dir, f"{self.total_steps}.png")
            image.save(save_img_path)
        return obs
    
    def extract_id(self, move_str):
        # 正则表达式提取括号内的数字
        match = re.search(r'Move\((\d+)\)', move_str)
        if match:
            return int(match.group(1))  # 返回整数
        else:
            raise ValueError("Invalid format. Expected 'Move(id)'.")
    
    def get_block_state(self):
        predicted_blocks = {}
        blocks_state = []
        for block_type in self.cube_type:
            for i in range(self.cube_idx[block_type]):
                pos = self.cube_dict[block_type][i].get_pos().cpu().numpy()
                quat = self.cube_dict[block_type][i].get_quat().cpu().numpy()
                r_back = R.from_quat(quat)
                euler_back = r_back.as_euler('yzx')
                blocks_state.append({
                    "type": block_type.split("_")[0],
                    "color": block_type.split("_")[1],
                    "position": pos.tolist(),
                    "euler": euler_back.tolist()
                })
        predicted_blocks["blocks"] = blocks_state
        predicted_blocks["level"] = self.task["level"]
        predicted_blocks["shape_name"] = self.task["shape_name"]
        return predicted_blocks


    def step(self, step, return_act=False):
        """
        vlm can not give the obj id
        """
        success, message = False, None
        feedback = ""
        stop = False
        if len(step)>0:
            self.total_steps += 1
            if not self.total_steps>self.max_steps:
                self.try_time = 0
                while self.try_time<self.max_try and success is False:
                    try:
                        self.try_time+=1
                        idx = self.extract_id(step)
                        step = "Move({})".format(idx)
                        ### step function
                        success = self.move_cube(idx)
                    except Exception as e:
                        print(e)
                        success, message = False, str(e)
                if success is False:
                    feedback = "You are wrong! Modify your answer. The following line failed in a simulator: " + step + "\n"            
                else:
                    ### return state
                    self.executable_steps += 1
            else:
                stop = True
                feedback = "Exceeded the maximum number of steps."
        try:
            ### get image
            image_array,_,_,_ = self.camera.render()
            image = Image.fromarray(image_array)
            if self.save_img:
                save_img_path = os.path.join(self.save_image_dir, f"{self.total_steps}.png")
                image.save(save_img_path)
            # print(image.shape)

        except Exception as e:
            print(e)
            stop = True
            feedback = "Simulator timeout error."
            image = None

        if return_act:
            return  image_array, success, message, feedback, stop, step
        else:
            return  image_array, success, message, feedback, stop 