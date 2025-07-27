import genesis as gs
import os
import cv2
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_background_plane(scene, glb_dir, background_type):
    """
    Load the background environment for the scene.

    Args:
        scene (gs.Scene): The simulation scene.
        glb_dir (str): Directory containing GLB files.
        background_type (str): Type of background ('White', 'DarkCheckerboard', etc.).

    Returns:
        gs.Entity: Background plane or mesh entity.
    """
    if background_type == "White":
        return scene.add_entity(gs.morphs.Plane(visualization=False))
    elif background_type == "DarkCheckerboard":
        return scene.add_entity(gs.morphs.Plane(visualization=True))
    else:
        background_path = os.path.join(glb_dir, f"{background_type}.glb")
        plane = scene.add_entity(gs.morphs.Plane(visualization=False))
        try:
            if "ConcreteFloor" in background_path:
                scene.add_entity(gs.morphs.Mesh(file=background_path, pos=(-1.0, 0.0, 0.0), euler=(90, 0, 0)))
                return scene.add_entity(
                    gs.morphs.Mesh(file=background_path, pos=(0.0, 0.0, 0.0), euler=(90, 0, 0))
                )
            else:
                return scene.add_entity(
                    gs.morphs.Mesh(file=background_path, pos=(0.0, 0.0, 0.0), euler=(90, 0, 0))
                )
        except Exception as e:
            print(f"[Error] Failed to load background mesh: {e}")
            return scene.add_entity(gs.morphs.Plane(visualization=False))


def load_scene_from_json(scene, json_path, obj_dir):
    """
    Load scene objects from a JSON file.

    Args:
        scene (gs.Scene): The simulation scene.
        json_path (str): Path to the JSON file.
        obj_dir (str): Directory containing block OBJ files.

    Returns:
        Tuple[str, Tuple[float, float, float]]: Scene name and lookat position.
    """
    with open(json_path, 'r') as f:
        scene_data = json.load(f)
    
    scene_name = scene_data["shape_name"]
    lookat_pos = tuple(scene_data["lookat_pos"])

    for block in scene_data["blocks"]:
        block_type = block["type"]
        color = block["color"]
        position = tuple(block["position"])
        euler = tuple(block["euler"])

        obj_filename = f"{block_type}_{color}.obj"
        obj_path = os.path.join(obj_dir, obj_filename)

        scene.add_entity(
            gs.morphs.Mesh(file=obj_path, pos=position, euler=euler)
        )
    
    return scene_name, lookat_pos


def render_and_save_views(scene, camera, scene_name, lookat_pos, output_dir):
    """
    Render 3 views (front, side, top) of the scene and save as images.

    Args:
        scene (gs.Scene): The simulation scene.
        camera (gs.Camera): The scene camera.
        scene_name (str): Scene name.
        lookat_pos (tuple): The center of the scene.
        output_dir (str): Directory to save outputs.
    """
    # Set front view
    front_pos = (lookat_pos[0] + 0.75, lookat_pos[1], lookat_pos[2] + 0.25)
    camera.set_pose(pos=front_pos, lookat=lookat_pos)
    camera.start_recording()
    for _ in range(16):
        scene.step()
        camera.render()

    # Set side view
    side_pos = (lookat_pos[0] + 0.5, lookat_pos[1] + 0.5, lookat_pos[2] + 0.25)
    camera.set_pose(pos=side_pos, lookat=lookat_pos)
    for _ in range(16):
        scene.step()
        camera.render()

    # Set top view
    top_pos = (lookat_pos[0], lookat_pos[1], lookat_pos[2] + 1.0)
    camera.set_pose(pos=top_pos, lookat=lookat_pos)
    for _ in range(16):
        scene.step()
        camera.render()

    # Save video
    video_path = os.path.join(output_dir, f"{scene_name}.mp4")
    camera.stop_recording(save_to_filename=video_path, fps=60)

    # Extract key frames from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Error] Failed to open rendered video.")
        return

    frame_indices = [15, 31, 47]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx, frame_index in enumerate(frame_indices):
        if frame_index >= total_frames:
            print(f"[Warning] Frame {frame_index} is out of range (max: {total_frames}).")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(output_dir, f"{scene_name}_{idx}.png")
            if frame_index == 47:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(frame_name, frame)
            print(f"[✓] Saved frame {frame_index} to {frame_name}")
        else:
            print(f"[Error] Could not read frame {frame_index}")
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Render 3 views of a scene described by JSON.")
    parser.add_argument("--scene_json_path", type=str, default=r"/data/SCENEs_400_Goal_Jsons/100.json",
                        help="Path to JSON file describing the scene.")
    parser.add_argument("--assets_obj_dir", type=str, default=r"/data/block_assets",
                        help="Directory containing block OBJ models.")
    parser.add_argument("--background_glb_dir", type=str, default=r"/data/floor_assets",
                        help="Directory containing background GLB files.")
    parser.add_argument("--background_type", type=str, default=r"LightWoodBoard",
                        choices=['White', 'DarkCheckerboard', 'LightWoodBoard', 'DarkWoodBoard', 'ConcreteFloor', 'Lawn'],
                        help="Background environment type.")
    parser.add_argument("--output_dir", type=str, default=r"/data/test",
                        help="Directory to save the rendered outputs.")
    parser.add_argument("--show_viewer", action="store_true", help="Whether to show the scene viewer.")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            background_color=(1.0, 1.0, 1.0),
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=4,
        ),
        show_viewer=args.show_viewer,
        renderer=gs.renderers.Rasterizer(),
    )

    # Load scene and camera
    load_background_plane(scene, args.background_glb_dir, args.background_type)
    camera = scene.add_camera(res=(1280, 1280), fov=30, GUI=False)

    scene_name, lookat_pos = load_scene_from_json(scene, args.scene_json_path, args.assets_obj_dir)
    scene.build()

    render_and_save_views(scene, camera, scene_name, lookat_pos, args.output_dir)


if __name__ == "__main__":
    main()
