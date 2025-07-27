import genesis as gs
import os

def show(scene):

    semi_cylinder = scene.add_entity(
        gs.morphs.Mesh(
            file = 'semi_cylinder_yellow.obj',
            pos = (0., 0., 0.0)
        )
    )

    triangle = scene.add_entity(
        gs.morphs.Mesh(
            file = 'triangle_orange.obj',
            pos = (0., 0.1, 0.0)
        )
    )

    cube_red = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cube_red.obj',
            pos = (0.0, 0.2, 0.025)
        )
    )

    cube_yellow = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cube_yellow.obj',
            pos = (0.1, 0.2, 0.025)
        )
    )

    cube_blue = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cube_blue.obj',
            pos = (0.2, 0.2, 0.025)
        )
    )

    cube_green = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cube_green.obj',
            pos = (0.3, 0.2, 0.025)
        )
    )

    cube_orange = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cube_orange.obj',
            pos = (0.4, 0.2, 0.025)
        )
    )


    cylinder = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cylinder_orange.obj',
            pos = (0.0, 0.3, 0.05)
        )
    )

    arch = scene.add_entity(
        gs.morphs.Mesh(
            file = 'arch_red.obj',
            pos = (0.0, 0.4, 0.025)
        )
    )

    cuboid_1 = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cuboid1_blue.obj',
            pos = (0.0, 0.5, 0.0125)
        )
    )

    cuboid_2 = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cuboid2_green.obj',
            pos = (0.0, 0.6, 0.0125)
        )
    )

    cuboid_3 = scene.add_entity(
        gs.morphs.Mesh(
            file = 'cuboid3_yellow.obj',
            pos = (0.0, 0.7, 0.0125)
        )
    )




if __name__ == "__main__":

    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (2.5, 2.5, 1),
            camera_lookat = (0.0, 0.0, 0.25),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
            substeps = 4, # 为了更稳定的抓取接触
        ),
        show_viewer = False,
        renderer = gs.renderers.Rasterizer(), # 使用光栅化渲染器
    )

    # build scene
    plane = scene.add_entity(gs.morphs.Plane(visualization=True))
    show(scene)


    cam = scene.add_camera(
        res    = (1280, 1280),
        pos    = (1.5, 1.5, 1.5),
        lookat = (0, 0.3, 0.2),
        fov    = 30,
        GUI    = False,
    )
    scene.build()


    cam.start_recording()

    for i in range(64):
        scene.step()
        cam.render()

    # stop recording and save video
    cam.stop_recording(save_to_filename="show_x.mp4", fps=60)