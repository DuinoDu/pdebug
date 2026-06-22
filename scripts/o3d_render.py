#!/usr/bin/env python
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import typer
from easydict import EasyDict as edict


# @task(name="my-tool")
def main(
    input_json: str,
    pcd_file: str = None,
    output: Optional[str] = typer.Option(None, help="output name"),
):
    """Render 3d demo using open3d."""
    typer.echo(typer.style(f"hello, tool", fg=typer.colors.GREEN))
    if not output:
        output = "o3d_render_output"
    os.makedirs(output, exist_ok=True)

    if not pcd_file:
        pcd_data = o3d.data.PCDPointCloud()
        pcd = o3d.io.read_point_cloud(pcd_data.path)
    else:
        pcd = o3d.io.read_point_cloud(pcd_file)

    test_data_path = "test_data"
    assert os.path.exists(
        test_data_path
    ), f"{test_data_path} not found, please create it first."
    render_option_path = os.path.join(test_data_path, "renderoption.json")

    image_path = os.path.join(output, "image")
    os.makedirs(image_path, exist_ok=True)
    depth_path = os.path.join(output, "depth")
    os.makedirs(depth_path, exist_ok=True)

    glb = edict()
    glb.index = -1
    glb.trajectory = o3d.io.read_pinhole_camera_trajectory(input_json)
    glb.vis = o3d.visualization.Visualizer()

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        nonlocal glb
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(
                os.path.join(depth_path, "{:05d}.png".format(glb.index)),
                np.asarray(depth),
                dpi=1,
            )
            plt.imsave(
                os.path.join(image_path, "{:05d}.png".format(glb.index)),
                np.asarray(image),
                dpi=1,
            )
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True
            )
        else:
            glb.vis.register_animation_callback(None)
        return False

    vis = glb.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    typer.run(main)
