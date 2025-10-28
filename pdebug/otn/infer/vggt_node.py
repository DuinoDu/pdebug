import gc
import glob
import os
from pathlib import Path
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.utils.env import TORCH_INSTALLED, VGGT_INSTALLED, VISER_INSTALLED
from pdebug.utils.fileio import do_system
from pdebug.visp import draw

import cv2
import numpy as np
import typer

if TORCH_INSTALLED:
    import torch
    import torch.nn.functional as F

if VGGT_INSTALLED:
    from vggt.dependency.np_to_pycolmap import (
        batch_np_matrix_to_pycolmap,
        batch_np_matrix_to_pycolmap_wo_track,
    )
    from vggt.dependency.track_predict import predict_tracks
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import (
        closed_form_inverse_se3,
        unproject_depth_map_to_point_map,
    )
    from vggt.utils.helper import (
        create_pixel_coordinate_grid,
        randomly_limit_trues,
    )
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

if VISER_INSTALLED:
    import viser
    import viser.transforms as viser_tf


@otn_manager.NODE.register(name="vggt")
def vggt_main(
    input_path: str = None,
    output: str = "vggt_output",
    max_image_num: int = 30,
    sample_mode: str = "random",  # topk
    use_ba: bool = False,
    save_colmap: bool = False,
    max_reproj_error: float = 8.0,
    vis_thresh: float = 0.2,
    query_frame_num: int = 8,
    max_query_pts: int = 4096,
    fine_tracking: bool = True,
    conf_thres_value: float = 5.0,
    apply_rgba_alpha: bool = True,
    cache: bool = True,
):
    """Main VGGT inference node for 3D reconstruction from images."""
    output = Path(output)
    output.mkdir(exist_ok=True)

    # Set device and dtype
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    # Load images
    image_dir = (
        Path(input_path) / "images"
        if (Path(input_path) / "images").exists()
        else Path(input_path)
    )
    image_path_list = glob.glob(str(image_dir / "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list)

    print(f"Find {len(image_path_list)} images.")
    if sample_mode == "random":
        sample_interval = 1
        if len(image_path_list) > max_image_num:
            sample_interval = len(image_path_list) // max_image_num
        image_path_list = image_path_list[::sample_interval]
    elif sample_mode == "topk":
        image_path_list = image_path_list[:max_image_num]
    else:
        raise NotImplementedError
    print(f"Sample to {len(image_path_list)} images.")

    # Load and preprocess images
    vggt_fixed_resolution = 518
    img_load_resolution = 1024
    images, original_coords = load_and_preprocess_images_square(
        image_path_list, img_load_resolution
    )
    images = images.to(device)
    original_coords = original_coords.to(device)

    # Run VGGT inference
    with torch.no_grad():
        # Resize to VGGT input size
        images_resized = F.interpolate(
            images,
            size=(vggt_fixed_resolution, vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )

        with torch.amp.autocast("cuda", dtype=dtype):
            images_resized = images_resized[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_resized)

            # Predict cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_resized.shape[-2:]
            )

            # Predict depth maps
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_resized, ps_idx
            )

    # Convert to numpy
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    # Unproject depth to 3D points
    points_3d = unproject_depth_map_to_point_map(
        depth_map, extrinsic, intrinsic
    )

    # Save outputs
    np.save(output / "extrinsic.npy", extrinsic)
    np.save(output / "intrinsic.npy", intrinsic)
    np.save(output / "depth_map.npy", depth_map)
    np.save(output / "depth_conf.npy", depth_conf)
    np.save(output / "points_3d.npy", points_3d)
    with open(output / "imglist.txt", "w") as fid:
        for img_path in image_path_list:
            fid.write(img_path + "\n")

    del model, images_resized, original_coords
    gc.collect()
    torch.cuda.empty_cache()

    # Bundle adjustment if requested
    if use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution

        with torch.amp.autocast("cuda", dtype=dtype):
            (
                pred_tracks,
                pred_vis_scores,
                pred_confs,
                points_3d_ba,
                points_rgb,
            ) = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=fine_tracking,
            )

            torch.cuda.empty_cache()

        # Rescale intrinsic matrix
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh

        # Convert to COLMAP format
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d_ba,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=False,
            camera_type="SIMPLE_PINHOLE",
            points_rgb=points_rgb,
        )

        if reconstruction is not None:
            import pycolmap

            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)

            # Save BA reconstruction
            sparse_dir = output / "sparse_ba"
            sparse_dir.mkdir(exist_ok=True)
            reconstruction.write(str(sparse_dir))
    elif save_colmap:
        max_points_for_colmap = 100000
        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images,
            size=(vggt_fixed_resolution, vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )
        # Save COLMAP reconstruction
        sparse_dir = output / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        reconstruction.write(str(sparse_dir))

    print(f"saved to {output}")
    return str(output)


@otn_manager.NODE.register(name="vggt-viser")
def vggt_viser(
    input_path: str,
    port: int = 6150,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
):
    """
    Visualize vggt result using viser.

    Args:
        input_path: Path to vggt output directory
        port: Port number for viser server
        init_conf_threshold: Initial percentage of low-confidence points to filter out
        use_point_map: Whether to use precomputed point map instead of depth-based points
        background_mode: Whether to run server in background thread
        mask_sky: Whether to apply sky segmentation
    """
    assert VGGT_INSTALLED, "vggt is required."
    assert VISER_INSTALLED, "viser is required."

    input_dir = Path(input_path)

    # Load VGGT output data
    extrinsic = np.load(input_dir / "extrinsic.npy")
    intrinsic = np.load(input_dir / "intrinsic.npy")
    depth_map = np.load(input_dir / "depth_map.npy")
    depth_conf = np.load(input_dir / "depth_conf.npy")
    points_3d = np.load(input_dir / "points_3d.npy")
    image_paths = sorted(
        [l.strip() for l in open(input_dir / "imglist.txt", "r")]
    )

    # Load and preprocess images for visualization
    import cv2

    images_list = []
    for img_path in image_paths[: len(depth_map)]:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (depth_map.shape[2], depth_map.shape[1]))
        images_list.append(img.transpose(2, 0, 1) / 255.0)

    images = np.array(images_list)

    # Create prediction dict similar to demo_viser.py
    pred_dict = {
        "images": images,
        "world_points": points_3d,
        "world_points_conf": depth_conf,
        "depth": depth_map[..., np.newaxis],
        "depth_conf": depth_conf,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
    }

    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port, show_axes=True)
    server.gui.configure_theme(
        titlebar_content=None, control_layout="collapsible"
    )

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    if depth_map.ndim == 5:
        depth_map = depth_map[..., 0]

    # Compute world points from depth if not using the precomputed point map
    if use_point_map:
        world_points = world_points_map
        conf = conf_map
    else:
        world_points = unproject_depth_map_to_point_map(
            depth_map, extrinsics_cam, intrinsics_cam
        )
        conf = depth_conf

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras", initial_value=True
    )

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    # Create the main point cloud handle
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # Store references to frames & frustums so we can toggle visibility
    frames = []
    frustums = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """Add camera frames and frustums to the scene."""
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum, frame):
            @frustum.on_click
            def _(_):
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in img_ids:
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=2 * np.arctan2(h / 2, 1.1 * h),
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud():
        """Update the point cloud based on current GUI selections."""
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_):
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_):
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_):
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Viser server started. Press Ctrl+C to exit.")

    if background_mode:
        import threading
        import time

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        try:
            while True:
                import time

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nShutting down viser server...")

    return server


if __name__ == "__main__":
    typer.run(vggt_main)
