import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pdebug.otn import manager as otn_manager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml


@otn_manager.NODE.register(name="linemod-verify")
def verify_data(
    data_path: str = None,
    category: str = None,
    topk: int = -1,
) -> bool:
    """
    Check linemod dataset given category, including: rgb, mask, label.
    Generate visualizing result of mask and label. Check train.txt and
    test.txt, ensure they are not overlapped.

    """
    data_path = Path(data_path)
    category_path = data_path / category

    # Check if category directory exists
    if not category_path.exists():
        print(f"Error: Category directory {category_path} does not exist")
        return False

    # Check required subdirectories
    required_dirs = ["JPEGImages", "mask", "labels"]
    for dir_name in required_dirs:
        dir_path = category_path / dir_name
        if not dir_path.exists():
            print(f"Error: Required directory {dir_path} does not exist")
            return False

    # Check train.txt and test.txt
    train_file = category_path / "train.txt"
    test_file = category_path / "test.txt"

    if not train_file.exists():
        print(f"Error: {train_file} does not exist")
        return False

    if not test_file.exists():
        print(f"Error: {test_file} does not exist")
        return False

    # Read train and test files
    with open(train_file, "r") as f:
        train_files = set(line.strip() for line in f if line.strip())

    with open(test_file, "r") as f:
        test_files = set(line.strip() for line in f if line.strip())

    # Check for overlap
    overlap = train_files.intersection(test_files)
    if overlap:
        print(
            f"Error: Found {len(overlap)} overlapping files between train.txt and test.txt"
        )
        for file in list(overlap)[:5]:  # Show first 5 overlaps
            print(f"  - {file}")
        return False

    # Check RGB, mask, and label files using train.txt and test.txt
    rgb_dir = category_path / "JPEGImages"
    mask_dir = category_path / "mask"
    label_dir = category_path / "labels"

    # Use files from train.txt and test.txt as the source of truth
    all_files = sorted(train_files.union(test_files))

    # Check each file has corresponding files in all directories
    missing_files = []
    found_rgb_files = set()
    found_mask_files = set()
    found_label_files = set()

    for file_name in all_files:
        missing = []

        # Check for RGB file (support both .png and .jpg)
        rgb_file = rgb_dir / file_name
        if rgb_file.exists():
            found_rgb_files.add(file_name)
        else:
            missing.append("rgb")

        # Check for mask file
        name = os.path.splitext(file_name)[0]
        assert len(name) == 6
        mask_path = mask_dir / f"{name[2:]}.png"
        if mask_path.exists():
            found_mask_files.add(file_name)
        elif (mask_dir / f"{name}.png").exists():
            found_mask_files.add(file_name)
        else:
            missing.append("mask")

        # Check for label file
        label_path = label_dir / f"{name}.txt"
        if label_path.exists():
            found_label_files.add(file_name)
        else:
            missing.append("label")

        if missing:
            missing_files.append((file_name, missing))

    if missing_files:
        print(
            f"Warning: Found {len(missing_files)} files with missing components:"
        )
        for file_name, missing in missing_files[:5]:  # Show first 5
            print(f"  - {file_name}: missing {', '.join(missing)}")

    # Generate MP4 visualization for all complete files
    visualize_dir = category_path / "visualization"
    if visualize_dir.exists():
        os.system(f"rm -rf {visualize_dir}")
    visualize_dir.mkdir(exist_ok=True)

    # Only visualize files that have all required components
    complete_files = found_rgb_files.intersection(
        found_mask_files
    ).intersection(found_label_files)
    all_files = sorted(list(complete_files))

    if not all_files:
        print("No complete files found for visualization")
        return True
    if topk > 0:
        all_files = all_files[:topk]

    print(
        f"Generating MP4 visualization for {len(all_files)} complete files..."
    )

    # Define edges for connecting keypoints
    edges_corners = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    from pdebug.piata import Output

    video_path = visualize_dir / f"{category}_visualization.mp4"
    video_writer = Output(video_path, fps=10, name="video_ffmpeg").get_writer()

    for file_idx, file_name in enumerate(all_files):
        name = os.path.splitext(file_name)[0]
        rgb_path = rgb_dir / file_name
        mask_path = mask_dir / f"{name[2:]}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{name}.png"
        label_path = label_dir / f"{name}.txt"

        try:
            # Load images
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is None:
                print(f"Warning: Could not load {rgb_path}")
                continue

            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Load label
            with open(label_path, "r") as f:
                labels = [line.strip().split() for line in f if line.strip()]

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # RGB image
            axes[0].imshow(rgb_img)
            if file_name != os.path.basename(os.path.realpath(rgb_path)):
                axes[0].set_title(
                    f"RGB Image: {file_name} | {os.path.basename(os.path.realpath(rgb_path))}"
                )
            else:
                axes[0].set_title(f"RGB Image: {file_name}")
            axes[0].axis("off")

            # Mask
            axes[1].imshow(mask_img, cmap="gray")
            axes[1].set_title(f"Mask: {os.path.basename(mask_path)}")
            axes[1].axis("off")

            # RGB with mask overlay and label visualization
            overlay = rgb_img.copy()
            # Create a more subtle mask visualization (red color for mask)
            mask_colored = np.zeros_like(rgb_img)
            mask_colored[:, :, 0] = mask_img  # Red channel for mask

            # Only apply mask where mask_img > 0
            mask_indices = mask_img > 0
            overlay[mask_indices] = cv2.addWeighted(
                overlay[mask_indices], 0.7, mask_colored[mask_indices], 0.3, 0
            )

            axes[2].imshow(overlay)
            axes[2].set_title(
                f"RGB + Mask + Keypoints: {os.path.basename(label_path)}"
            )
            axes[2].axis("off")

            # Parse and visualize keypoints from label
            if labels and len(labels) > 0:
                label_data = labels[0]
                if (
                    len(label_data) >= 19
                ):  # class_id + 9*2 keypoints = 19 values
                    try:
                        # Extract keypoints (skip class_id)
                        keypoints = np.array(
                            [float(x) for x in label_data[1:19]]
                        ).reshape(9, 2)

                        # Scale keypoints to image dimensions
                        h, w = rgb_img.shape[:2]
                        keypoints[:, 0] *= w
                        keypoints[:, 1] *= h

                        # Plot keypoints
                        axes[2].scatter(
                            keypoints[:, 0],
                            keypoints[:, 1],
                            c="red",
                            s=30,
                            zorder=3,
                        )

                        # Annotate keypoint indices
                        for i, (x, y) in enumerate(keypoints):
                            axes[2].annotate(
                                str(i),
                                (x, y),
                                xytext=(5, 5),
                                textcoords="offset points",
                                color="white",
                                fontsize=8,
                                weight="bold",
                                bbox=dict(
                                    boxstyle="circle,pad=0.2",
                                    facecolor="red",
                                    alpha=0.7,
                                ),
                            )

                        # Draw connecting lines
                        keypoints_corner = keypoints[1:, :]
                        for edge in edges_corners:
                            if edge[0] < len(keypoints_corner) and edge[
                                1
                            ] < len(keypoints_corner):
                                axes[2].plot(
                                    [
                                        keypoints_corner[edge[0], 0],
                                        keypoints_corner[edge[1], 0],
                                    ],
                                    [
                                        keypoints_corner[edge[0], 1],
                                        keypoints_corner[edge[1], 1],
                                    ],
                                    color="green",
                                    linewidth=2,
                                    alpha=0.8,
                                )
                    except (ValueError, IndexError) as e:
                        print(
                            f"Warning: Could not parse keypoints from {label_path}: {e}"
                        )

            # Convert plot to image
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()

            # Convert RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            # Write frame to video
            video_writer.write_frame(img_array)

            if file_idx % 10 == 0:
                print(f"  Processed {file_idx + 1}/{len(all_files)} files...")

        except Exception as e:
            if e is KeyboardInterrupt:
                raise e
            print(f"Error processing {file_name}: {e}")
            continue

    # Release video writer
    video_writer.close()
    print(f"MP4 visualization saved to: {video_path}")

    # Summary
    print(f"Dataset verification complete for category: {category}")
    print(f"  - Train samples: {len(train_files)}")
    print(f"  - Test samples: {len(test_files)}")
    print(f"  - RGB images: {len(found_rgb_files)}")
    print(f"  - Masks: {len(found_mask_files)}")
    print(f"  - Labels: {len(found_label_files)}")
    print(f"  - Complete samples (all files present): {len(all_files)}")
    print(f"  - Missing components: {len(missing_files)}")
    print(
        f"  - MP4 visualization saved to: {visualize_dir / f'{category}_visualization.mp4'}"
    )

    return True


@otn_manager.NODE.register(name="linemod-combine")
def linemod_combine(
    src_path: str,
    dst_path: str,
    names: str = None,
    combined_name: str = None,
) -> bool:
    """
    Merge two linemod datasets into a single combined dataset.

    Args:
        srcs_path: Path list to combine.
        dst_path: Path to create combined dataset (e.g., /path/to/cup_coaster)
        names: Name of dataset to combine (used for yaml configuration)
        combined_name: Name of combined dataset

    Returns:
        bool: True if successful, False otherwise
    """
    src_path_list = src_path.split(",")
    assert len(src_path_list) > 1
    if not names:
        names = [os.path.basename(p) for p in src_path_list]
    else:
        names = names.split(",")
    assert len(src_path_list) == len(names)

    if not combined_name:
        combined_name = "_".join(names)

    dst_path = Path(dst_path)
    combined_path = dst_path / combined_name

    # Create destination directory structure
    combined_path.mkdir(parents=True, exist_ok=True)
    (combined_path / "JPEGImages").mkdir(exist_ok=True)
    (combined_path / "mask").mkdir(exist_ok=True)
    (combined_path / "labels").mkdir(exist_ok=True)

    # Process file count and copy files
    def get_file_count_and_process(
        src_path: Path, name: str, class_id: int, offset: int
    ) -> int:
        """Process all files for a dataset and return the count"""

        # Read train and test files
        all_files = []
        for split_name in ["train.txt", "test.txt"]:
            src_file = src_path / split_name
            if src_file.exists():
                with open(src_file, "r") as f:
                    all_files.extend(
                        [line.strip() for line in f if line.strip()]
                    )

        # Copy and rename files
        for idx, old_filename in enumerate(all_files):
            new_index = offset + idx
            new_filename = f"{new_index:06d}.png"
            name_without_ext = os.path.splitext(old_filename)[0]

            # Copy RGB image
            src_rgb = src_path / "JPEGImages" / old_filename
            dst_rgb = combined_path / "JPEGImages" / new_filename
            if src_rgb.exists():
                # shutil.copy2(src_rgb, dst_rgb)
                os.system(f"cp -Pf {src_rgb} {dst_rgb}")

            # Copy mask
            src_mask = src_path / "mask" / f"{name_without_ext[2:]}.png"
            if not src_mask.exists():
                src_mask = src_path / "mask" / f"{name_without_ext}.png"

            dst_mask = combined_path / "mask" / f"{new_index:06d}.png"
            if src_mask.exists():
                # shutil.copy2(src_mask, dst_mask)
                os.system(f"cp -Pf {src_mask} {dst_mask}")

            # Process label file
            src_label = src_path / "labels" / f"{name_without_ext}.txt"
            dst_label = combined_path / "labels" / f"{new_index:06d}.txt"
            if src_label.exists():
                with open(src_label, "r") as f:
                    lines = f.readlines()

                # Update class_id in labels
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if (
                        len(parts) >= 20
                    ):  # class_id + 9*2 keypoints + camera params
                        # Change class_id based on dataset
                        new_parts = [str(class_id)] + parts[1:]
                        new_line = " ".join(new_parts) + "\n"
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)

                with open(dst_label, "w") as f:
                    f.writelines(new_lines)

        return len(all_files)

    class_idx = 0
    total_count = 0
    for src, name in zip(src_path_list, names):
        src = Path(src)

        # Validate source paths
        if not src.exists():
            print(f"Error: Source path {src} does not exist")
            return False

        # Check required directories exist in both sources
        required_dirs = ["JPEGImages", "mask", "labels"]
        for dir_name in required_dirs:
            dir_path = src / dir_name
            if not dir_path.exists():
                print(f"Error: Required directory {dir_path} does not exist")
                return False

        # Check yaml files exist
        yaml_path = src / f"{name}.yaml"

        if not yaml_path.exists():
            print(f"Error: YAML file {yaml_path} does not exist")
            return False

        # Read yaml files
        with open(yaml_path, "r") as f:
            yaml1 = yaml.safe_load(f)

        print(f"Processing {name} dataset...")
        count = get_file_count_and_process(src, name, class_idx, total_count)
        class_idx += 1
        total_count += count

    # Create new train/test splits (80/20 split)
    all_indices = list(range(total_count))
    np.random.shuffle(all_indices)

    split_idx = int(0.5 * total_count)
    train_indices = all_indices[:split_idx]
    test_indices = all_indices[split_idx:]

    # Write train.txt and test.txt
    with open(combined_path / "train.txt", "w") as f:
        for idx in sorted(train_indices):
            f.write(f"{idx:06d}.png\n")

    with open(combined_path / "test.txt", "w") as f:
        for idx in sorted(test_indices):
            f.write(f"{idx:06d}.png\n")

    # Create combined YAML file
    combined_yaml = {
        "background_path": yaml1.get("background_path", None),
        "fx": yaml1["fx"],
        "fy": yaml1["fy"],
        "u0": yaml1["u0"],
        "v0": yaml1["v0"],
        "diam": yaml1["diam"],
        "nc": 2,  # Two classes: cup and coaster
        "names": names,
        "mesh": [
            str(combined_path.absolute() / f"{name}.ply") for name in names
        ],
        "train": str(combined_path.absolute() / "train.txt"),
        "test": str(combined_path.absolute() / "test.txt"),
        "val": str(combined_path.absolute() / "test.txt"),
    }

    # Save combined YAML
    with open(combined_path / f"{combined_name}.yaml", "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)

    # Copy ply files and camera json
    for src, name in zip(src_path_list, names):
        src_ply = os.path.join(src, f"{name}.ply")
        dst_ply = os.path.join(combined_path, f"{name}.ply")
        if os.path.exists(src_ply):
            shutil.copy2(src_ply, dst_ply)

        src_json = os.path.join(src, "linemod_camera.json")
        dst_json = os.path.join(combined_path, "linemod_camera.json")
        shutil.copy2(src_json, dst_json)

    print(f"Successfully created combined dataset: {combined_name}")
    print(f"  Total samples: {total_count}")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Testing samples: {len(test_indices)}")
    print(f"  Classes: {combined_yaml['names']}")
    print(f"  Dataset location: {combined_path}")

    return True
