"""
SAM2 segmentation infer node for OnePoseviaGen pipeline.

Depends:
    decode
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.fileio import do_system
from pdebug.visp import draw

import cv2
import numpy as np
import typer


@otn_manager.NODE.register(name="sam2")
def sam2_main(
    input_path: str,
    points: str = None,
    mask_path: str = None,
    output: str = "tmp_sam2",
    vis_output: str = "tmp_sam2_vis",
    checkpoint: str = "large",
    sam2_checkpoint_path: Optional[str] = None,
    repo: str = None,
    cache: bool = True,
):
    """SAM2 video segmentation.

    Args:
        input_path: Path to RGB image folder, or Path to video file.
        points: JSON file containing initial points for SAM2 (format: [[x, y, label], ...])
        mask_path: PNG file containing initial mask for SAM2
        repo: Path to OnePoseviaGen repository
        output: Output directory for masks
        checkpoint: SAM2 model size (tiny, small, base-plus, large)
        sam2_checkpoint_path: Custom path to SAM2 checkpoint
    """

    # Expand paths
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    vis_output = Path(vis_output).expanduser().resolve()
    if repo:
        repo = Path(repo).expanduser().resolve()

    if (not cache) and output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    vis_output.mkdir(parents=True, exist_ok=True)

    try:
        import sam2
    except Exception as e:
        raise ModuleNotFoundError(
            "Install sam2 first. Use install-x to install it."
        )

    from sam2.build_sam import (
        build_sam2_video_predictor,
        build_sam2_video_predictor_hf,
    )

    # Setup SAM2 model
    if sam2_checkpoint_path:
        sam2_checkpoint_path = (
            Path(sam2_checkpoint_path).expanduser().resolve()
        )
    elif repo:
        sam2_checkpoint_path = repo / "checkpoints"
    else:
        sam2_checkpoint_path = None
        # use hf
        pass

    model_id = f"facebook/sam2.1-hiera-{checkpoint}"
    if sam2_checkpoint_path:
        if checkpoint == "tiny":
            sam2_checkpoint = sam2_checkpoint_path / "sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif checkpoint == "small":
            sam2_checkpoint = sam2_checkpoint_path / "sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif checkpoint == "base-plus":
            sam2_checkpoint = (
                sam2_checkpoint_path / "sam2.1_hiera_base_plus.pt"
            )
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif checkpoint == "large":
            sam2_checkpoint = sam2_checkpoint_path / "sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            raise ValueError(f"Invalid checkpoint: {checkpoint}")
        if not sam2_checkpoint.exists():
            raise RuntimeError(f"SAM2 checkpoint not found: {sam2_checkpoint}")

    # Get input
    if input_path.is_dir():
        reader = Input(str(input_path), name="imgdir").get_reader()
        if len(reader) == 0:
            raise RuntimeError("No RGB files found")
        typer.echo(
            typer.style(
                f"Found {len(reader)} RGB files", fg=typer.colors.GREEN
            )
        )
    elif input_path.is_file():
        reader = Input(str(input_path), name="video").get_reader()
        typer.echo(
            typer.style(
                f"Found {len(reader)} frames in video", fg=typer.colors.GREEN
            )
        )
    else:
        raise RuntimeError("No RGB files or video found")

    if cache and len(os.listdir(output)) == len(reader):
        print(f"{output} exists, skip")
        return

    # Initialize SAM2 predictor
    if sam2_checkpoint_path:
        predictor = build_sam2_video_predictor(
            str(model_cfg), str(sam2_checkpoint)
        )
    else:
        predictor = build_sam2_video_predictor_hf(model_id)

    # Setup inference state
    inference_state = predictor.init_state(video_path=str(input_path))
    predictor.reset_state(inference_state)

    if points is not None:
        # Load points
        with open(points, "r") as f:
            points_data = json.load(f)

        # Add points for each object
        for obj_id, obj_points in enumerate(points_data, 1):
            np_points = np.array(
                [[p[0], p[1]] for p in obj_points], dtype=np.float32
            )
            labels = np.array([p[2] for p in obj_points], dtype=np.int32)

            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=np_points,
                labels=labels,
            )

    elif mask_path is not None:
        mask = cv2.imread(mask_path, -1)
        if mask.ndim == 3 and mask[2] > 1:
            mask = mask[:, :, 0]
        assert (
            np.unique(mask).shape[0] == 2
        ), "Not supported more than 1 object mask."
        BACKGROUND_VALUE = 0
        for value in np.unique(mask):
            if value == BACKGROUND_VALUE:
                continue
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=value,
                mask=(mask == value),
            )

    fps = reader.fps if hasattr(reader, "fps") else 10
    vis_writer = Output(
        vis_output / "visualization.mp4", name="video_ffmpeg", fps=fps
    ).get_writer()

    # Propagate masks through video
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state):
        # Create mask for this frame
        mask = np.zeros_like(
            out_mask_logits[0].cpu().numpy().squeeze(), dtype=np.uint8
        )

        for i, out_obj_id in enumerate(out_obj_ids):
            obj_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            mask[obj_mask] = 255

        if input_path.is_dir():
            file_name = Path(reader.imgfiles[out_frame_idx]).stem
        else:
            file_name = f"{out_frame_idx:06d}.png"

        # Save mask
        mask_file = output / file_name
        cv2.imwrite(str(mask_file) + ".png", mask)

        vis_mask = draw.semseg(mask, reader.__next__())
        vis_writer.write_frame(vis_mask)

    typer.echo(
        typer.style(
            f"Saved masks to {output} and {vis_output}", fg=typer.colors.GREEN
        )
    )
    return str(output)


@otn_manager.NODE.register(name="sam_with_prompt")
def sam_with_prompt_main(
    input_path: str,
    output: str = "sam_prompt_output",
    port: int = 6150,
    checkpoint: str = "large",
):
    """SAM2 segmentation with gradio interface for prompt input.

    Args:
        input_path: Path to input image folder
        output: Output directory for masks
        port: Port for gradio server
        checkpoint: SAM2 model size (tiny, small, base-plus, large)
    """

    # Expand paths
    input_path = Path(input_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    # Import gradio and required modules
    try:
        import gradio as gr
        from PIL import Image
    except ImportError:
        raise ModuleNotFoundError(
            "Install gradio and PIL first: pip install gradio pillow"
        )

    # Import SAM2
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ModuleNotFoundError(
            "Install sam2 first. Use install-x to install it."
        )

    # Constants for point visualization (from app.py)
    COLORS = [
        (0, 0, 255),
        (0, 255, 255),
    ]  # BGR: Red for negative, Yellow for positive
    MARKERS = [1, 5]  # Cross for negative, Star for positive
    MARKER_SIZE = 8

    # Global variables for the gradio interface
    predictor = None
    current_image_path = None
    current_image_array = None
    selected_points = []
    objects = {}
    current_point_type = "positive_point"  # Default to positive points

    def initialize_sam():
        """Initialize SAM2 predictor"""
        nonlocal predictor
        if predictor is None:
            typer.echo(
                typer.style(
                    f"Initializing SAM2 with checkpoint: {checkpoint}",
                    fg=typer.colors.BLUE,
                )
            )
            predictor = SAM2ImagePredictor.from_pretrained(
                f"facebook/sam2.1-hiera-{checkpoint}", local_files_only=False
            )
            typer.echo(
                typer.style(
                    "SAM2 predictor initialized successfully!",
                    fg=typer.colors.GREEN,
                )
            )
        return predictor

    def load_first_image():
        """Load the first image from input directory"""
        nonlocal current_image_path, current_image_array

        if input_path.is_dir():
            image_files = (
                Input(input_path, name="imgdir").get_reader().imgfiles
            )
            if image_files:
                current_image_path = str(image_files[0])
            else:
                return None, "No image files found in the specified directory"
        elif input_path.is_file():
            current_image_path = str(input_path)
        else:
            return None, "Invalid input path"

        if current_image_path:
            # Load image
            current_image_array = cv2.imread(current_image_path)
            if current_image_array is None:
                return None, f"Failed to load image: {current_image_path}"

            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            return (
                pil_image,
                f"Loaded image: {Path(current_image_path).name} ({image_rgb.shape[1]}x{image_rgb.shape[0]})",
            )
        else:
            return None, "No image found"

    def segment_with_points(points_input):
        """Segment image with point prompts"""
        nonlocal current_image_path, current_image_array, predictor

        if current_image_path is None or current_image_array is None:
            return None, "Please load an image first"

        # Parse points input (format: "x1,y1,label1;x2,y2,label2;...")
        try:
            points = []
            labels = []

            if points_input.strip():
                point_strs = points_input.strip().split(";")
                for point_str in point_strs:
                    if point_str.strip():
                        parts = point_str.strip().split(",")
                        if len(parts) >= 3:
                            x, y, label = (
                                float(parts[0]),
                                float(parts[1]),
                                int(parts[2]),
                            )
                            points.append([x, y])
                            labels.append(label)

            if not points:
                return (
                    None,
                    "No valid points provided. Format: x1,y1,label1;x2,y2,label2;...",
                )

            # Initialize predictor if needed
            predictor = initialize_sam()

            # Prepare image (convert BGR to RGB)
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)

            # Set image for SAM
            predictor.set_image(image_rgb)

            # Run prediction
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)

            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # Create visualization (don't save here, just display)
            vis_image = image_rgb.copy()

            # Apply mask overlay (light blue with transparency)
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[mask] = [20, 60, 200]  # Light blue
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

            # Draw points with markers like in click-based selection
            for point, label in zip(points, labels):
                cv2.drawMarker(
                    vis_image,
                    (int(point[0]), int(point[1])),
                    COLORS[label],
                    markerType=MARKERS[label],
                    markerSize=MARKER_SIZE,
                    thickness=2,
                )

            # Convert to PIL Image
            vis_pil = Image.fromarray(vis_image)

            result_message = f"Text segmentation completed! Points used: {len(points)}, Best mask score: {scores[best_idx]:.3f}\nUse 'Save Mask' button to save the result."
            return vis_pil, result_message

        except Exception as e:
            return None, f"Error during segmentation: {str(e)}"

    def select_point(evt: gr.SelectData):
        """Handle point selection for SAM (based on app.py implementation)"""
        nonlocal current_image_path, current_image_array, predictor, selected_points, current_point_type

        if current_image_path is None or current_image_array is None:
            return None, "Please load an image first", []

        try:
            # Get click coordinates
            x, y = evt.index[0], evt.index[1]
            label = 1 if current_point_type == "positive_point" else 0

            # Add point to list
            selected_points.append((x, y, label))

            # Initialize predictor if needed
            predictor = initialize_sam()

            # Prepare image (convert BGR to RGB)
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)

            # Set image for SAM
            predictor.set_image(image_rgb)

            # Prepare points for prediction
            points_array = np.array(
                [[p[0], p[1]] for p in selected_points], dtype=np.float32
            )
            labels_array = np.array(
                [p[2] for p in selected_points], dtype=np.int32
            )

            # Run prediction
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # Create visualization
            vis_image = image_rgb.copy()

            # Draw points on display image with proper markers
            for point_x, point_y, point_label in selected_points:
                cv2.drawMarker(
                    vis_image,
                    (int(point_x), int(point_y)),
                    COLORS[point_label],
                    markerType=MARKERS[point_label],
                    markerSize=MARKER_SIZE,
                    thickness=2,
                )
                # Add text label near the point
                point_text = "+" if point_label == 1 else "-"
                cv2.putText(
                    vis_image,
                    point_text,
                    (int(point_x) + 10, int(point_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    COLORS[point_label],
                    2,
                )

            # Draw mask overlay (light blue with transparency)
            if mask is not None:
                overlay = vis_image.copy()
                overlay[mask] = [20, 60, 200]  # Light blue
                vis_image = cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0)

            # Convert to PIL Image
            vis_pil = Image.fromarray(vis_image)

            point_type_str = "positive" if label == 1 else "negative"
            result_message = f"{point_type_str.capitalize()} point added at ({x}, {y}). Total points: {len(selected_points)}, Best mask score: {scores[best_idx]:.3f}"
            return vis_pil, result_message, selected_points

        except Exception as e:
            return (
                None,
                f"Error during point selection: {str(e)}",
                selected_points,
            )

    def set_point_type(point_type: str):
        """Set the current point type for click selection"""
        nonlocal current_point_type
        current_point_type = point_type
        return (
            f"Point type set to: {point_type.replace('_', ' ').capitalize()}"
        )

    def reset_points():
        """Reset all selected points"""
        nonlocal selected_points
        selected_points = []
        if current_image_path and current_image_array is not None:
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return pil_image, "Points reset, showing original image", []
        return None, "No image loaded", []

    def save_final_mask():
        """Save the current mask to output directory"""
        nonlocal current_image_path, current_image_array, predictor, selected_points

        if current_image_path is None or current_image_array is None:
            return "Please load an image first"

        if not selected_points:
            return "Please select at least one point"

        try:
            # Initialize predictor if needed
            predictor = initialize_sam()

            # Prepare image (convert BGR to RGB)
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)

            # Set image for SAM
            predictor.set_image(image_rgb)

            # Prepare points for prediction
            points_array = np.array(
                [[p[0], p[1]] for p in selected_points], dtype=np.float32
            )
            labels_array = np.array(
                [p[2] for p in selected_points], dtype=np.int32
            )

            # Run prediction
            masks, scores, logits = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True,
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # Save mask
            image_stem = Path(current_image_path).stem
            mask_uint8 = (mask * 255).astype(np.uint8)
            output_path = output / f"{image_stem}_mask.png"
            cv2.imwrite(str(output_path), mask_uint8)

            return (
                f"Mask saved to: {output_path} (score: {scores[best_idx]:.3f})"
            )

        except Exception as e:
            return f"Error saving mask: {str(e)}"

    def reset_view():
        """Reset to original image"""
        if current_image_path and current_image_array is not None:
            image_rgb = cv2.cvtColor(current_image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            return pil_image, "View reset to original image"
        return None, "No image loaded"

    # Create Gradio interface
    with gr.Blocks(title="SAM2 with Prompt", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 SAM2 Segmentation with Prompt")
        gr.Markdown(f"**Input Directory:** `{input_path}`")
        gr.Markdown(f"**Output Path:** `{output}`")

        # State variables
        points_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Image Loading")
                load_btn = gr.Button(
                    "📂 Load First Image", variant="primary", size="lg"
                )

                gr.Markdown("### 🎯 Point Selection")

                # Point type selector
                point_type_radio = gr.Radio(
                    choices=["positive_point", "negative_point"],
                    value="positive_point",
                    label="Point Type",
                    info="Positive: what to include, Negative: what to exclude",
                )

                points_input = gr.Textbox(
                    label="Manual Points (format: x1,y1,label1;x2,y2,label2;...)",
                    placeholder="Example: 100,100,1;200,200,0",
                    info="Label 1=foreground (positive), 0=background (negative)",
                    lines=2,
                )

                with gr.Row():
                    segment_btn = gr.Button(
                        "🚀 Run Text Segmentation", variant="secondary"
                    )
                    save_btn = gr.Button("💾 Save Mask", variant="primary")

                with gr.Row():
                    reset_points_btn = gr.Button(
                        "🔄 Reset Points", variant="secondary"
                    )
                    reset_btn = gr.Button("🔄 Reset View", variant="secondary")

                status_text = gr.Textbox(
                    label="📋 Status", interactive=False, lines=4
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Interactive Image")
                gr.Markdown(
                    "**Click on the image to add points for segmentation**"
                )

                interactive_image = gr.Image(
                    label="Click to Select Points",
                    type="pil",
                    height=600,
                    interactive=True,
                )

        # Instructions
        with gr.Accordion("📖 Instructions", open=False):
            gr.Markdown(
                """
            ### How to use:
            1. **Load Image**: Click 'Load First Image' to load the first image from the input directory
            2. **Select Point Type**: Choose between Positive (include) or Negative (exclude) points
            3. **Select Points**:
               - **Click Method**: Click directly on the image to add points of the selected type
               - **Text Method**: Enter point coordinates using format: `x1,y1,label1;x2,y2,label2;...`
            4. **Save Mask**: Click 'Save Mask' to save the segmentation result
            5. **Reset**: Use 'Reset Points' to clear all points, or 'Reset View' to show original image

            ### Point Types & Markers:
            - **Positive Points** (include): Yellow star marker ⭐ with '+' label
            - **Negative Points** (exclude): Red cross marker ❌ with '-' label

            ### Examples:
            - Single foreground point: `150,200,1`
            - Foreground + background: `150,200,1;300,100,0`
            - Multiple points: `100,100,1;200,200,1;50,300,0`

            ### Features:
            - **Real-time segmentation**: Mask updates automatically when you click
            - **Visual feedback**: Points are marked with colored markers
            - **Interactive**: Click anywhere on the image to add points
            - **Flexible input**: Support both click-based and text-based point input
            """
            )

        # Event handlers
        load_btn.click(
            fn=load_first_image, outputs=[interactive_image, status_text]
        )

        # Point type selection
        point_type_radio.change(
            fn=set_point_type, inputs=[point_type_radio], outputs=[status_text]
        )

        # Click-based point selection
        interactive_image.select(
            fn=select_point,
            outputs=[interactive_image, status_text, points_state],
        )

        # Text-based segmentation
        segment_btn.click(
            fn=segment_with_points,
            inputs=[points_input],
            outputs=[interactive_image, status_text],
        )

        # Save mask
        save_btn.click(fn=save_final_mask, outputs=[status_text])

        # Reset points
        reset_points_btn.click(
            fn=reset_points,
            outputs=[interactive_image, status_text, points_state],
        )

        # Reset view
        reset_btn.click(
            fn=reset_view, outputs=[interactive_image, status_text]
        )

        # Add some styling
        demo.css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        """

    # Launch gradio server
    typer.echo(
        typer.style(
            f"🚀 Starting Gradio server on port {port}...", fg=typer.colors.BLUE
        )
    )
    typer.echo(
        typer.style(f"📁 Input path: {input_path}", fg=typer.colors.GREEN)
    )
    typer.echo(typer.style(f"💾 Output path: {output}", fg=typer.colors.GREEN))
    typer.echo(
        typer.style(
            f"🌐 Open your browser at: http://localhost:{port}",
            fg=typer.colors.CYAN,
        )
    )

    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_api=False,
            quiet=False,
        )
        return str(output)
    except KeyboardInterrupt:
        typer.echo(
            typer.style("Server stopped by user", fg=typer.colors.YELLOW)
        )
        return str(output)
    except Exception as e:
        typer.echo(
            typer.style(
                f"Error running gradio server: {e}", fg=typer.colors.RED
            )
        )
        raise


if __name__ == "__main__":
    typer.run(sam2_main)
