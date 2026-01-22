#!/usr/bin/env python3
"""
Trajectory Analysis and State Classification Tool

This script analyzes TUM format trajectory data, segments it based on curvature,
classifies each point's state based on orientation differences, and creates
a visualization of the results.

Format: timestamp x y z qx qy qz qw
States: stop, forward, backward, forward_left, forward_right, backward_left, backward_right

Usage:
    python3 trajectory_analysis.py [OPTIONS]
    
Options:
    --input FILE       Input trajectory file (default: tum.txt)
    --output FILE      Output image file (default: trajectory_analysis.png)
    --states FILE      Output states file (default: trajectory_states.txt)
    --no-viz          Disable visualization
    --skip N          Skip N points between orientation arrows (default: 10)
    --threshold T     Velocity threshold for stop detection (default: 0.1)
    --future N        Number of future points to use for orientation comparison (default: 3)
"""

from sqlite3.dbapi2 import Timestamp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import math
import argparse
import sys
from typing import List, Tuple, Dict
import json
from typing import List, Any
import lance
import pyarrow as pa

# Constants
STATE_THRESHOLD_DEG = 15  # ±15 degrees threshold for state classification
ARROW_LENGTH = 2.0        # Length of orientation arrows in visualization
VELOCITY_THRESHOLD = 0.1  # Velocity threshold for stop detection

class TrajectoryPoint:
    """Represents a single point in the trajectory with state information."""
    
    def __init__(self, timestamp: float, x: float, y: float, qx: float, qy: float, qz: float, qw: float, trigger: str = None):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.trigger = trigger
        self.yaw = None  # Will be calculated from quaternion, yaw in world coordinate system
        self.state = None
        self.future_yaw_diff = None
        
    def calculate_yaw(self):
        """Calculate yaw angle from quaternion."""
        # Convert quaternion to yaw angle (rotation around z-axis)
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        self.yaw = math.atan2(2 * (self.qw * self.qz + self.qx * self.qy), 
                             1 - 2 * (self.qy**2 + self.qz**2))

def read_trajectory_txt(filename: str) -> List[TrajectoryPoint]:
    """
    Read TUM format trajectory file and return list of TrajectoryPoint objects.
    
    Args:
        filename: Path to the trajectory file
        
    Returns:
        List of TrajectoryPoint objects
    """
    points = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        try:
            parts = line.split()
            if len(parts) >= 8:
                timestamp = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])  # Ignored for 2D analysis
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
                
                point = TrajectoryPoint(timestamp, x, y, qx, qy, qz, qw)
                point.calculate_yaw()
                points.append(point)
        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping malformed line {line_num + 1}: {line}")
    
    return points

def read_trajectory_json(filename: str) -> List[TrajectoryPoint]:
    """
    Read JSON format trajectory file and return list of TrajectoryPoint objects.
    
    Args:
        filename: Path to the trajectory file
        
    Returns:
        List of TrajectoryPoint objects
    """
    points = []
    
    with open(filename, 'r') as f:
        data = json.load(f)

    for timestamp, odo in data.items():
        point = TrajectoryPoint(float(timestamp), odo[0], odo[1], odo[3], odo[4], odo[5], odo[6])
        point.calculate_yaw()
        points.append(point)
    return points

def _unwrap_odo(odo_val: Any) -> list[float]:
    if hasattr(odo_val, "as_py"):
        odo_val = odo_val.as_py()

    # [[[...]]] / [[...]] → [...]
    while isinstance(odo_val, (list, tuple)) and len(odo_val) == 1:
        odo_val = odo_val[0]

    if not isinstance(odo_val, (list, tuple)):
        raise TypeError(f"Invalid odo type: {type(odo_val)}")

    return list(odo_val)

def read_trajectory_lance(dataset_uri: str, override: bool = False) -> Tuple[List[TrajectoryPoint], bool]:
    ds = lance.dataset(dataset_uri)
    columns = ["timestamp", "odo", "trigger"]
    
    # Check if action_tag exists
    has_action_tag = "action_tag" in ds.schema.names
    load_existing = has_action_tag and not override
    
    if load_existing:
        columns.append("action_tag")
        print(f"Found existing 'action_tag' column, loading from dataset...")

    scanner = ds.scanner(columns=columns)

    points: List[TrajectoryPoint] = []

    for batch in scanner.to_batches():
        ts_col = batch["timestamp"]
        odo_col = batch["odo"]
        trigger_col = batch["trigger"]
        tag_col = batch["action_tag"] if load_existing else None

        for i in range(batch.num_rows):
            # timestamp: int64 ns
            t_ns = ts_col[i]
            t_sec = float(t_ns.as_py()) * 1e-9

            # odo: list<list<list<double>>>
            odo = _unwrap_odo(odo_col[i])

            # trigger: string or nested list of strings
            trigger = trigger_col[i].as_py()
            while isinstance(trigger, (list, tuple)) and len(trigger) > 0:
                trigger = trigger[0]
            
            if trigger is not None and not isinstance(trigger, str):
                trigger = str(trigger)

            if len(odo) < 7:
                raise ValueError(f"odo length < 7: {odo}")

            p = TrajectoryPoint(
                t_sec,
                odo[0],
                odo[1],
                odo[3],
                odo[4],
                odo[5],
                odo[6],
                trigger=trigger
            )
            p.calculate_yaw()
            
            if load_existing:
                p.state = tag_col[i].as_py()
                
            points.append(p)

    return points, load_existing

def write_states_to_lance(points: List[TrajectoryPoint], input_uri: str, output_uri: str) -> None:
    """
    Write classification results to a Lance dataset using streaming to save memory.
    """
    import pyarrow as pa
    import lance
    import os
    
    ds = lance.dataset(input_uri)
    action_tags = [p.state for p in points]
    
    if len(action_tags) != ds.count_rows():
        print(f"Error: Point count ({len(action_tags)}) does not match dataset row count ({ds.count_rows()})")
        return

    # Prepare the schema for the new dataset
    schema = ds.schema
    if "action_tag" in schema.names:
        # Remove existing action_tag from schema fields to avoid duplicates
        fields = [f for f in schema if f.name != "action_tag"]
        schema = pa.schema(fields)
    
    # Append the new action_tag field to the schema
    new_schema = schema.append(pa.field("action_tag", pa.string()))

    def updated_batches():
        curr_idx = 0
        # Use a small batch_size to avoid "byte array offset overflow" (2GB limit per RecordBatch)
        # This is especially important if the dataset contains large images or masks
        for batch in ds.scanner(batch_size=100).to_batches():
            batch_len = batch.num_rows
            batch_tags = action_tags[curr_idx : curr_idx + batch_len]
            
            # If action_tag already exists in this batch, remove it
            if "action_tag" in batch.schema.names:
                idx = batch.schema.names.index("action_tag")
                batch = batch.remove_column(idx)
            
            new_batch = batch.append_column("action_tag", pa.array(batch_tags))
            yield new_batch
            curr_idx += batch_len

    # Write the dataset. mode="overwrite" handles both new files and in-place version updates.
    lance.write_dataset(updated_batches(), output_uri, mode="overwrite", schema=new_schema)
    print(f"Successfully wrote states to Lance dataset at {output_uri} (Streaming mode)")

def classify_state(points: List[TrajectoryPoint], future_points: int = 3) -> None:
    """
    Classify the state of each trajectory point based on orientation differences.
    
    Args:
        points: List of trajectory points to classify
        future_points: Number of future points to use for orientation comparison
    """
    for i, point in enumerate(points):
        # Check if we have enough future points within the same trigger segment
        has_enough_future = True
        if i >= len(points) - future_points:
            has_enough_future = False
        else:
            # Check if all future points up to future_points have the same trigger
            for j in range(1, future_points + 1):
                if points[i + j].trigger != point.trigger:
                    has_enough_future = False
                    break
        
        if not has_enough_future:
            # Not enough future points for classification
            point.state = "stop"
            continue
        
        # Get current orientation
        current_yaw = point.yaw
        
        # Get future point orientation (future_points ahead)
        future_point = points[i + future_points]
        future_yaw = future_point.yaw
        
        # Calculate orientation difference
        yaw_diff = future_yaw - current_yaw
        
        # Normalize angle to [-π, π]
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        
        # Convert to degrees
        yaw_diff_deg = math.degrees(yaw_diff)
        point.future_yaw_diff = yaw_diff_deg
        
        # Calculate movement direction based on position change
        if i < len(points) - 1 and points[i+1].trigger == point.trigger:
            dx = points[i+1].x - point.x
            dy = points[i+1].y - point.y
        else:
            dx = 0
            dy = 0
        
        # Calculate velocity magnitude (ignoring z)
        velocity = math.sqrt(dx**2 + dy**2)
        
        # Determine if moving forward or backward
        # Check alignment between movement direction and orientation
        if velocity < VELOCITY_THRESHOLD:  # Threshold for considering as stopped
            point.state = "stop"
        else:
            movement_angle = math.atan2(dy, dx)
            orientation_diff = abs(movement_angle - current_yaw)
            
            # Normalize orientation difference
            if orientation_diff > math.pi:
                orientation_diff = 2 * math.pi - orientation_diff
            
            # Determine forward/backward based on alignment
            is_forward = orientation_diff < math.pi / 2
            
            # Classify based on yaw difference
            if abs(yaw_diff_deg) <= STATE_THRESHOLD_DEG:
                point.state = "forward" if is_forward else "backward"
            elif yaw_diff_deg > STATE_THRESHOLD_DEG:
                point.state = "forward_left" if is_forward else "backward_left"
            else:  # yaw_diff_deg < -STATE_THRESHOLD_DEG
                point.state = "forward_right" if is_forward else "backward_right"

def calculate_angle_statistics(points: List[TrajectoryPoint]) -> Dict:
    """
    Calculate statistics for yaw angles and future yaw differences.
    
    Args:
        points: List of trajectory points
        
    Returns:
        Dictionary containing angle statistics
    """
    # Extract yaw angles (in degrees)
    yaw_angles = [math.degrees(p.yaw) for p in points if p.yaw is not None]
    
    # Extract future yaw differences (in degrees)
    future_yaw_diffs = [p.future_yaw_diff for p in points if p.future_yaw_diff is not None]
    
    stats = {
        'yaw': {
            'values': yaw_angles,
            'mean': np.mean(yaw_angles) if yaw_angles else 0.0,
            'std': np.std(yaw_angles) if yaw_angles else 0.0,
            'min': np.min(yaw_angles) if yaw_angles else 0.0,
            'max': np.max(yaw_angles) if yaw_angles else 0.0,
            'median': np.median(yaw_angles) if yaw_angles else 0.0,
        },
        'future_yaw_diff': {
            'values': future_yaw_diffs,
            'mean': np.mean(future_yaw_diffs) if future_yaw_diffs else 0.0,
            'std': np.std(future_yaw_diffs) if future_yaw_diffs else 0.0,
            'min': np.min(future_yaw_diffs) if future_yaw_diffs else 0.0,
            'max': np.max(future_yaw_diffs) if future_yaw_diffs else 0.0,
            'median': np.median(future_yaw_diffs) if future_yaw_diffs else 0.0,
        }
    }
    
    return stats

def write_states_to_file(points: List[TrajectoryPoint], output_filename: str) -> None:
    """
    Write trajectory points with their states to a file.
    
    Args:
        points: List of classified trajectory points
        output_filename: Output filename for states
    """
    with open(output_filename, 'w') as f:
        f.write("# timestamp x y z qx qy qz qw yaw_deg action_tag future_yaw_diff_deg velocity trigger\n")
        for i, point in enumerate(points):
            # Calculate velocity for this point
            if i < len(points) - 1 and points[i+1].trigger == point.trigger:
                dx = points[i+1].x - point.x
                dy = points[i+1].y - point.y
                velocity = math.sqrt(dx**2 + dy**2)
            else:
                velocity = 0.0
            
            yaw_deg = math.degrees(point.yaw) if point.yaw is not None else 0.0
            future_diff = point.future_yaw_diff if point.future_yaw_diff is not None else 0.0
            trigger_str = point.trigger if point.trigger is not None else "None"
            
            f.write(f"{point.timestamp} {point.x} {point.y} 0.0 "
                   f"{point.qx} {point.qy} {point.qz} {point.qw} "
                   f"{yaw_deg:.6f} {point.state} {future_diff:.6f} {velocity:.6f} {trigger_str}\n")

def visualize_trajectory(points: List[TrajectoryPoint], output_filename: str, skip_points: int, future_points: int) -> None:
    """
    Create visualization of trajectory with states and orientation information.
    
    Args:
        points: List of classified trajectory points
        output_filename: Output image filename
        skip_points: Number of points to skip between orientation arrows
        future_points: Number of future points used for orientation comparison
    """
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, width_ratios=[2, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])  # Trajectory plot spans both columns
    ax2 = fig.add_subplot(gs[1, 0])  # State distribution
    ax3 = fig.add_subplot(gs[1, 1])  # Angle distribution
    
    # Extract coordinates
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    
    # Color mapping for states
    state_colors = {
        'stop': 'red',
        'forward': 'green',
        'backward': 'blue',
        'forward_left': 'orange',
        'forward_right': 'purple',
        'backward_left': 'cyan',
        'backward_right': 'magenta'
    }
    
    # Plot 1: Trajectory with colored states
    ax1.set_title('Trajectory Analysis - State Classification')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot trajectory line in segments to avoid jumps between triggers
    current_segment_x = []
    current_segment_y = []
    last_trigger = points[0].trigger if points else None
    
    first_segment = True
    for p in points:
        if p.trigger != last_trigger:
            if current_segment_x:
                ax1.plot(current_segment_x, current_segment_y, 'k-', alpha=0.3, linewidth=1, label='Trajectory' if first_segment else "")
                first_segment = False
            current_segment_x = [p.x]
            current_segment_y = [p.y]
            last_trigger = p.trigger
        else:
            current_segment_x.append(p.x)
            current_segment_y.append(p.y)
    
    if current_segment_x:
        ax1.plot(current_segment_x, current_segment_y, 'k-', alpha=0.3, linewidth=1, label='Trajectory' if first_segment else "")
    
    # Plot points colored by state
    for point in points:
        color = state_colors.get(point.state, 'black')
        ax1.scatter(point.x, point.y, c=color, s=20, alpha=0.7)
    
    # Add state legend
    for state, color in state_colors.items():
        ax1.scatter([], [], c=color, label=state, s=50)
    ax1.legend()
    
    # Add orientation arrows every skip_points points
    for i in range(0, len(points), skip_points):
        point = points[i]
        # Ensure we have future orientation within the same trigger segment
        has_future = False
        if i < len(points) - future_points:
            future_point = points[i + future_points]
            if future_point.trigger == point.trigger:
                has_future = True
                
        if has_future:
            # Current orientation arrow
            dx1 = ARROW_LENGTH * math.cos(point.yaw)
            dy1 = ARROW_LENGTH * math.sin(point.yaw)
            
            # Future orientation arrow (future_points ahead)
            future_point = points[i + future_points]
            dx2 = ARROW_LENGTH * math.cos(future_point.yaw)
            dy2 = ARROW_LENGTH * math.sin(future_point.yaw)
            
            # Current orientation (blue)
            ax1.arrow(point.x, point.y, dx1, dy1, 
                     head_width=0.5, head_length=0.5, 
                     fc='blue', ec='blue', alpha=0.7, label='Current' if i == 0 else "")
            
            # Future orientation (red)
            ax1.arrow(point.x, point.y, dx2, dy2,
                     head_width=0.5, head_length=0.5,
                     fc='red', ec='red', alpha=0.7, label='Future' if i == 0 else "")
            
            # Add diff annotation
            if point.future_yaw_diff is not None:
                ax1.annotate(f'{point.future_yaw_diff:.1f}°', 
                           (point.x + 1, point.y + 1),
                           fontsize=8, alpha=0.7)
    
    # Plot 2: State distribution over time
    ax2.set_title('State Distribution Over Time')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('State')
    
    # Map states to y-values for plotting
    state_values = {state: idx for idx, state in enumerate(state_colors.keys())}
    y_values = [state_values[p.state] for p in points]
    
    # Create color array for scatter plot
    colors = [state_colors[p.state] for p in points]
    
    ax2.scatter(range(len(points)), y_values, c=colors, s=20, alpha=0.7)
    ax2.set_yticks(list(state_values.values()))
    ax2.set_yticklabels(list(state_values.keys()))
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    state_counts = {}
    for state in state_colors.keys():
        state_counts[state] = sum(1 for p in points if p.state == state)
    
    stats_text = "State Distribution:\n"
    for state, count in state_counts.items():
        if count > 0:
            percentage = (count / len(points)) * 100
            stats_text += f"{state}: {count} ({percentage:.1f}%)\n"
    
    ax1.text(0.02, 0.9, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Add future points info
    ax1.text(0.02, 0.97, f'Future points used: {future_points}', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             fontsize=10)
    
    # Plot 3: Angle distribution histogram
    # Calculate angle statistics
    angle_stats = calculate_angle_statistics(points)
    
    # Plot future yaw difference distribution (more relevant for state classification)
    if angle_stats['future_yaw_diff']['values']:
        future_diffs = angle_stats['future_yaw_diff']['values']
        ax3.hist(future_diffs, bins=50, alpha=0.7, color='orange', edgecolor='black', label='Future Yaw Difference')
        ax3.axvline(angle_stats['future_yaw_diff']['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {angle_stats["future_yaw_diff"]["mean"]:.2f}°')
        ax3.axvline(angle_stats['future_yaw_diff']['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {angle_stats["future_yaw_diff"]["median"]:.2f}°')
        ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero')
        ax3.axvline(STATE_THRESHOLD_DEG, color='blue', linestyle=':', linewidth=1, alpha=0.5, label=f'Threshold: ±{STATE_THRESHOLD_DEG}°')
        ax3.axvline(-STATE_THRESHOLD_DEG, color='blue', linestyle=':', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Future Yaw Difference (degrees)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Future Yaw Difference Distribution')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text_diff = (f"Future Yaw Diff Statistics:\n"
                          f"Mean: {angle_stats['future_yaw_diff']['mean']:.2f}°\n"
                          f"Std: {angle_stats['future_yaw_diff']['std']:.2f}°\n"
                          f"Min: {angle_stats['future_yaw_diff']['min']:.2f}°\n"
                          f"Max: {angle_stats['future_yaw_diff']['max']:.2f}°\n"
                          f"Median: {angle_stats['future_yaw_diff']['median']:.2f}°")
        
        ax3.text(0.98, 0.98, stats_text_diff, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {output_filename}")

def main():
    """Main function to run the trajectory analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trajectory state classification tool')
    parser.add_argument('--input', default='tum.txt', help='Input trajectory file (default: tum.txt)')
    parser.add_argument('--output', default='trajectory_analysis.png', help='Output image file (default: trajectory_analysis.png)')
    parser.add_argument('--states', default='trajectory_states.txt', help='Output states file (default: trajectory_states.txt)')
    parser.add_argument('--override', action='store_true', help='Override existing action_tag in lance dataset')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--skip', type=int, default=100, help='Skip N points between orientation arrows (default: 10) for visualization')
    parser.add_argument('--threshold', type=float, default=0.01, help='Velocity threshold for stop detection (default: 0.1) for classification')
    parser.add_argument('--future', type=int, default=3, help='Number of future points to use for orientation comparison (default: 3) for classification')
    parser.add_argument('--separate-triggers', action='store_true', help='Visualize each trigger segment separately')
    
    args = parser.parse_args()
    
    # Read trajectory data
    print(f"Reading trajectory data from {args.input}...")
    
    loaded_existing = False
    if args.input.endswith('.txt'):
        points = read_trajectory_txt(args.input)
    elif args.input.endswith('.json'):
        points = read_trajectory_json(args.input)
    elif args.input.endswith('.lance'):
        points, loaded_existing = read_trajectory_lance(args.input, args.override)
    else:
        print("Error: Unsupported file type")
        return
    print(f"Loaded {len(points)} trajectory points")
    
    # Update velocity threshold
    global VELOCITY_THRESHOLD
    VELOCITY_THRESHOLD = args.threshold
    
    # Classify states
    if not loaded_existing:
        print("Classifying trajectory states...")
        classify_state(points, args.future)
    else:
        print("Using existing states from dataset...")
    
    # Write states
    if args.states.endswith('.lance'):
        if not args.input.endswith('.lance'):
            print("Error: Writing to lance is only supported when input is also a lance dataset (to preserve other columns)")
        else:
            print(f"Writing states to Lance dataset {args.states}...")
            write_states_to_lance(points, args.input, args.states)
    else:
        print(f"Writing states to {args.states}...")
        write_states_to_file(points, args.states)
    
    # Print some statistics
    states = [p.state for p in points]
    unique_states = set(states)
    print(f"Detected states: {sorted(unique_states)}")
    
    for state in sorted(unique_states):
        count = states.count(state)
        print(f"  {state}: {count} points ({count/len(points)*100:.1f}%)")
    
    # Print angle statistics
    print("\nAngle Statistics:")
    angle_stats = calculate_angle_statistics(points)
    
    print("\nYaw Angle Statistics:")
    if angle_stats['yaw']['values']:
        print(f"  Mean: {angle_stats['yaw']['mean']:.2f}°")
        print(f"  Std: {angle_stats['yaw']['std']:.2f}°")
        print(f"  Min: {angle_stats['yaw']['min']:.2f}°")
        print(f"  Max: {angle_stats['yaw']['max']:.2f}°")
        print(f"  Median: {angle_stats['yaw']['median']:.2f}°")
    
    print("\nFuture Yaw Difference Statistics:")
    if angle_stats['future_yaw_diff']['values']:
        print(f"  Mean: {angle_stats['future_yaw_diff']['mean']:.2f}°")
        print(f"  Std: {angle_stats['future_yaw_diff']['std']:.2f}°")
        print(f"  Min: {angle_stats['future_yaw_diff']['min']:.2f}°")
        print(f"  Max: {angle_stats['future_yaw_diff']['max']:.2f}°")
        print(f"  Median: {angle_stats['future_yaw_diff']['median']:.2f}°")
        
        # Count points in different angle ranges
        within_threshold = sum(1 for d in angle_stats['future_yaw_diff']['values'] if abs(d) <= STATE_THRESHOLD_DEG)
        left_turn = sum(1 for d in angle_stats['future_yaw_diff']['values'] if d > STATE_THRESHOLD_DEG)
        right_turn = sum(1 for d in angle_stats['future_yaw_diff']['values'] if d < -STATE_THRESHOLD_DEG)
        total = len(angle_stats['future_yaw_diff']['values'])
        
        print(f"\nAngle-based Classification:")
        print(f"  Straight (|diff| <= {STATE_THRESHOLD_DEG}°): {within_threshold} points ({within_threshold/total*100:.1f}%)")
        print(f"  Left turn (diff > {STATE_THRESHOLD_DEG}°): {left_turn} points ({left_turn/total*100:.1f}%)")
        print(f"  Right turn (diff < -{STATE_THRESHOLD_DEG}°): {right_turn} points ({right_turn/total*100:.1f}%)")
    
    # Create visualization if not disabled
    if not args.no_viz:
        if args.separate_triggers:
            # Group points by trigger
            from collections import defaultdict
            trigger_groups = defaultdict(list)
            for p in points:
                trigger_groups[p.trigger].append(p)
            
            print(f"Creating separate visualizations for {len(trigger_groups)} trigger segments...")
            for trigger, group_points in trigger_groups.items():
                trigger_str = str(trigger) if trigger is not None else "None"
                # Filter out illegal filename characters
                safe_trigger = "".join([c for c in trigger_str if c.isalnum() or c in (' ', '.', '_')]).strip()
                safe_trigger = safe_trigger.replace(' ', '_')
                
                output_base = args.output.rsplit('.', 1)
                trigger_output = f"{output_base[0]}_{safe_trigger}.{output_base[1]}"
                
                print(f"  Visualizing trigger '{trigger_str}' -> {trigger_output}")
                visualize_trajectory(group_points, trigger_output, args.skip, args.future)
        else:
            print(f"Creating visualization as {args.output}...")
            visualize_trajectory(points, args.output, args.skip, args.future)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()