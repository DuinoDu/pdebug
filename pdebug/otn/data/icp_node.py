#!/usr/bin/env python3
"""
ICP (Iterative Closest Point) 点云匹配算法Demo
使用Open3D库实现点云配准
"""
import copy
import time
from pathlib import Path
from typing import Optional

from pdebug.data_types import PointcloudTensor
from pdebug.otn import manager as otn_manager
from pdebug.utils.env import OPEN3D_INSTALLED

import matplotlib.pyplot as plt
import numpy as np
import typer

if OPEN3D_INSTALLED:
    import open3d as o3d

__all__ = ["run_icp_registration", "apply_transformation"]


def create_sample_point_clouds():
    """
    使用Open3D官方兔子数据创建示例点云
    返回源点云和目标点云
    """
    print("加载Open3D官方兔子数据...")

    # 下载并加载兔子点云数据
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)

    # 将网格转换为点云
    source_pcd = mesh.sample_points_uniformly(number_of_points=3000)

    # 创建目标点云（对源点云进行变换）
    # 1. 旋转
    angle = np.pi / 6  # 30度
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array(
        [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
    )

    # 2. 平移
    translation = np.array([0.1, 0.05, 0.02])

    # 3. 应用变换
    target_pcd = copy.deepcopy(source_pcd)
    target_pcd.transform(np.eye(4))  # 先重置变换

    # 创建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

    target_pcd.transform(transform_matrix)

    # 4. 添加噪声
    points = np.asarray(target_pcd.points)
    noise = np.random.normal(0, 0.005, points.shape)
    target_pcd.points = o3d.utility.Vector3dVector(points + noise)

    print(f"源点云点数: {len(source_pcd.points)}")
    print(f"目标点云点数: {len(target_pcd.points)}")

    return source_pcd, target_pcd


def visualize_point_clouds(source_pcd, target_pcd, title="点云可视化"):
    """
    可视化点云
    """
    # 设置颜色
    source_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 源点云
    target_pcd.paint_uniform_color([0, 1, 0])  # 绿色 - 目标点云

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)

    # 添加几何体
    vis.add_geometry(source_pcd)
    vis.add_geometry(target_pcd)

    # 设置视角
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    # 运行可视化
    vis.run()
    vis.destroy_window()


def run_icp_registration(
    source_pcd,
    target_pcd,
    method="point_to_point",
    max_iteration=50,
    distance_threshold=0.05,
    verbose=False,
):
    """
    运行ICP配准算法

    Args:
        source_pcd: 源点云
        target_pcd: 目标点云
        method: ICP方法 ("point_to_point", "point_to_plane", "generalized")

    Returns:
        transformation_matrix: 变换矩阵
        fitness: 配准质量
        inlier_rmse: 均方根误差
    """
    if verbose:
        print(f"开始ICP配准 (方法: {method})...")

    if isinstance(source_pcd, PointcloudTensor):
        source_pcd = source_pcd.to_open3d()
    if isinstance(target_pcd, PointcloudTensor):
        target_pcd = target_pcd.to_open3d()

    start_time = time.time()

    if method == "point_to_point":
        # 点到点ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            distance_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
    elif method == "point_to_plane":
        # 点到平面ICP（需要法向量）
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )

        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            distance_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
    elif method == "generalized":
        # 广义ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            distance_threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
    else:
        raise ValueError(f"不支持的ICP方法: {method}")

    end_time = time.time()
    if verbose:
        print(f"ICP配准完成!")
        print(f"运行时间: {end_time - start_time:.3f} 秒")
        print(f"配准质量 (fitness): {result.fitness:.6f}")
        print(f"均方根误差 (RMSE): {result.inlier_rmse:.6f}")
        print(f"变换矩阵:\n{result.transformation}")

    return result.transformation, result.fitness, result.inlier_rmse


def apply_transformation(pcd, transformation_matrix):
    """
    对点云应用变换矩阵
    """
    if isinstance(pcd, PointcloudTensor):
        transformed_pcd = pcd.to_open3d()
    else:
        transformed_pcd = copy.deepcopy(pcd)
    transformed_pcd.transform(transformation_matrix)
    if isinstance(pcd, PointcloudTensor):
        return PointcloudTensor.from_open3d(transformed_pcd)
    else:
        return transformed_pcd


def compare_methods(source_pcd, target_pcd, output, vis=False):
    """
    比较不同ICP方法的性能
    """
    methods = ["point_to_point", "point_to_plane", "generalized"]
    results = {}

    print("\n" + "=" * 50)
    print("比较不同ICP方法的性能")
    print("=" * 50)

    for method in methods:
        print(f"\n测试方法: {method}")
        try:
            transformation, fitness, rmse = run_icp_registration(
                source_pcd, target_pcd, method, verbose=True
            )
            results[method] = {
                "transformation": transformation,
                "fitness": fitness,
                "rmse": rmse,
            }
        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            results[method] = None

    # 可视化结果
    visualize_results(source_pcd, target_pcd, results, output)

    return results


def visualize_results(source_pcd, target_pcd, results, output):
    """
    可视化不同方法的结果
    """
    # 创建子图
    fig = plt.figure(figsize=(15, 10))

    # 原始点云
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    ax1.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        c="red",
        s=1,
        label="Source Point Cloud",
    )
    ax1.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],
        c="green",
        s=1,
        label="Target Point Cloud",
    )
    ax1.set_title("Original Point Clouds")
    ax1.legend()

    # 显示每种方法的结果
    for i, (method, result) in enumerate(results.items()):
        if result is None:
            continue

        ax = fig.add_subplot(2, 3, i + 2, projection="3d")

        # 应用变换
        transformed_source = apply_transformation(
            source_pcd, result["transformation"]
        )
        transformed_points = np.asarray(transformed_source.points)

        ax.scatter(
            transformed_points[:, 0],
            transformed_points[:, 1],
            transformed_points[:, 2],
            c="blue",
            s=1,
            label="Transformed Source",
        )
        ax.scatter(
            target_points[:, 0],
            target_points[:, 1],
            target_points[:, 2],
            c="green",
            s=1,
            label="Target Point Cloud",
        )
        ax.set_title(
            f'{method}\nFitness: {result["fitness"]:.4f}\nRMSE: {result["rmse"]:.4f}'
        )
        ax.legend()

    plt.savefig(output / "icp_results.png")


@otn_manager.NODE.register(name="icp")
def main(
    src_path: Optional[str] = None,
    dst_path: Optional[str] = None,
    output: Optional[str] = "tmp_icp_output",
    vis: bool = False,
    unittest: bool = False,
):
    """ICP"""
    typer.echo(typer.style(f"hello, tool", fg=typer.colors.GREEN))

    if unittest:
        source_pcd, target_pcd = create_sample_point_clouds()
    else:
        # 处理自定义点云文件
        if src_path and dst_path:
            print(f"加载源点云: {src_path}")
            source_pcd = o3d.io.read_point_cloud(src_path)

            print(f"加载目标点云: {dst_path}")
            target_pcd = o3d.io.read_point_cloud(dst_path)

            if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
                typer.echo(typer.style("错误: 无法加载点云文件", fg=typer.colors.RED))
                return

            print(f"源点云点数: {len(source_pcd.points)}")
            print(f"目标点云点数: {len(target_pcd.points)}")
        else:
            typer.echo(typer.style("请提供源点云和目标点云文件路径", fg=typer.colors.YELLOW))
            typer.echo("使用 --unittest 运行内置测试")
            return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # 2. 可视化原始点云
    if vis:
        print("\n显示原始点云...")
        visualize_point_clouds(source_pcd, target_pcd, "原始点云")

    # 3. 运行ICP配准
    print("\n运行ICP配准...")
    result = run_icp_registration(
        source_pcd, target_pcd, "point_to_point", verbose=True
    )
    transformation, fitness, rmse = result

    # 4. 应用变换并可视化结果
    transformed_source = apply_transformation(source_pcd, transformation)
    if vis:
        visualize_point_clouds(transformed_source, target_pcd, "ICP配准结果")

    # 5. 比较不同方法
    compare_methods(source_pcd, target_pcd, output, vis)

    # 6. 保存变换矩阵（如果指定了输出路径）
    print(f"\n保存变换矩阵到: {output}")
    # 保存变换矩阵为JSON格式
    import json

    transform_dict = {
        "transformation_matrix": transformation.tolist(),
        "shape": list(transformation.shape),
        "fitness": float(fitness),
        "rmse": float(rmse),
        "method": "point_to_point",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output / "transform.json", "w", encoding="utf-8") as f:
        json.dump(transform_dict, f, indent=2, ensure_ascii=False)

    typer.echo(typer.style(f"变换矩阵已保存到: {output}", fg=typer.colors.GREEN))
    print(f"变换矩阵形状: {transformation.shape}")
    print(f"配准质量: {fitness:.6f}")
    print(f"均方根误差: {rmse:.6f}")

    print("\nDemo完成!")


@otn_manager.NODE.register(name="icp_apply")
def icp_apply_main(
    src_path: str,
    rt_path: str,
    output: Optional[str] = "tmp_icp_apply.ply",
):
    """
    加载变换矩阵并应用到源点云

    Args:
        src_path: 源点云文件路径
        rt_path: 变换矩阵文件路径
        output: 输出点云文件路径（可选）
    """
    # 加载源点云
    print(f"加载源点云: {src_path}")
    source_pcd = o3d.io.read_point_cloud(src_path)

    if len(source_pcd.points) == 0:
        typer.echo(typer.style("错误: 无法加载源点云文件", fg=typer.colors.RED))
        return

    # 加载变换矩阵
    print(f"加载变换矩阵: {rt_path}")
    try:
        if rt_path.endswith(".json"):
            # 加载JSON格式的变换矩阵
            import json

            with open(rt_path, "r", encoding="utf-8") as f:
                transform_data = json.load(f)

            transform_matrix = np.array(
                transform_data["transformation_matrix"]
            )
            print(f"变换矩阵形状: {transform_matrix.shape}")
            print(f"配准质量: {transform_data.get('fitness', 'N/A')}")
            print(f"均方根误差: {transform_data.get('rmse', 'N/A')}")
            print(f"配准方法: {transform_data.get('method', 'N/A')}")
            print(f"时间戳: {transform_data.get('timestamp', 'N/A')}")
        else:
            # 加载numpy格式的变换矩阵
            transform_matrix = np.load(rt_path)
            print(f"变换矩阵形状: {transform_matrix.shape}")
            print(f"变换矩阵:\n{transform_matrix}")
    except Exception as e:
        typer.echo(typer.style(f"错误: 无法加载变换矩阵文件: {e}", fg=typer.colors.RED))
        return

    # 应用变换
    print("应用变换矩阵...")
    transformed_pcd = source_pcd.copy()
    transformed_pcd.transform(transform_matrix)

    # 可视化结果
    print("显示变换结果...")
    source_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 原始点云
    transformed_pcd.paint_uniform_color([0, 0, 1])  # 蓝色 - 变换后点云

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="变换结果", width=800, height=600)
    vis.add_geometry(source_pcd)
    vis.add_geometry(transformed_pcd)
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    print("红色: 原始点云")
    print("蓝色: 变换后点云")
    print("按 'q' 退出可视化")

    vis.run()
    vis.destroy_window()

    # 保存结果（如果指定了输出路径）
    if output:
        print(f"保存变换后的点云到: {output}")
        o3d.io.write_point_cloud(output, transformed_pcd)
        typer.echo(typer.style(f"结果已保存到: {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
