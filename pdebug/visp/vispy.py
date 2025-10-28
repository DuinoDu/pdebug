"""
WARNING: If you get error message like:
    OMP: Error #179: Function pthread_mutex_init failed:
    OMP: System error #22: Invalid argument

Then you need to set environment variables:
    >> export OMP_NUM_THREADS=1; export KMP_DUPLICATE_LIB_OK=TRUE
"""
import time

from pdebug.utils.env import VISPY_INSTALLED

import numpy as np

if VISPY_INSTALLED:
    import vispy
    from vispy import scene
    from vispy.io import write_png
else:
    vispy = None


__all__ = ["DDDRenderer"]


class DDDRenderer:

    """
    Example:
        >>> from pdebug.visp.vispy import DDDRenderer
        >>> show_window = True
        >>> renderer = DDDRenderer(1024, 768, show=show_window)
        >>>
        >>> n_points = 1000
        >>> points = np.random.randn(n_points, 3)
        >>> colors = np.random.rand(n_points, 4)
        >>> renderer.add_pointcloud(points, colors)
        >>>
        >>> renderer.set_camera_params(
        >>>         elevation=-74,
        >>>         azimuth=-0.5,
        >>>         distance=1.05,
        >>>         center=(0., 0., 0.),
        >>>         fov=60
        >>> )
        >>>
        >>> image = renderer.save_image('pointcloud_render.png')
        >>> print(f"Render finished, image size: {image.shape}")
        >>>
        >>> if show_window:
        >>>     vispy.app.run()
    """

    def __init__(self, width=800, height=600, show=False):
        if vispy is None:
            raise RuntimeError(
                "vispy is required. Please install by `pip3 install vispy`"
            )
        self.width = width
        self.height = height

        # 创建离屏canvas
        self.canvas = scene.SceneCanvas(
            size=(width, height), show=show, bgcolor="black"
        )

        self.view = self.canvas.central_widget.add_view()

        # 设置相机
        self.camera = scene.cameras.TurntableCamera(
            elevation=30, azimuth=45, distance=10, fov=60
        )
        self.view.camera = self.camera

        if show:
            # 连接事件
            self.canvas.events.mouse_move.connect(self.on_mouse_move)
            self.canvas.events.mouse_press.connect(self.on_mouse_press)
            self.canvas.events.mouse_release.connect(self.on_mouse_release)
            # 状态变量
            self.is_dragging = False
            self.last_update_time = 0
            self.update_interval = 0.1  # 更新间隔（秒）
            self.update_camera_info()

            # timer = vispy.app.Timer(interval=self.update_interval, connect=self.update_camera_info)
            # timer.start()

    def add_pointcloud(self, points, colors=None):
        """添加点云数据"""
        if colors is None:
            colors = np.ones((len(points), 4)) * 0.7  # 灰色

        scatter = scene.visuals.Markers()
        scatter.set_data(points, face_color=colors, size=3)
        self.view.add(scatter)

    def add_mesh(self, vertices, faces, colors=None):
        """添加网格数据"""
        if colors is None:
            colors = "gray"

        mesh = scene.visuals.Mesh(vertices, faces, color=colors)
        self.view.add(mesh)

    def set_camera_params(
        self, elevation=30, azimuth=45, distance=10, fov=60, center=None
    ):
        """设置相机参数"""
        if center is not None:
            self.camera.center = center
        self.camera.elevation = elevation
        self.camera.azimuth = azimuth
        self.camera.distance = distance
        self.camera.fov = fov

    def render(self):
        """渲染并返回图像"""
        self.canvas.update()
        image = self.canvas.render()
        return image

    def save_image(self, filename):
        """渲染并保存图像"""
        image = self.render()
        write_png(filename, image)
        return image

    def print_camera_params(self):
        """打印当前相机参数到控制台"""
        print("\n=== 当前相机参数 ===")
        print(f"仰角 (elevation): {self.camera.elevation:.2f}")
        print(f"方位角 (azimuth): {self.camera.azimuth:.2f}")
        print(f"距离 (distance): {self.camera.distance:.2f}")
        print(f"中心点 (center): {list(self.camera.center)}")
        print(f"视场角 (fov): {self.camera.fov:.2f}")
        print("==================\n")

    def update_camera_info(self):
        """更新相机参数显示"""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time

        # 获取相机参数
        elevation = self.camera.elevation
        azimuth = self.camera.azimuth
        distance = self.camera.distance
        center = self.camera.center
        fov = self.camera.fov
        print(
            f"""
            renderer.set_camera_params(
                    elevation={elevation:.1f},
                    azimuth={azimuth:.1f},
                    distance={distance:.2f},
                    center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}),
                    fov={fov:.1f}
            )"""
        )

    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if self.is_dragging:
            self.update_camera_info()

    def on_mouse_press(self, event):
        """鼠标按下事件"""
        self.is_dragging = True

    def on_mouse_release(self, event):
        """鼠标释放事件"""
        self.is_dragging = False
        self.update_camera_info()
