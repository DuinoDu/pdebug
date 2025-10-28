import multiprocessing
import os
import shutil
import socket
import ssl
import tempfile
from abc import ABC
from http import server
from multiprocessing import Process
from pathlib import Path
from typing import Dict, Optional, Tuple

from pdebug.data_types import Tensor
from pdebug.templates import STATIC_ROOT, factory, render
from pdebug.utils.env import VISDOM_INSTALLED
from pdebug.utils.visdom_utils import get_global_visdom

import numpy as np

if VISDOM_INSTALLED:
    import visdom

import cv2

__all__ = ["ThreeEngine", "ImageEngine", "VisdomImageEngine"]


KEY_FILE = os.path.expanduser("~/.cache/pdebug_privatekey.pem")
CERT_FILE = os.path.expanduser("~/.cache/pdebug_cert.pem")


def generate_key_and_cert() -> Tuple[str, str]:
    """Generate key and cert file required by https."""
    if os.path.exists(CERT_FILE):
        return KEY_FILE, CERT_FILE
    tmpdir = os.path.dirname(CERT_FILE)
    os.makedirs(tmpdir, exist_ok=True)
    os.system(f"openssl genrsa > {KEY_FILE}")
    os.system(
        f"openssl req -new -x509 -key {KEY_FILE} -out {CERT_FILE} -days 365"
    )
    assert os.path.exists(CERT_FILE), "Generate certificate file failed."
    return KEY_FILE, CERT_FILE


class WebEngine(ABC):
    def __init__(
        self,
        res_dir: Optional[str] = None,
        serve_first: bool = False,
        serve_kwargs: Optional[Dict] = None,
    ):
        self._res_dir = res_dir if res_dir else tempfile.mkdtemp()
        if not isinstance(self._res_dir, Path):
            self._res_dir = Path(self._res_dir)

        if not os.path.exists(self._res_dir):
            self._res_dir.mkdir(parents=True, exist_ok=True)

        self.serve_first = serve_first
        if self.serve_first:
            self.p = Process(
                target=self.serve, kwargs=serve_kwargs if serve_kwargs else {}
            )
            self.p.start()

    @property
    def res_dir(self) -> Path:
        return self._res_dir

    def serve(
        self, port: int = 5160, timeout: int = 10, use_ssl: bool = False
    ) -> None:
        """Serve result html."""
        if self.serve_first:
            curr_proc = multiprocessing.current_process()
            if curr_proc.name == "MainProcess":
                while self.p.is_alive():
                    pass
                return
            else:
                pass
        old_cwd = os.getcwd()
        os.chdir(self.res_dir)

        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            ip = "0.0.0.0"

        try:
            server_address = (ip, port)
            httpd = server.HTTPServer(
                server_address, server.SimpleHTTPRequestHandler
            )
            if use_ssl:
                keyfile, certfile = generate_key_and_cert()
                httpd.socket = ssl.wrap_socket(
                    httpd.socket,
                    server_side=True,
                    keyfile=keyfile,
                    certfile=certfile,
                    ssl_version=ssl.PROTOCOL_TLSv1_2,
                )
                prefix = "https"
            else:
                prefix = "http"
            print(f"\n>>> serve {self.name} at {prefix}://{ip}:{port}")
            httpd.serve_forever()

        except (SystemExit, KeyboardInterrupt):
            # :( seems useless
            if self.res_dir.exists() and str(self.res_dir).startswith("/tmp"):
                os.system(f"rm -r {self.res_dir}")
        os.chdir(old_cwd)


class ThreeEngine(WebEngine):
    """Simple engine using threejs and jinja2.

    You can use it to inject threejs code into html and run http server.
    """

    name: str = "threejs-engine"

    def __init__(self, name: str, res_dir: Optional[str] = None):
        super(ThreeEngine, self).__init__(res_dir=res_dir)
        assert name in factory, (
            f"{name} not found in templates, " f"only support {factory.keys()}"
        )
        self.template_file = factory[name]

    def render(self, **kwargs) -> None:
        """Render js_code into html template.

        Args:
            js_code: input javascript code str.
        """
        res = render(self.template_file, **kwargs)
        with open(self.res_dir / "index.html", "w") as fid:
            fid.write(res)
        print(f"Render result saved to {self.res_dir}/index.html")
        # copy static folder
        if (self.res_dir / "static").exists():
            os.system(f"rm -r {self.res_dir / 'static'}")
        shutil.copytree(STATIC_ROOT, self.res_dir / "static")


class ImageEngine(WebEngine):
    """Simple image engine server."""

    name: str = "image-engine"

    def __init__(
        self,
        res_dir: Optional[str] = None,
        serve_first: bool = False,
        serve_kwargs: Optional[Dict] = None,
    ):
        super(ImageEngine, self).__init__(
            res_dir=res_dir, serve_first=serve_first, serve_kwargs=serve_kwargs
        )
        self._idx = 0
        print(f"Image will be saved in {self.res_dir}")

        if VISDOM_INSTALLED:
            print(
                "Found visdom installed, please make sure "
                "`visdom` service is running."
            )
            self.vis = visdom.Visdom()

    def add_image(self, image: Tensor, prefix=None):
        """Add image to res_dir."""
        if not prefix:
            prefix = f"{self._idx:06d}"
        savename = self.res_dir / f"{prefix}.jpg"
        cv2.imwrite(str(savename), image)
        if VISDOM_INSTALLED:
            ImageEngine.add_visdom_image(image, text=prefix)
        self._idx += 1

    @staticmethod
    def add_visdom_image(
        image: Tensor, vis: Optional = None, text: Optional[str] = None
    ):
        """Add image to visdom."""
        assert VISDOM_INSTALLED, "Please install visdom."
        if text and image.shape[0] > 100 and image.shape[1] > 100:
            image = cv2.putText(
                image,
                text,
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        # bgr2rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # (h, w, 3) -> (3, h, w)
        image = np.rollaxis(image, 2)
        if vis is None:
            vis = get_global_visdom()
        vis.image(image)
