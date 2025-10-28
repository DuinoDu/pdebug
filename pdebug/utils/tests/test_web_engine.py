from pdebug.utils.env import VISDOM_INSTALLED
from pdebug.utils.web_engine import ImageEngine, ThreeEngine

import numpy as np
import pytest


def test_three_engine():
    js_code = """
const dir = new THREE.Vector3( 1, 2, 0  );
dir.normalize();

const origin = new THREE.Vector3( 0, 0, 0  );
const length = 1;
const color = 0xffff00;

const arrowHelper = new THREE.ArrowHelper( dir, origin, length, color);
scene.add( arrowHelper  );
"""
    engine = ThreeEngine("simple")
    engine.render(js_code=js_code)
    # engine.serve(port=5160)


def test_image_engine(tmpdir):
    engine = ImageEngine(res_dir=tmpdir)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    engine.add_image(image)
    # engine.serve(port=5160)


@pytest.mark.skipif(True, reason="skipped")
def test_image_engine_sync():
    engine = ImageEngine(serve_first=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    engine.add_image(image)
    # engine.serve(port=5160)


def test_parallel():
    engine = ImageEngine()

    from pdebug.utils.decorator import mp

    @mp(nums=4)
    def _process(process_id, r):
        for i in r:
            print(f"{i} / 8")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            image[:, :, 0] = 100 + i * 5
            engine.add_image(image, prefix=f"{process_id}_{i:06d}")

    _process(list(range(8)))
    # engine.serve(port=5160)
