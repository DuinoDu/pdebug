import os
import zipfile

from pdebug.utils.fileio import add_folder_to_zip

import pytest


def test_add_folder_to_zip(tmpdir):
    test_dir = os.path.join(tmpdir, "test")
    os.makedirs(test_dir, exist_ok=True)
    with open(f"{test_dir}/1.txt", "w") as fid:
        fid.write("hello\n")
    with open(f"{test_dir}/2.txt", "w") as fid:
        fid.write("hello\n")

    with zipfile.ZipFile(f"{tmpdir}/test.zip", "w") as fid:
        add_folder_to_zip(fid, test_dir, base_path=tmpdir)

    with zipfile.ZipFile(f"{tmpdir}/test.zip", "r") as fid:
        assert "test/1.txt" in fid.namelist()
        assert "test/2.txt" in fid.namelist()
