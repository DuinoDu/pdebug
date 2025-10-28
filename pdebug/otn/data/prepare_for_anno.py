import os
import shutil
import zipfile
from io import StringIO
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.decorator import mp
from pdebug.utils.fileio import add_folder_to_zip, no_print

import tqdm
import typer


def is_valid_zipfile(filename, target_length):
    try:
        with zipfile.ZipFile(filename, "r") as fid:
            return len(fid.namelist()) == target_length
    except Exception as e:
        return False


@otn_manager.NODE.register(name="prepare_for_anno")
def main(
    json_or_dir: str,
    image_root: str,
    output: str = "gathered_for_annotation",
    num_workers: int = 0,
    make_zip: bool = False,
    cache: bool = False,
):
    """Gather for cvat annontation."""
    if os.path.isfile(json_or_dir):
        jsonfiles = [json_or_dir]
    else:
        assert os.path.isdir(json_or_dir)
        jsonfiles = sorted(
            [
                os.path.join(json_or_dir, x)
                for x in sorted(os.listdir(json_or_dir))
                if x.endswith(".json")
            ]
        )
    typer.echo(
        typer.style(f"Found {len(jsonfiles)} jsonfiles", fg=typer.colors.GREEN)
    )
    num_workers = min(int(num_workers), len(jsonfiles))
    os.makedirs(output, exist_ok=True)

    @mp(nums=num_workers)
    def _process(jsonfiles):
        t = tqdm.tqdm(total=len(jsonfiles))
        for jsonfile in jsonfiles:
            t.update()
            # find json and imgdir relation in name
            name = os.path.splitext(os.path.basename(jsonfile))[0]
            name = name.replace("_pred", "")
            if name in image_root:
                imgdir = image_root
            else:
                imgdir = os.path.join(image_root, name)
            assert os.path.exists(imgdir), f"{imgdir} not exists"

            with no_print():
                roidb = Input(jsonfile, name="coco").get_roidb(as_dict=True)

            if make_zip:
                zip_filename = os.path.join(output, name + ".zip")
                if (
                    cache
                    and os.path.exists(zip_filename)
                    and is_valid_zipfile(zip_filename, len(roidb) + 1)
                ):
                    continue
                try:
                    fid = zipfile.ZipFile(zip_filename, "w")
                    for imgname in roidb:
                        image_path = os.path.join(imgdir, imgname)
                        assert os.path.exists(
                            image_path
                        ), f"{image_path} not exists."
                        fid.write(
                            image_path, arcname=os.path.join(name, imgname)
                        )
                    fid.write(jsonfile, arcname=os.path.basename(jsonfile))
                    fid.close()
                except Exception as e:
                    if os.path.exists(zip_filename):
                        os.remove(zip_filename)
                    raise e
                print(
                    f"{len(roidb)} images and {jsonfile} zipped to \n  {zip_filename}"
                )
            else:
                output_i = os.path.join(output, name)
                if (
                    cache
                    and os.path.exists(output_i)
                    and os.listdir(output_i) == len(roidb)
                ):
                    continue
                os.makedirs(output_i, exist_ok=True)
                for imgname in roidb:
                    shutil.copy(
                        os.path.join(imgdir, imgname),
                        os.path.join(output_i, imgname),
                    )
                shutil.copy(
                    jsonfile, os.path.join(output, os.path.basename(jsonfile))
                )
                print(
                    f"{len(roidb)} images and {jsonfile} saved to {output_i}"
                )

    _process(jsonfiles)
    return output


if __name__ == "__main__":
    typer.run(main)
