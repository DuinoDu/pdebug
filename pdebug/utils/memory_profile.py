import gc
import sys

__all__ = ["show_memory"]


def show_memory():
    print("*" * 60)
    objects_list = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        # size_mb = size / 1024 / 1024
        # if size_mb < 1: continue
        objects_list.append((obj, size))

    sorted_values = sorted(objects_list, key=lambda x: x[1], reverse=True)

    for obj, size in sorted_values[:10]:
        print(
            f"OBJ: {id(obj)},"
            f"TYPE: {type(obj)},"
            f"SIZE: {size/1024/1024:.2f}MB,"
            f"REPR: {str(obj)[:100]}"
        )
