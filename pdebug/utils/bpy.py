from .env import BLENDER_INSTALLED

if BLENDER_INSTALLED:
    import bpy

__all__ = ["load_gltf"]


def load_gltf(filepath: str, location=None, name="ImportedGLTFParent"):
    if not BLENDER_INSTALLED:
        raise RuntimeError("Please run in blender.")

    before_import = set(bpy.context.scene.objects.keys())
    bpy.ops.import_scene.gltf(filepath=filepath)
    after_import = set(bpy.context.scene.objects.keys())

    new_objects = after_import - before_import
    root_objects = [
        bpy.context.scene.objects[obj]
        for obj in new_objects
        if bpy.context.scene.objects[obj].parent is None
    ]

    if not root_objects:
        print("Not root object found in gltf.")
        return

    if location:
        # create a parent object
        parent_object = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(parent_object)

        # put all gltf object to parent object
        for root_obj in root_objects:
            root_obj.parent = parent_object

        # move parent object
        parent_object.location = location
