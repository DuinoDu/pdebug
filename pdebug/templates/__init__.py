import functools
import os

from jinja2 import Environment, PackageLoader, select_autoescape

__all__ = ["factory", "render", "STATIC_ROOT"]


# factory store template path.
#   key: data_convert
#   value: template_path
factory = dict()
template_root = os.path.dirname(__file__)
templates = [x for x in os.listdir(template_root) if x.endswith(".jinja")]
for template in templates:
    name = os.path.splitext(template)[0].split(".")[0]
    factory[name] = template


STATIC_ROOT = os.path.join(template_root, "static")


env = Environment(
    loader=PackageLoader("pdebug", "templates"),
    autoescape=select_autoescape(["jinja"]),
)


PATCH_LIST = ["js_code"]

# patch
def root_render_func(ctx, self, old_func):
    """Fix quatation mark ' bugs in jinja2 render.

    ascii code for ' is &#39

    """
    for ind, item in enumerate(old_func(ctx)):
        if "&#39" in item:
            item = ctx["js_code"]
        yield item


def render(template_file: str, **kwargs) -> str:  # type: ignore
    """Render template_file."""
    template = env.get_template(template_file)
    old_func = template.root_render_func
    template.root_render_func = functools.partial(
        root_render_func, self=template, old_func=old_func
    )
    return template.render(**kwargs)
