import importlib.util
import inspect
import os
from collections import defaultdict
from typing import Any, List, Tuple, Union

import typer
from fvcore.common.registry import Registry as _Registry

__all__ = [
    "check_optional",
    "is_cli_mode",
    "reset_cli_mode",
    "load_extension_nodes",
    "create",
]


_CLI_MODE = False


def check_optional(*value) -> Union[Tuple, Any]:
    """Convert task optional args to default value.

    When call task using api, `Optional` args need to be
    converted to default value manually.
    """
    output = []
    for v in value:
        if isinstance(v, typer.models.OptionInfo):
            output.append(v.default)
        else:
            output.append(v)
    if len(output) == 1:
        return output[0]
    return tuple(output)


def is_cli_mode() -> bool:
    """Return if is in cli mode or not."""
    return _CLI_MODE


def reset_cli_mode():
    """Set _CLI_MODE to False."""
    global _CLI_MODE
    _CLI_MODE = False


class OTNRegistry(_Registry):
    """OTN Registry."""

    @classmethod
    def find_node_from_folder(cls, node_dir, skip=(), strict=True):
        # load from otn folder or cache
        errors = []
        node_files = []
        for path, dirs, files in os.walk(node_dir, followlinks=True):

            skip_this_path = False
            for skip_item in skip:
                if skip_item in os.path.abspath(path):
                    skip_this_path = True
                    break
            if skip_this_path:
                continue

            node_files += [
                os.path.join(path, x)
                for x in files
                if x.endswith(".py")
                and (not x.startswith("."))
                and (not x.startswith("_"))
                and (not x.startswith("test_"))
                and os.path.basename(__file__) not in x
                and x not in skip
                and os.path.join(path, x) not in skip
            ]

        for f in node_files:
            try:
                module_name = os.path.splitext(os.path.basename(f))[0]
                spec = importlib.util.spec_from_file_location(module_name, f)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # cost time
            except Exception as e:
                if strict:
                    raise e
                errors.append((f, e))
        return errors

    def register(self, name: str = None, obj: object = None):
        """
        support name.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                _name = func_or_class.__name__ if name is None else name
                if _name in self:
                    pass
                else:
                    self._do_register(_name, func_or_class)
                return func_or_class

            return deco
        # used as a function call
        name = obj.__name__ if name is None else name
        self._do_register(name, obj)

    def __repr__(self) -> str:
        groups = defaultdict(list)
        for k, v in self._obj_map.items():
            filepath = inspect.getsourcefile(v)
            if v.__doc__ is None:
                print(f"{filepath} has not docstring")
            else:
                dirname = os.path.basename(os.path.dirname(filepath))
                groups[dirname].append(k)

        all_strs = ""
        for g, names in groups.items():
            obj_map = {}
            for n in names:
                try:
                    obj_map[n] = self._obj_map[n].__doc__.split("\n")[0]
                except Exception as e:
                    print(f"{n} has no docstring.")
                    raise e
            try:
                from tabulate import tabulate

                table = tabulate(
                    obj_map.items(),
                    headers=["Names", "Description"],
                    tablefmt="fancy_grid",
                )
            except ImportError:
                table = "\n".join(
                    f"{node_name}: {description}"
                    for node_name, description in obj_map.items()
                )
            all_strs += f"{g} nodes:\n" + table + "\n"
        return all_strs


NODE = OTNRegistry("NODE")

_CORE_MODULES = (
    "pdebug.otn.dag",
    "pdebug.otn.run_shell",
    "pdebug.otn.single_node",
)
_EXTENSION_DIRS = (
    "analysis",
    "data",
    "infer",
    "train",
    "../debug",
)
_EXTENSIONS_LOADED = False
_EXTENSION_LOAD_ERRORS: List[Tuple[str, Exception]] = []


def _load_core_nodes() -> None:
    """Register lightweight built-in nodes required by OTN itself."""
    for module_name in _CORE_MODULES:
        importlib.import_module(module_name)


def load_extension_nodes(strict: bool = False) -> None:
    """Discover optional OTN nodes on demand.

    Most nodes live in modules with heavy optional dependencies. Loading them
    during ``import pdebug.otn`` makes unrelated imports fail when one optional
    dependency is broken, so extension discovery is delayed until a requested
    node is missing from the core registry.
    """
    global _EXTENSIONS_LOADED, _EXTENSION_LOAD_ERRORS
    if _EXTENSIONS_LOADED and not strict:
        return

    errors = []
    base_dir = os.path.dirname(__file__)
    for node_dir in _EXTENSION_DIRS:
        errors.extend(
            NODE.find_node_from_folder(
                os.path.join(base_dir, node_dir), strict=strict
            )
        )
    _EXTENSION_LOAD_ERRORS = errors
    if not strict:
        _EXTENSIONS_LOADED = True


_load_core_nodes()


def create(name: str) -> Any:
    """Create node from OTN."""
    if name not in NODE:
        load_extension_nodes(strict=False)
    if name in NODE:
        return NODE.get(name)
    message = f"Unknown node name: {name}.\nAvailable: {NODE}"
    if _EXTENSION_LOAD_ERRORS:
        skipped = "\n".join(
            f"- {path}: {type(error).__name__}: {error}"
            for path, error in _EXTENSION_LOAD_ERRORS
        )
        message += f"\nSkipped extension modules:\n{skipped}"
    raise ValueError(message)
