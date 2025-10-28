from fvcore.common.registry import Registry as _Registry

__all__ = ["ROIDB_REGISTRY", "SOURCE_REGISTRY", "OUTPUT_REGISTRY"]


class Registry(_Registry):
    """
    Registry plus name.
    """

    def register(self, name: str = None, obj: object = None):
        """
        support name.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                _name = func_or_class.__name__ if name is None else name
                self._do_register(_name, func_or_class)
                return func_or_class

            return deco
        # used as a function call
        name = obj.__name__ if name is None else name
        self._do_register(name, obj)


ROIDB_REGISTRY = Registry("ROIDB")
SOURCE_REGISTRY = Registry("SOURCE")
OUTPUT_REGISTRY = Registry("OUTPUT")
