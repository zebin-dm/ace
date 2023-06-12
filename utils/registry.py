from typing import Any

from omegaconf.dictconfig import DictConfig


class Registry:
    """
    Registry for any modular functions or methods.
    """

    def __init__(self, name: str) -> None:
        self._dict = {}
        self._name = name

    def __len__(self) -> int:
        return len(self._dict)

    def __getitem__(self, module_name: str) -> Any:
        if module_name in self._dict:
            return self._dict[module_name]
        else:
            raise KeyError(
                f"{module_name} is not registerd in {self._name} yet.")

    def __setitem__(self, module_name: str, module: Any) -> None:
        if module_name in self._dict:
            raise KeyError(
                f"{module_name} has already been registered in {self._name}.")
        self._dict[module_name] = module

    def __contains__(self, module_name: str):
        return self.get(module_name) is not None

    def __repr__(self) -> str:
        msg = "Registered modules:\n"
        msg += ', '.join(list(self._dict.keys()))
        return msg

    @property
    def name(self) -> str:
        return self._name

    def _register_module(self, module: Any) -> Any:
        module_name = module.__name__
        self[module_name] = module
        return module

    def register(self, module=None) -> Any:
        if module is not None:
            return self._register_module(module)

        def _register(module):
            return self._register_module(module)

        return _register

    def build(self, cfg: DictConfig, **kwargs) -> Any:
        if cfg.get("type") is None:
            raise TypeError("cfg must have 'type' for building class.")
        cfg_args = cfg.copy()
        obj_name = cfg_args.pop("type")
        func_args = {**kwargs}
        func_args.update(cfg_args)
        return self[obj_name](**func_args)
