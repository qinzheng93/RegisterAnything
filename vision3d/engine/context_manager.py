import os.path as osp
import warnings
from typing import Any, Self

from torch import Tensor


class ContextManager:
    def __init__(self):
        self.data_dict = {}
        self.scope_list: list[str] = []

    def register(self, name: str, value: Any, retain_grad: bool = True) -> None:
        name = osp.join(*self.scope_list, name)
        if isinstance(value, Tensor) and retain_grad:
            if value.requires_grad:
                value.retain_grad()
        self.data_dict[name] = value

    def get(self, name: str | None = None, full_name: str | None = None) -> Any:
        assert name is not None or full_name is not None
        if full_name is None:
            full_name = osp.join(*self.scope_list, name)  # type: ignore
        if full_name not in self.data_dict:
            warnings.warn(f"Key '{full_name}' not found in ContextManager.")
            return None
        return self.data_dict[full_name]

    def clear(self) -> None:
        self.data_dict.clear()
        self.scope_list.clear()

    def enter_scope(self, scope_name: str) -> Self:
        self.scope_list.append(scope_name)
        return self

    def exit_scope(self) -> None:
        if self.scope_list:
            self.scope_list.pop()

    @property
    def scope(self) -> str | None:
        if not self.scope_list:
            return None
        return osp.join(*self.scope_list)

    @property
    def data_keys(self) -> list[str]:
        return [x for x in self.data_dict.keys()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_scope()

    def __repr__(self) -> str:
        format_string = "ContextManager(\n"
        format_string += f"  scope='{self.scope}',\n"
        format_string += f"  names={self.data_keys}\n"
        format_string += ")"
        return format_string


_CONTEXT_MANAGER = ContextManager()


def get_context_manager() -> ContextManager:
    return _CONTEXT_MANAGER


def clear_context_manager() -> None:
    _CONTEXT_MANAGER.clear()
