from collections.abc import Callable

type LayerCfg = str | dict

type TransformFn = Callable[[dict], dict]

type CollateFn = Callable[[list[dict]], dict]
