from __future__ import annotations

from typing import TYPE_CHECKING

from .callbacks import (
    CallbackEnvironment as CallbackEnvironment,
    merge_callbacks as merge_callbacks,
)
from .common import (
    deserialize_net_arch as deserialize_net_arch,
    episode_name_string as episode_name_string,
    episode_results_path as episode_results_path,
    is_closed as is_closed,
    is_vectorized as is_vectorized,
)
from .logging import (
    log_net_arch as log_net_arch,
    log_run_info as log_run_info,
    log_to_file as log_to_file,
)
from .sb3_extensions.schedules import LinearSchedule as LinearSchedule

if TYPE_CHECKING:
    from .sb3_extensions.extractors import CustomExtractor as CustomExtractor
    from .sb3_extensions.policies import NoPolicy as NoPolicy
    from .sb3_extensions.processors import (
        Fold1d as Fold1d,
        Split1d as Split1d,
    )


def __getattr__(name: str) -> object:
    if name == "CustomExtractor":
        from .sb3_extensions.extractors import CustomExtractor  # noqa: PLC0415

        return CustomExtractor
    if name == "NoPolicy":
        from .sb3_extensions.policies import NoPolicy  # noqa: PLC0415

        return NoPolicy
    if name == "Fold1d":
        from .sb3_extensions.processors import Fold1d  # noqa: PLC0415

        return Fold1d
    if name == "Split1d":
        from .sb3_extensions.processors import Split1d  # noqa: PLC0415

        return Split1d
    msg = f"module 'eta_ctrl.common' has no attribute {name!r}"
    raise AttributeError(msg)
