from __future__ import annotations

import copy
from collections.abc import Mapping
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


log = getLogger(__name__)


def dict_get_any(dikt: dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
    """Get any of the specified items from dictionary, if any are available.

    The function will return the first value it finds, even if there are multiple matches.

    :param dikt: Dictionary to get values from.
    :param names: Item names to look for.
    :param fail: Flag to determine, if the function should fail with a KeyError, if none of the items are found.
                 If this is False, the function will return the value specified by 'default'.
    :param default: Value to return, if none of the items are found and 'fail' is False.
    :return: Value from dictionary.
    :raise: KeyError, if none of the requested items are available and fail is True.
    """
    for name in names:
        if name in dikt:
            # Return first value found in dictionary
            return dikt[name]

    if fail is True:
        msg = (
            f"Did not find one of the required keys in the configuration: {names}. Possibly Check the correct spelling"
        )
        raise KeyError(msg)
    return default


def dict_pop_any(dikt: dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
    """Pop any of the specified items from dictionary, if any are available.

    The function will return the first value it finds, even if there are multiple matches.
    This function removes the found values from the dictionary!

    :param dikt: Dictionary to pop values from.
    :param names: Item names to look for.
    :param fail: Flag to determine, if the function should fail with a KeyError, if none of the items are found.
                 If this is False, the function will return the value specified by 'default'.
    :param default: Value to return, if none of the items are found and 'fail' is False.
    :return: Value from dictionary.
    :raise: KeyError, if none of the requested items are available and fail is True.
    """
    for name in names:
        if name in dikt:
            # Return first value found in dictionary
            return dikt.pop(name)

    if fail is True:
        msg = f"Did not find one of the required keys in the configuration: {names}"
        raise KeyError(msg)

    return default


def dict_search(dikt: dict[str, str], val: str) -> str:
    """Get key of _psr_types dictionary, given value.

    Raise ValueError in case of value not specified in data.

    :param val: value to search
    :param data: dictionary to search for value
    :return: key of the dictionary
    """
    for key, value in dikt.items():
        if val == value:
            return key
    msg = f"Value: {val} not specified in specified dictionary"
    raise ValueError(msg)


def deep_mapping_update(
    source: Any, overrides: Mapping[str, str | Mapping[str, Any]]
) -> dict[str, str | Mapping[str, Any]]:
    """Perform a deep update of a nested dictionary or similar mapping.

    :param source: Original mapping to be updated.
    :param overrides: Mapping with new values to integrate into the new mapping.
    :return: New Mapping with values from the source and overrides combined.
    """
    output = dict(copy.deepcopy(source)) if isinstance(source, Mapping) else {}

    for key, value in overrides.items():
        if isinstance(value, Mapping):
            output[key] = deep_mapping_update(dict(source).get(key, {}), value)
        else:
            output[key] = value
    return output
