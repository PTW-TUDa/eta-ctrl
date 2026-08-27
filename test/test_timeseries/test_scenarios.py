import pytest

from eta_ctrl.timeseries.scenarios import _fix_col_name

FIX_COL_NAME_TEST_CASES = [
    # name, prefix, prefix_renamed, rename_cols, expected_result, test_id
    # Basic cases, no renaming
    ("col", None, False, None, "col", "no_prefix_no_rename"),
    ("col", "pre", False, None, "pre_col", "prefix_no_rename"),
    # With renaming, prefix_renamed=False
    ("old", None, False, {"old": "new"}, "new", "rename_no_prefix_at_all"),
    ("old", "pre", False, {"old": "new"}, "new", "rename_no_prefix_on_renamed"),
    # With renaming, prefix_renamed=True
    ("old", None, True, {"old": "new"}, "new", "rename_prefix_renamed_but_no_prefix"),
    ("old", "pre", True, {"old": "new"}, "pre_new", "rename_with_prefix_on_renamed"),
    # Edge cases
    ("", "pre", False, None, "pre_", "empty_name_with_prefix"),
    ("col", "", False, None, "_col", "empty_prefix"),
    ("col", "pre", False, {}, "pre_col", "empty_rename_dict"),
]


@pytest.mark.parametrize(
    ("name", "prefix", "prefix_renamed", "rename_cols", "expected", "test_id"),
    FIX_COL_NAME_TEST_CASES,
    ids=[case[5] for case in FIX_COL_NAME_TEST_CASES],  # Use id for readable output
)
def test_fix_col_name(name, prefix, prefix_renamed, rename_cols, expected, test_id):
    """Test _fix_col_name with various parameter combinations."""
    result = _fix_col_name(name=name, prefix=prefix, prefix_renamed=prefix_renamed, rename_cols=rename_cols)
    assert result == expected
