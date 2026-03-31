import pytest

import eta_ctrl.common as common_module


class TestCommonLazyImports:
    """Test that __getattr__ lazy-loading in eta_ctrl.common works correctly."""

    def test_custom_extractor_is_accessible(self) -> None:
        cls = common_module.CustomExtractor
        assert cls is not None
        assert cls.__name__ == "CustomExtractor"

    def test_fold1d_is_accessible(self) -> None:
        cls = common_module.Fold1d
        assert cls is not None
        assert cls.__name__ == "Fold1d"

    def test_split1d_is_accessible(self) -> None:
        cls = common_module.Split1d
        assert cls is not None
        assert cls.__name__ == "Split1d"

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match=r"module 'eta_ctrl\.common' has no attribute 'NonExistent'"):
            _ = common_module.NonExistent
