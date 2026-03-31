import pytest

import eta_ctrl.simulators as simulators_module


class TestSimulatorsLazyImports:
    """Test that __getattr__ lazy-loading in eta_ctrl.simulators works correctly."""

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match=r"module 'eta_ctrl\.simulators' has no attribute 'NonExistent'"):
            _ = simulators_module.NonExistent
