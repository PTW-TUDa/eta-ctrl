from eta_ctrl.common.callbacks import CallbackEnvironment


class TestCallbackEnvironmentStringRepresentations:
    """Tests for CallbackEnvironment.__str__ and __repr__."""

    def test_str(self):
        assert str(CallbackEnvironment(plot_interval=10)) == "CallbackEnvironment(rendering every 10 episode(s))"

    def test_repr(self):
        assert repr(CallbackEnvironment(plot_interval=10)) == "CallbackEnvironment(plot_interval=10)"
