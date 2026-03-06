from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo import opt
from pyomo.common.errors import InfeasibleConstraintException

from eta_ctrl.agents.mpc_agent import MpcAgent
from eta_ctrl.envs.state import StateConfig, StateVar
from test.resources.agents.mpc_basic_model import MPCBasicModel

MODEL_IMPORT = "test.resources.agents.mpc_basic_model.MPCBasicModel"


class TestMpcAgent:
    @pytest.fixture(scope="class")
    def mpc_agent(self, unified_env_factory):
        state_config = StateConfig(
            StateVar(name="u", is_agent_action=True, low_value=-5, high_value=5),
            StateVar(name="x0", is_agent_observation=True),
        )
        env = unified_env_factory(state_config=state_config)
        return MpcAgent(
            env=env,
            sampling_time=1,
            prediction_horizon=10,
            model_import=MODEL_IMPORT,
            solver_name="cplex_direct",
        )

    def test_model_is_loaded(self, mpc_agent):
        assert isinstance(mpc_agent.model, MPCBasicModel)
        assert mpc_agent.concrete_model is mpc_agent.model.model

    def test_actions_order(self, mpc_agent):
        assert mpc_agent.actions_order == ["u"]

    def test_learn_returns_self(self, mpc_agent):
        result = mpc_agent.learn(total_timesteps=5)
        assert result is mpc_agent

    def test_solve_produces_optimal_solution(self, mpc_agent):
        """Test that solve() returns the correct optimal solution using a real solver.

        MPCBasicModel minimizes (x[0]-1)² + (u[0]+1)² with a single prediction step.
        """

        solved_model = mpc_agent.solve()

        assert pyo.value(solved_model.u[0]) == pytest.approx(0.618, abs=1e-2)
        assert pyo.value(solved_model.x[0]) == pytest.approx(0.0, abs=1e-2)

        assert pyo.value(solved_model.u[1]) == pytest.approx(0.235, abs=1e-2)
        assert pyo.value(solved_model.x[1]) == pytest.approx(0.618, abs=1e-2)

    @pytest.mark.parametrize(("x0", "expected"), [(0.0, 0.618), (2.0, -0.618)])
    def test_predict(self, mpc_agent, x0, expected):
        actions, _ = mpc_agent.predict(observation={"x0": np.array([[x0]])})
        assert actions.item() == pytest.approx(expected, abs=1e-2)

    def test_solver_continues_on_suboptimal_solution(self, mpc_agent, caplog):
        """Test that solver continues with warning when reaching maxTimeLimit with suboptimal solution."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.maxTimeLimit,
            status=opt.SolverStatus.ok,
            has_solution=True,
            gap=0.05,
        )

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            result = mpc_agent.solve()

            assert any("did not reach optimal solution" in record.message for record in caplog.records)
            assert any("maxTimeLimit" in record.message for record in caplog.records)
            assert result is not None

    def test_solver_exits_on_infeasible_problem(self, mpc_agent):
        """Test that solver raises InfeasibleConstraintException when problem is truly infeasible."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            with patch.object(mpc_agent, "get_env"):
                with pytest.raises(InfeasibleConstraintException) as exc_info:
                    mpc_agent.solve()

            assert "Solver failed to find feasible solution" in str(exc_info.value)
            assert "infeasible" in str(exc_info.value)

    def test_solver_exits_on_solver_error(self, mpc_agent):
        """Test that solver raises InfeasibleConstraintException when encountering a solver error."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.error, status=opt.SolverStatus.error, has_solution=False
        )

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            with patch.object(mpc_agent, "get_env"):
                with pytest.raises(InfeasibleConstraintException) as exc_info:
                    mpc_agent.solve()

            assert "Solver failed to find feasible solution" in str(exc_info.value)
            assert "error" in str(exc_info.value)

    def test_solver_continues_on_iteration_limit(self, mpc_agent, caplog):
        """Test that solver continues when hitting iteration limit with a feasible solution."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.maxIterations, status=opt.SolverStatus.ok, has_solution=True
        )

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            result = mpc_agent.solve()

            assert any("did not reach optimal solution" in record.message for record in caplog.records)
            assert result is not None

    def test_solver_logs_small_result_directly(self, mpc_agent, caplog):
        """Test that small result objects are logged directly without saving to disk."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        mock_result.__str__ = MagicMock(return_value="Small result: " + "x" * 100)

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            with patch.object(mpc_agent, "get_env"):
                with pytest.raises(InfeasibleConstraintException):
                    mpc_agent.solve()

                assert any("Full solver result object" in record.message for record in caplog.records)
                assert not any("saved to:" in record.message for record in caplog.records)

    def test_solver_saves_large_result_to_disk(self, mpc_agent, caplog, temp_dir):
        """Test that large result objects are saved to disk instead of logging."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        large_content = "Large result: " + "x" * 15000
        mock_result.__str__ = MagicMock(return_value=large_content)

        mock_env = MagicMock()
        mock_config_run = MagicMock()
        mock_config_run.results_path = str(temp_dir)
        mock_env.get_attr.return_value = [mock_config_run]

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            with patch.object(mpc_agent, "get_env", return_value=mock_env):
                with pytest.raises(InfeasibleConstraintException):
                    mpc_agent.solve()

                assert any("Full solver result saved to:" in record.message for record in caplog.records)

                saved_files = list(temp_dir.glob("solver_result_failure_*.txt"))
                assert len(saved_files) == 1
                assert saved_files[0].read_text(encoding="utf-8") == large_content

    def test_solver_handles_disk_write_failure(self, mpc_agent, caplog, temp_dir):
        """Test that solver logs truncated result when disk write fails."""
        mock_result = _create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        large_content = "Large result: " + "x" * 15000
        mock_result.__str__ = MagicMock(return_value=large_content)

        mock_env = MagicMock()
        mock_config_run = MagicMock()
        mock_config_run.results_path = str(temp_dir)
        mock_env.get_attr.return_value = [mock_config_run]

        with patch.object(mpc_agent, "solver") as mock_solver:
            mock_solver.solve.return_value = mock_result
            with patch.object(mpc_agent, "get_env", return_value=mock_env):
                with patch("pathlib.Path.write_text", side_effect=PermissionError("Disk write failed")):
                    with pytest.raises(InfeasibleConstraintException):
                        mpc_agent.solve()

                    assert any(
                        "Could not save result to disk" in record.message and record.levelname == "WARNING"
                        for record in caplog.records
                    )
                    assert any("truncated" in record.message for record in caplog.records)


def _create_mock_solver_result(
    termination_condition: opt.TerminationCondition,
    status: opt.SolverStatus,
    has_solution: bool,
    gap: float | None = None,
) -> MagicMock:
    """Helper to create mock solver results."""
    mock_result = MagicMock()
    mock_result.solver.termination_condition = termination_condition
    mock_result.solver.status = status

    result_dict: dict[str, list] = {
        "Problem": [{}],
        "Solver": [{}],
    }

    if has_solution:
        solution_dict: dict = {}
        if gap is not None:
            solution_dict["Gap"] = MagicMock(value=gap)
        result_dict["Solution"] = [solution_dict]
    else:
        result_dict["Solution"] = []

    mock_result.__getitem__.side_effect = lambda key: result_dict[key]

    return mock_result
