from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo import opt
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.opt.results import SolverResults
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eta_ctrl.agents.mpc_agent import MpcAgent
from eta_ctrl.envs.state import StateConfig, StateVar
from test.resources.pyomo_basic_model import PyomoBasicModel

MODEL_IMPORT = "test.resources.pyomo_basic_model.PyomoBasicModel"


@pytest.fixture(scope="module")
def mpc_agent_factory():
    def factory(env=None, sampling_time=1, prediction_horizon=10):
        return MpcAgent(
            env=env,
            sampling_time=sampling_time,
            prediction_horizon=prediction_horizon,
            model_import=MODEL_IMPORT,
            solver_name="cplex_direct",
        )

    return factory


class TestMpcAgent:
    # Tests that use mpc_agent_factory directly
    def test_no_env(self, mpc_agent_factory):
        with pytest.raises(AttributeError):
            mpc_agent_factory()

    def test_no_vec_normalize(self, mpc_agent_factory, unified_env_factory):
        vec_normalize = VecNormalize(DummyVecEnv([unified_env_factory]))
        msg = "The MPC agent does not allow the use of normalized environments."
        with pytest.raises(TypeError, match=msg):
            mpc_agent_factory(env=vec_normalize)

    @pytest.fixture(scope="class")
    def mpc_agent(self, mpc_agent_factory, unified_env_factory):
        env = unified_env_factory()
        return mpc_agent_factory(env=env)

    def test_model_is_loaded(self, mpc_agent):
        assert isinstance(mpc_agent.model, PyomoBasicModel)
        assert mpc_agent.concrete_model is mpc_agent.model.model

    def test_learn_returns_self(self, mpc_agent):
        result = mpc_agent.learn(total_timesteps=5)
        assert result is mpc_agent


class TestMpcAgentSolverIntegration:
    """Exercise solve and predict with a real solver."""

    @pytest.fixture(scope="class")
    def mpc_agent(self, mpc_agent_factory, unified_env_factory):
        state_config = StateConfig(
            StateVar(name="x", is_agent_action=True, low_value=0, high_value=1),
            StateVar(name="temp0", is_agent_observation=True),
        )
        env = unified_env_factory(state_config=state_config)
        return mpc_agent_factory(env, prediction_horizon=4, sampling_time=1)

    def test_actions_order(self, mpc_agent):
        assert mpc_agent.actions_order == ["x"]

    def test_solve_produces_optimal_solution(self, mpc_agent):
        """Test that solve() returns the correct optimal solution using a real solver.

        Prices overridden to [10,1,1,1,1]: step 0 expensive, rest cheap.
        temp0=55, temp_min=50. Dynamics: +2°C when heating, -2°C when cooling.
        Optimal: avoid heating at t=0, cool to 53 → 51, then heat back to 53.
        """
        mpc_agent.model.pyo_update_params({"p": [10, 1, 1, 1, 1]})
        solved_model = mpc_agent.solve()

        assert pyo.value(solved_model.temp[0]) == pytest.approx(55.0, abs=1e-2)

        assert pyo.value(solved_model.x[0]) == pytest.approx(0.0, abs=1e-2)
        assert pyo.value(solved_model.temp[1]) == pytest.approx(53.0, abs=1e-2)

        assert pyo.value(solved_model.x[1]) == pytest.approx(0.0, abs=1e-2)
        assert pyo.value(solved_model.temp[2]) == pytest.approx(51.0, abs=1e-2)

        assert pyo.value(solved_model.x[2]) == pytest.approx(1.0, abs=1e-2)
        assert pyo.value(solved_model.temp[3]) == pytest.approx(53.0, abs=1e-2)

    @pytest.mark.parametrize(("temp0", "expected"), [(50.0, 1.0), (60.0, 0.0)])
    def test_predict(self, mpc_agent, temp0, expected):
        actions, _ = mpc_agent.predict(observation={"temp0": np.array([[temp0]])})
        assert actions.item() == pytest.approx(expected, abs=1e-2)

    def test_predict_value_error(self, mpc_agent):
        concrete_model = mpc_agent.model.model
        concrete_model.test_var = pyo.Var()
        concrete_model.test_var_indexed = pyo.Var(concrete_model.T)
        msg = "Couldn't fetch the value for action 'test_var_indexed' in the PyomoModel PyomoBasicModel"
        with pytest.raises(ValueError, match=msg):
            mpc_agent.predict(observation={"temp0": np.array([[55]])})


class TestMpcAgentHandleSolveFailedUnit:
    @pytest.fixture
    def mpc_agent_stub(self):
        agent = MagicMock(spec=MpcAgent)
        agent.num_timesteps = 0
        return agent

    @pytest.mark.parametrize(
        ("termination_condition", "solution_loaded", "number_of_solutions"),
        [
            (opt.TerminationCondition.maxTimeLimit, True, 0),
            (opt.TerminationCondition.feasible, True, 0),
            (opt.TerminationCondition.maxTimeLimit, False, 1),
            (opt.TerminationCondition.maxEvaluations, True, 1),
        ],
    )
    def test_continues_with_suboptimal_solution(
        self, mpc_agent_stub, caplog, termination_condition, solution_loaded, number_of_solutions
    ):
        result = _create_solver_result(
            termination_condition, solution_loaded=solution_loaded, number_of_solutions=number_of_solutions
        )

        MpcAgent.handle_solve_failed(mpc_agent_stub, result)

        assert "Continuing with best available solution" in caplog.text
        assert str(termination_condition) in caplog.text

    @pytest.mark.parametrize(
        ("termination_condition", "status"),
        [
            (opt.TerminationCondition.infeasible, opt.SolverStatus.ok),
            (opt.TerminationCondition.error, opt.SolverStatus.error),
        ],
    )
    def test_raises_without_feasible_solution(self, mpc_agent_stub, caplog, termination_condition, status):
        result = _create_solver_result(termination_condition, status=status)

        with pytest.raises(InfeasibleConstraintException, match="Solver failed to find feasible solution"):
            MpcAgent.handle_solve_failed(mpc_agent_stub, result)

        assert str(termination_condition) in caplog.text

    def test_logs_small_result_directly(self, mpc_agent_stub, caplog):
        caplog.set_level("DEBUG", logger="eta_ctrl.agents.mpc_agent")
        result = _create_solver_result(opt.TerminationCondition.infeasible)

        with pytest.raises(InfeasibleConstraintException):
            MpcAgent.handle_solve_failed(mpc_agent_stub, result)

        assert "Full solver result" in caplog.text
        assert "saved to:" not in caplog.text

    def test_saves_large_result_to_disk(self, mpc_agent_stub, caplog, tmp_path):
        caplog.set_level("DEBUG", logger="eta_ctrl.agents.mpc_agent")
        result = _create_solver_result(opt.TerminationCondition.infeasible)
        large_content = "Large result: " + "x" * 15000
        mpc_agent_stub.get_env.return_value.get_attr.return_value[0].results_path = str(tmp_path)

        with patch.object(SolverResults, "__str__", return_value=large_content):
            with pytest.raises(InfeasibleConstraintException):
                MpcAgent.handle_solve_failed(mpc_agent_stub, result)

        assert "Full solver result saved to:" in caplog.text
        saved_files = list(tmp_path.glob("solver_result_failure_*.txt"))
        assert len(saved_files) == 1
        assert saved_files[0].read_text(encoding="utf-8") == large_content

    def test_logs_truncated_result_when_disk_write_fails(self, mpc_agent_stub, caplog, tmp_path):
        result = _create_solver_result(opt.TerminationCondition.infeasible)
        large_content = "Large result: " + "x" * 15000
        mpc_agent_stub.get_env.return_value.get_attr.return_value[0].results_path = str(tmp_path)

        with (
            patch.object(SolverResults, "__str__", return_value=large_content),
            patch("pathlib.Path.write_text", side_effect=PermissionError("Disk write failed")),
            pytest.raises(InfeasibleConstraintException),
        ):
            MpcAgent.handle_solve_failed(mpc_agent_stub, result)

        assert "Could not save result to disk" in caplog.text
        assert "truncated" in caplog.text


def _create_solver_result(
    termination_condition: opt.TerminationCondition,
    *,
    status: opt.SolverStatus = opt.SolverStatus.ok,
    solution_loaded: bool = False,
    number_of_solutions: int = 0,
    gap: float | None = None,
) -> SolverResults:
    """Helper to create solver results."""
    result = SolverResults()
    result.solver.termination_condition = termination_condition
    result.solver.status = status
    result.problem.number_of_solutions = number_of_solutions

    if solution_loaded:
        solution = cast("Any", result.solution).add()
        if gap is not None:
            solution.gap = gap

    return result
