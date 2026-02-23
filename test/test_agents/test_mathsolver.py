from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pyomo import opt
from pyomo.common.errors import InfeasibleConstraintException

from eta_ctrl.agents.math_solver import MathSolver
from eta_ctrl.common import NoPolicy
from eta_ctrl.config import ConfigRun
from eta_ctrl.timeseries.scenario_manager import CsvScenarioManager
from test.resources.agents.mpc_basic_env import MPCBasicEnv


class DummyScenarioManager(CsvScenarioManager):
    """Dummy class for testing purposes"""

    def __init__(self) -> None:
        self.scenarios = pd.DataFrame()

    def get_scenario_state(self, n_steps: int) -> dict[str, np.ndarray]:
        return {}

    def get_scenario_state_with_duration(self, n_step: int, duration: int) -> dict[str, np.ndarray]:
        return {}


class TestMathSolver:
    @pytest.fixture(scope="class")
    def mpc_basic_env(self, temp_dir):
        config_run = ConfigRun(
            series="MPC_Basic_test_2023",
            name="test_mpc_basic",
            description="",
            root_path=temp_dir,
            results_path=temp_dir,
            scenarios_path=temp_dir,
        )

        # Create the environment
        env = MPCBasicEnv(
            env_id=1,
            config_run=config_run,
            prediction_horizon=10,
            episode_duration=1800,
            sampling_time=1,
            model_parameters={},
            scenario_manager=DummyScenarioManager(),
        )
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def mpc_agent(self, mpc_basic_env):
        # set up the agent
        return MathSolver(NoPolicy, mpc_basic_env)

    def test_mpc_save_load(self, mpc_basic_env, mpc_agent, temp_dir):
        # save
        path = temp_dir / "test_mpc_basic_agent.zip"
        mpc_agent.save(path)

        # Load the agent from the saved file
        loaded_agent = MathSolver.load(path=path, env=mpc_basic_env)

        assert isinstance(loaded_agent, MathSolver)
        assert isinstance(loaded_agent.policy, NoPolicy)

        # Compare attributes before and after loading
        assert loaded_agent.model == mpc_agent.model
        assert loaded_agent.observation_space == mpc_agent.observation_space
        assert loaded_agent.num_timesteps == mpc_agent.num_timesteps

    def test_mpc_learn(self, mpc_agent):
        assert mpc_agent.learn(total_timesteps=5) is not None
        assert isinstance(mpc_agent, MathSolver)

    def test_solver_continues_on_suboptimal_solution(self, mpc_agent, caplog):
        """Test that solver continues with warning when reaching maxTimeLimit with suboptimal solution."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.maxTimeLimit,
            status=opt.SolverStatus.ok,
            has_solution=True,
            gap=0.05,
        )

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            # Mock env_method to prevent it from being called on error path
            with patch.object(mpc_agent.get_env(), "env_method"):
                result = mpc_agent.solve()

                # Verify warning logged and execution continued
                assert any("did not reach optimal solution" in record.message for record in caplog.records)
                assert any("maxTimeLimit" in record.message for record in caplog.records)
                assert result is not None

    def test_solver_exits_on_infeasible_problem(self, mpc_agent):
        """Test that solver raises InfeasibleConstraintException when problem is truly infeasible."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            # Mock env_method to avoid DummyScenarioManager issues when handle_failed_solve is called
            with patch.object(mpc_agent.get_env(), "env_method"):
                with pytest.raises(InfeasibleConstraintException) as exc_info:
                    mpc_agent.solve()

                assert "Solver failed to find feasible solution" in str(exc_info.value)
                assert "infeasible" in str(exc_info.value)

    def test_solver_exits_on_solver_error(self, mpc_agent):
        """Test that solver raises InfeasibleConstraintException when encountering a solver error."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.error, status=opt.SolverStatus.error, has_solution=False
        )

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            with patch.object(mpc_agent.get_env(), "env_method"):
                with pytest.raises(InfeasibleConstraintException) as exc_info:
                    mpc_agent.solve()

                assert "Solver failed to find feasible solution" in str(exc_info.value)
                assert "error" in str(exc_info.value)

    def test_solver_continues_on_iteration_limit(self, mpc_agent, caplog):
        """Test that solver continues when hitting iteration limit with a feasible solution."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.maxIterations, status=opt.SolverStatus.ok, has_solution=True
        )

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            with patch.object(mpc_agent.get_env(), "env_method"):
                result = mpc_agent.solve()

                # Verify warning logged and execution continued
                assert any("did not reach optimal solution" in record.message for record in caplog.records)
                assert result is not None

    def test_solver_logs_small_result_directly(self, mpc_agent, caplog):
        """Test that small result objects are logged directly without saving to disk."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        # Make str(result) return a small string (< 10KB)
        mock_result.__str__ = MagicMock(return_value="Small result: " + "x" * 100)

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            with patch.object(mpc_agent.get_env(), "env_method"):
                with pytest.raises(InfeasibleConstraintException):
                    mpc_agent.solve()

                # Verify result was logged directly at debug level
                assert any("Full solver result object" in record.message for record in caplog.records)
                # Verify no disk save message
                assert not any("saved to:" in record.message for record in caplog.records)

    def test_solver_saves_large_result_to_disk(self, mpc_agent, caplog, temp_dir):
        """Test that large result objects are saved to disk instead of logging."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        # Make str(result) return a large string (> 10KB)
        large_content = "Large result: " + "x" * 15000
        mock_result.__str__ = MagicMock(return_value=large_content)

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            with patch.object(mpc_agent.get_env(), "env_method"):
                with pytest.raises(InfeasibleConstraintException):
                    mpc_agent.solve()

                # Verify file was saved
                assert any("Full solver result saved to:" in record.message for record in caplog.records)

                # Verify file exists and contains correct content
                saved_files = list(temp_dir.glob("solver_result_failure_*.txt"))
                assert len(saved_files) == 1
                assert saved_files[0].read_text(encoding="utf-8") == large_content

    def test_solver_handles_disk_write_failure(self, mpc_agent, caplog, temp_dir):
        """Test that solver logs truncated result when disk write fails."""
        mock_result = self._create_mock_solver_result(
            termination_condition=opt.TerminationCondition.infeasible, status=opt.SolverStatus.ok, has_solution=False
        )
        # Make str(result) return a large string (> 10KB)
        large_content = "Large result: " + "x" * 15000
        mock_result.__str__ = MagicMock(return_value=large_content)

        with patch("pyomo.environ.SolverFactory") as mock_solver_factory:
            self._setup_mock_solver(mock_solver_factory, mock_result)

            with patch.object(mpc_agent.get_env(), "env_method"):
                # Mock Path.write_text to raise an exception (simulate disk write failure)
                with patch("pathlib.Path.write_text", side_effect=PermissionError("Disk write failed")):
                    with pytest.raises(InfeasibleConstraintException):
                        mpc_agent.solve()

                    # Verify warning about disk write failure
                    assert any(
                        "Could not save result to disk" in record.message and record.levelname == "WARNING"
                        for record in caplog.records
                    )
                    # Verify truncated result was logged
                    assert any("truncated" in record.message for record in caplog.records)

    @staticmethod
    def _create_mock_solver_result(termination_condition, status, has_solution, gap=None):
        """Helper to create mock solver results."""
        mock_result = MagicMock()
        mock_result.solver.termination_condition = termination_condition
        mock_result.solver.status = status

        # Explicitly set __getitem__ to return the correct dictionaries
        result_dict = {}
        result_dict["Problem"] = [{}]
        result_dict["Solver"] = [{}]

        if has_solution:
            # Create an actual list with one dict element to properly handle len() checks
            solution_dict = {}
            if gap is not None:
                solution_dict["Gap"] = MagicMock(value=gap)
            result_dict["Solution"] = [solution_dict]
        else:
            # Empty list for no solution
            result_dict["Solution"] = []

        mock_result.__getitem__.side_effect = lambda key: result_dict[key]

        return mock_result

    @staticmethod
    def _setup_mock_solver(mock_solver_factory, mock_result):
        """Helper to setup mock solver."""
        mock_solver_instance = MagicMock()
        mock_solver_instance.solve.return_value = mock_result
        mock_solver_instance.options = MagicMock()
        mock_solver_factory.return_value = mock_solver_instance
