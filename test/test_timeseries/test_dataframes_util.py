from datetime import datetime, timedelta

import pandas as pd

from eta_ctrl.timeseries.dataframes import find_time_slice


class TestFindTimeSlice:
    """Test suite for find_time_slice function."""

    # Fixed reference time for consistent testing
    REF_TIME = datetime(2023, 1, 1, 12, 0, 0)
    INTERVAL = timedelta(minutes=10)
    TOTAL_TIME = pd.Timedelta(hours=2)

    def test_normal(self):
        new_start, new_end = find_time_slice(
            time_begin=self.REF_TIME, total_time=self.TOTAL_TIME, round_to_interval=self.INTERVAL
        )
        assert new_start == self.REF_TIME
        assert new_end == self.REF_TIME + self.TOTAL_TIME

    def test_pd_timestamp(self):
        """Pandas Timestamp behave differently than python datetime objects"""
        start = pd.Timestamp(ts_input=self.REF_TIME)
        new_start, new_end = find_time_slice(
            time_begin=start, total_time=self.TOTAL_TIME, round_to_interval=self.INTERVAL
        )

        assert new_start == self.REF_TIME
        assert new_end == self.REF_TIME + self.TOTAL_TIME

    def test_uneven_time_add(self):
        start = self.REF_TIME + timedelta(minutes=1)

        new_start, new_end = find_time_slice(
            time_begin=start, total_time=self.TOTAL_TIME, round_to_interval=self.INTERVAL
        )

        assert new_start == self.REF_TIME  # 12:00
        assert new_end == self.REF_TIME + self.TOTAL_TIME  # 14:00

    def test_uneven_time_substract(self):
        start = self.REF_TIME - timedelta(minutes=1)

        new_start, new_end = find_time_slice(
            time_begin=start, total_time=self.TOTAL_TIME, round_to_interval=self.INTERVAL
        )

        new_ref_time = self.REF_TIME - self.INTERVAL
        assert new_start == new_ref_time  # 11:50
        assert new_end == new_ref_time + self.TOTAL_TIME  # 13:50
