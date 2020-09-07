import pandas as pd

from .update_trigger import UpdateTrigger


class PeriodicTrigger(UpdateTrigger):
    """Triggering model update periodically.

    Attributes:
        _update_interval (int or pd.Timedelta): Model will be updated when elapsed time exceeds update interval.
        _last_triggered_time (int or pd.Timedelta): Time when the last model update trigger.
        _first_curr_time (int or pd.Timestamp): First time that `is_triggered` is called.
    """

    def __init__(self, update_interval: pd.Timedelta):
        assert isinstance(update_interval, pd.Timedelta)
        assert update_interval > pd.Timedelta(0)

        self._update_interval = update_interval
        self._first_curr_time = None
        self._last_triggered_time = None

    def _calculate_elapsed(self, curr_time):
        """Calculate elapsed time from the last triggered point"""
        if self._first_curr_time is None:
            elapsed = pd.Timedelta(0)
            self._first_curr_time = curr_time
        else:
            if self._last_triggered_time is None:
                elapsed = curr_time - self._first_curr_time
            else:
                elapsed = curr_time - self._last_triggered_time

        return elapsed

    def is_triggered(self, curr_time, state):
        """Check if elapsed time from the last trigger is longer than update interval.

        Args:
            curr_time (int or pd.Timestamp): Time to check if model should be updated.
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.

        Returns:
            bool: True if model update is needed, and False otherwise.
        """

        elapsed = self._calculate_elapsed(curr_time)
        is_triggered = elapsed >= self._update_interval

        if is_triggered:
            self._last_triggered_time = curr_time
        return is_triggered
