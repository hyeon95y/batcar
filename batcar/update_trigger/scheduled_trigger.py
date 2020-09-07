import pandas as pd

from typing import Iterable
from typing import Union

from .update_trigger import UpdateTrigger


class ScheduledTrigger(UpdateTrigger):
    """Triggering model update at specified times.

    Attributes:
        _schedule: Times to trigger model update
        _n_passed: Number of scheduled times for model update that are already passed.
    """

    def __init__(self, schedule: Iterable[Union[pd.Timestamp, str]]):
        assert hasattr(schedule, '__iter__')

        schedule = sorted(map(pd.Timestamp, schedule))

        self._schedule = schedule
        self._n_passed = 0

    def is_triggered(self, curr_time: pd.Timestamp, state) -> bool:
        """Check if any scheduled time is up.

        Args:
            curr_time (int or pd.Timestamp): Time to check if model should be updated.
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.

        Returns:
            bool: True if model update is needed, and False otherwise.
        """

        if self._n_passed >= len(self._schedule):
            return False
        else:
            next_trigger_time = self._schedule[self._n_passed]

            if next_trigger_time > curr_time:
                return False
            else:
                self._n_passed = sum([t <= curr_time for t in self._schedule])
                return True
