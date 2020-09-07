from dateutil.relativedelta import relativedelta
import pandas as pd

from typing import Optional

from .update_trigger import UpdateTrigger


class MonthlyTrigger(UpdateTrigger):
    """Triggering model update periodically.

    Attributes:
        _day: Day (1 ~ 31) that model update is triggered.
        _hour: Hour (0 ~ 23) on which model update is triggered.
        _period: Frequency of model update.
        _start_time: First time when `is_triggered` is called.
        _last_trigger_time: Time of the last update.
        _next_trigger_time: Time of the next update.
    """

    def __init__(
        self,
        day: Optional[int] = None,
        hour: Optional[int] = None,
        period: int = 1
    ):
        assert isinstance(day, int) or (day is None)
        assert isinstance(hour, int) or (hour is None)
        assert isinstance(period, int)

        self._day = day
        self._hour = hour
        self._period = period

        self._start_time = None
        self._last_trigger_time = None
        self._next_trigger_time = None

    def is_triggered(self, curr_time: pd.Timestamp, state=None) -> bool:
        """Check if it is monthly model update time.

        Args:
            curr_time (int or pd.Timestamp): Time to check if model should be updated.
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.

        Returns:
            bool: True if model update is needed, and False otherwise.
        """
        if self._start_time is None:
            self._start_time = curr_time
            self._next_trigger_time = self._get_first_trigger_time()
            self._last_trigger_time = self._get_trigger_time_of_month(
                self._next_trigger_time - relativedelta(months=self._period)
            )

        if self._next_trigger_time > curr_time:
            return False
        else:
            month_step = relativedelta(months=self._period)

            candidates = [self._last_trigger_time]
            while candidates[-1] <= curr_time:
                past_candidate = self._get_trigger_time_of_month(candidates[-1] + month_step)
                candidates.append(past_candidate)

            self._next_trigger_time = candidates[-1]
            self._last_trigger_time = candidates[-2]

            return True

    def _get_first_trigger_time(self) -> pd.Timestamp:
        start_time = self._start_time
        candidate_time = self._get_trigger_time_of_month(start_time)

        if candidate_time >= start_time:
            return candidate_time
        else:
            next_month = candidate_time + relativedelta(months=1)
            next_candidate = self._get_trigger_time_of_month(next_month)
            return next_candidate

    def _get_trigger_time_of_month(self, time: pd.Timestamp):
        trigger_month = pd.Timestamp(time.year, time.month, 1).tz_localize(time.tz)

        day = self._get_trigger_day()
        hour = self._get_trigger_hour()

        actual_day = min(day, trigger_month.days_in_month)
        trigger_time = trigger_month.replace(day=actual_day, hour=hour)

        return trigger_time

    def _get_trigger_day(self) -> int:
        if self._day is None:
            return self._start_time.day
        elif isinstance(self._day, int):
            return self._day
        else:
            raise ValueError('day: {self._day} ({self._day}) is not supported.')

    def _get_trigger_hour(self) -> int:
        if self._hour is None:
            return self._start_time.hour
        elif isinstance(self._hour, int):
            return self._hour
        else:
            raise ValueError('hour: {self._hour} ({self._hour}) is not supported.')
