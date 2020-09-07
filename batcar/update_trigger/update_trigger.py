import pandas as pd

from abc import ABC
from abc import abstractmethod


class UpdateTrigger(ABC):
    """Interface for classes triggering model update.
    """

    @abstractmethod
    def is_triggered(self, curr_time: pd.Timestamp, state):
        """Check if model update condition is met.

        Args:
            curr_time (int or pd.Timestamp): Time to check if model should be updated.
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.

        Returns:
            bool: True if model update is needed, and False otherwise.
        """
        pass

