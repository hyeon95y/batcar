from abc import abstractmethod

from typing import Tuple

from .data_selectors import RandomSelector
from .data_selectors import RecentSelector


class DataSplitter:
    '''Abstract class for classes splitting data.
    '''

    def __call__(self, x, y, state=None):
        '''Split data into training and validation sets.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        x_valid, y_valid = self.select_valid(x, y)

        x_train = x.drop(x_valid.index)
        y_train = y.drop(y_valid.index)

        return x_train, y_train, x_valid, y_valid

    @abstractmethod
    def select_valid(self, x, y, state=None) -> Tuple:
        '''Determine a validation set.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        pass


class RandomSplitter(DataSplitter):
    '''Random split.

    Attributes:
        n: Number of samples of the validation set.
        frac: Ratio of samples of the validation set.
    '''
    def __init__(self, n=None, frac=None):
        super().__init__()

        self.selector = RandomSelector(n, frac)

    def select_valid(self, x, y, state=None) -> Tuple:
        '''Select random samples.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        return self.selector(x, y, state)


class RecentSplitter(DataSplitter):
    '''Validation set is constructed with recent samples.

    Attributes:
        n: Number of samples of the validation set.
        frac: Ratio of samples of the validation set.
    '''

    def __init__(self, n=None, frac=None):
        super().__init__()

        self.selector = RecentSelector(n, frac)

    def select_valid(self, x, y, state=None) -> Tuple:
        '''Select recent samples.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        return self.selector(x, y, state)


class VoidSplitter(DataSplitter):
    '''Validation becomes empty.
    '''

    def __init__(self):
        super().__init__()

    def select_valid(self, x, y, state=None) -> Tuple:
        '''Select no sample.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        return x.head(0), y.head(0)
