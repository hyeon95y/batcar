class RandomSelector:
    '''Random data selection

    Attributes:
        n: Number of samples to be selected.
        frac: Ratio of samples to be selected.
    '''

    def __init__(self, n=None, frac=None):
        assert \
            (isinstance(n, int) and (frac is None)) or \
            (n is None) and isinstance(frac, float)

        self.n = n
        self.frac = frac

    def __call__(self, x, y, state=None):
        '''Select data randomly.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        x_rand = x.sample(n=self.n, frac=self.frac)
        y_rand = y.loc[x_rand.index]

        return x_rand, y_rand


class RecentSelector:
    '''Recent data selection

    Attributes:
        n: Number of samples to be selected.
        frac: Ratio of samples to be selected.
    '''
    def __init__(self, n=None, frac=None):
        assert \
            (isinstance(n, int) and (frac is None)) or \
            (n is None) and isinstance(frac, float)

        self.n = n
        self.frac = frac

    def __call__(self, x, y, state=None):
        '''Select recent data.

        Args:
            x: Features
            y: Targets
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''
        if self.n is None:
            n_recent = int(self.frac * len(x))
        else:
            n_recent = self.n

        x_recent = x.tail(n_recent)
        y_recent = y.tail(n_recent)

        return x_recent, y_recent


class AllSelector(RecentSelector):
    '''Entire data selection.
    '''
    def __init__(self):
        super().__init__(frac=1.)
