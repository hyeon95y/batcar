import inspect
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

from typing import Callable
from typing import Optional
from typing import Tuple


class UpdatableModel:
    '''Wrapping a fit_predictable model for use in ModelUpdater.

    To be used in ModelUpdater, the following four functions should be implemented.
    * fit (with `eval_set` argument)
    * predict
    * fit_predict
    * fit_log
    Any sklearn or sklearn-like model object can be wrapped.

    Attributes:
        fit_predictable: Any object implementing `fit` and `predict`.
        preproc: Any object implementing two methods: fit and transform.
            Feature matrix X processed by `preproc` will be the first input of
            `fit_predictable.predict`.
        y_preproc: Any object implementing two methods: fit and transform.
            Target matrix y processed by `y_preproc` will be the second input of 
            `fit_predictable.predict`.
        y_postproc: Any object implementing two methods: fit and transform.
            Output `y_pred` of `fit_predictable.predict` will be processed further by
            `y_postproc`.
    '''

    def __init__(self, fit_predictable, preproc=None, y_preproc=None, y_postproc=None):
        assert hasattr(fit_predictable, 'fit')
        assert hasattr(fit_predictable, 'predict')

        self._fit_predictable = fit_predictable
        self._preproc = preproc
        self._y_preproc = y_preproc
        self._y_postproc = y_postproc

    def fit(self,
            X: pd.DataFrame,
            y: Optional[pd.DataFrame] = None,
            eval_set: Tuple[pd.DataFrame, pd.DataFrame] = None):
        '''Fit the `fit_predictable' object with X and y.

        Args:
            X: Features
            y: Targets
            eval_set: Validation set. This is passed to `fit_predictable.fit` with
            keyword `eval_set`.
        '''

        if self._preproc is not None:
            self._preproc.fit(X)
            X = self._preproc.transform(X)

        if self._y_preproc is not None:
            self._y_preproc.fit(y)
            y = self._y_preproc.transform(y)

        if eval_set is not None:
            eval_X, eval_y = eval_set

            if self._preproc is not None:
                eval_X = self._preproc.transform(eval_set[0])
            if self._y_preproc is not None:
                eval_y = self._y_preproc.transform(eval_set[1])

            eval_set = eval_X, eval_y

        fit_args = inspect.getfullargspec(self._fit_predictable.fit).args

        if 'eval_set' in fit_args:
            self._fit_predictable.fit(X, y, eval_set=eval_set)
        elif 'valid' in fit_args:
            self._fit_predictable.fit(X, y, valid=eval_set)
        else:
            self._fit_predictable.fit(X, y)

        if self._y_postproc is not None:
            y_pred = self._fit_predictable.predict(X)
            self._y_postproc.fit(y_pred)

        return self

    def predict(self, X: pd.DataFrame):
        '''Make prediction for X.

        Args:
            X: Features for which prediction will be made.
        '''
        if self._preproc is not None:
            X = self._preproc.transform(X)

        y_pred = self._fit_predictable.predict(X)

        if self._y_postproc is not None:
            y_pred = self._y_postproc.transform(y_pred)

        return y_pred

    def fit_predict(self, x, y=None, eval_set=None):
        return self.fit(x, y, eval_set).predict(x)

    def fit_log(self):
        '''Return log during `fit_predictable.fit`.
        '''
        if hasattr(self._fit_predictable, 'fit_log'):
            return self._fit_predictable.fit_log()
        else:
            return {}


class VoidModel:
    def __init__(self):
        self._y_names = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self._y_names = y.columns
        return self

    def predict(self, X):
        return pd.DataFrame(np.nan, index=X.index, columns=self._y_names)


class FrameIO:
    '''Wrapping a fit_predictable model to make it have DataFrame interface.

    Attributes:
        fit_predictable: Any object implementing `fit` and `predict`.
    '''
    def __init__(self, numpy_io):
        assert hasattr(numpy_io, 'fit')
        assert hasattr(numpy_io, 'predict') or hasattr(numpy_io, 'transform')

        self._numpy_io = numpy_io

        self._x_names = None
        self._y_names = None
        self._x_trans_names = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        '''Fit the `fit_predictable' object with X and y.

        Args:
            X: Features
            y: Targets
        '''
        X_arr = X.to_numpy()
        y_arr = y.to_numpy() if y is not None else None

        self._numpy_io.fit(X_arr, y_arr)

        self._x_names = X.columns

        if y is not None:
            self._y_names = y.columns

        if hasattr(self._numpy_io, 'transform'):
            x_trans = self._numpy_io.transform(X_arr[:2])
            if x_trans.shape[1] == X.shape[1]:
                self._x_trans_names = X.columns
            else:
                self._x_trans_names = [f'Feat {i}' for i in range(x_trans.shape[1])]

        return self

    def predict(self, X: pd.DataFrame):
        '''Make prediction for X.

        Args:
            X: Features for which prediction will be made.
        '''
        y_pred = self._numpy_io.predict(X.to_numpy())

        return pd.DataFrame(y_pred, index=X.index, columns=self._y_names)

    def transform(self, X: pd.DataFrame):
        x_trans = self._numpy_io.transform(X.to_numpy())
        return pd.DataFrame(x_trans, index=X.index, columns=self._x_trans_names)

    def inverse_transform(self, X_trans: pd.DataFrame):
        X_revert = self._numpy_io.inverse_transform(X_trans.to_numpy())
        return pd.DataFrame(X_revert, index=X_trans.index, columns=self._x_names)


class Transformer(ABC):
    """Fit and transform.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, y=None):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    @staticmethod
    def from_func(trans_func: Callable, *args, **kwargs):
        """Create a Transformer object that has the dummy fit function.

        Args:
            trans_func: used as a `transform` function of the returned transformer.
        """
        import functools

        trans_func = functools.partial(trans_func, *args, **kwargs)

        return SimpleTransformer(trans_func)


class SimpleTransformer(Transformer):
    def __init__(self, trans_func, trans_name=None):
        self.trans_func = trans_func
        self.trans_name = trans_name

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.trans_func(x)

    def __str__(self):
        return '{} ({})'.format(self.trans_name, self.trans_func)


class Pipeline(Transformer):
    def __init__(self, list_of_transformerable):
        self.transformers = []

        for t in list_of_transformerable:
            if hasattr(t, 'fit') and hasattr(t, 'transform'):
                self.transformers.append(t)
            elif callable(t):
                self.transformers.append(Transformer.from_func(t))
            else:
                raise TypeError(f'{t} is not supported. Transformable should either 1) have the three methods of fit, transform and fit_transform, or be callable.')

    def fit(self, x, y=None):
        x_proc = x
        for t in self.transformers:
            t.fit(x_proc)
            x_proc = t.transform(x_proc)

        return self

    def transform(self, x):
        x_proc = x
        for t in self.transformers:
            x_proc = t.transform(x_proc)
        return x_proc
