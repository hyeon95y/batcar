import copy
import pandas as pd

from abc import ABC
from abc import abstractmethod

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from .wrappers import UpdatableModel, VoidModel, Pipeline
from .data_selectors import AllSelector
from .data_splitters import VoidSplitter


class ModelUpdater(ABC):
    '''Abstract class to define how to build a model.

    Attributes:
        data_selector: Selecting data to be used for model building.
        data_splitter: Splitting the data into training and validation sets.
        metrics: Performance metrics to evaluate the trained model.
    '''

    def __init__(
        self,
        data_selector: Optional[Callable] = None,
        data_splitter: Optional[Callable] = None,
        metrics: Optional[Union[str, List[str], Dict[str, Callable]]] = None
    ):
        assert callable(data_selector) or (data_selector is None)
        assert callable(data_splitter) or (data_splitter is None)
        assert isinstance(metrics, str) or hasattr(metrics, '__iter__') or (metrics is None)

        self._data_selector = data_selector or AllSelector()
        self._data_splitter = data_splitter or VoidSplitter()
        self._metrics = metrics or 'mse'

        self._metrics = self._as_metric_dict(self._metrics)

        self.model = None
        self.update_log = None

    def _as_metric_dict(self, metrics) -> Dict[str, Callable]:
        if isinstance(metrics, dict):
            return metrics

        else:
            if isinstance(metrics, str):
                metrics = [metrics]

            metric_dict = {}

            for metric_name in metrics:
                if metric_name == 'mse':
                    metric_func = self._mean_squared_error
                else:
                    raise ValueError(f'metric_name = {metric_name} is not supported.')

                metric_dict[metric_name] = metric_func

            return metric_dict

    def _mean_squared_error(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        return ((y_true - y_pred) ** 2).mean().mean()

    def update(self, curr_time, state):
        '''Update a model.

        Args:
            curr_time: Time to update the model.
            state: BatCar state at `curr_time`.
                TODO: Description for the attributes of `state` will be updated.
        '''

        x, y = state.x, state.y

        x_chunk, y_chunk = self._data_selector(x, y, state)

        split_data = self._data_splitter(x_chunk, y_chunk, state)
        x_train, y_train, x_valid, y_valid = split_data

        if x_chunk.empty:
            model = VoidModel()
            model = self._to_updatable(model)
        else:
            model = self._create_model(curr_time, state)
            model = self._to_updatable(model)

        model.fit(x_train, y_train, eval_set=(x_valid, y_valid))

        # update log
        update_log = {
            'train_index': x_train.index.tolist(),
            'valid_index': x_valid.index.tolist(),
        }

        y_pred = model.predict(x_chunk)
        y_train_pred = y_pred.loc[x_train.index]
        y_valid_pred = y_pred.loc[x_valid.index]

        for name, func in self._metrics.items():
            update_log[f'train_{name}'] = func(y_train, y_train_pred)
            update_log[f'valid_{name}'] = func(y_valid, y_valid_pred)

        update_log.update(model.fit_log())

        self.model = model
        self.update_log = update_log

        return model

    def _to_updatable(self, model):
        if isinstance(model, UpdatableModel):
            return model
        else:
            return UpdatableModel(model)

    def log(self) -> Union[Dict[str, Any], None]:
        '''Logs during the model update.
        '''
        return self.update_log

    @abstractmethod
    def _create_model(self, curr_time, state):
        pass

    @staticmethod
    def from_generator(
        generator,
        data_selector=None,
        data_splitter=None,
        metrics: Optional[Union[str, List[str], Dict[str, Callable]]] = None,
        preproc=None,
        y_preproc=None,
        y_postproc=None,
        *gen_args, **gen_kwargs
    ):
        '''Generate ModelUpdater with a generator.

        `StaticModelUpdater` is created with the arguments.

        Args:
            generator:
                If this is callable, it should return a fit_predictable obejct.
                If this is fit_predictable, it will be deep-copied whenever model is updated.
            data_selector: Selecting data to be used for model building.
            data_splitter: Splitting the data into training and validation sets.
            metrics: Performance metrics to evaluate the trained model.
            preproc: Any object implementing two methods: fit and transform.
                Feature matrix X processed by `preproc` will be the first input of
                the model from `generator`.
            y_preproc: Any object implementing two methods: fit and transform.
                Target matrix y processed by `y_preproc` will be the second input of
                the model from `generator`.
            y_postproc: Any object implementing two methods: fit and transform.
                Output `y_pred` of the model from `generator` will be processed further by
                `y_postproc`.
            *gen_args, **gen_kwargs: Passed to `generator` when callable.
        '''
        return StaticModelUpdater(
            generator,
            data_selector,
            data_splitter,
            metrics,
            preproc,
            y_preproc,
            y_postproc,
            *gen_args,
            **gen_kwargs
        )


class StaticModelUpdater(ModelUpdater):
    '''Update the same model regardless of time.

    Attributes:
        generator:
            If this is callable, it should return a fit_predictable obejct.
            If this is fit_predictable, it will be deep-copied whenever model is updated.
        data_selector: Selecting data to be used for model building.
        data_splitter: Splitting the data into training and validation sets.
        metrics: Performance metrics to evaluate the trained model.
        preproc: Any object implementing two methods: fit and transform.
            Feature matrix X processed by `preproc` will be the first input of
            the model from `generator`.
        y_preproc: Any object implementing two methods: fit and transform.
            Target matrix y processed by `y_preproc` will be the second input of
            the model from `generator`.
        y_postproc: Any object implementing two methods: fit and transform.
            Output `y_pred` of the model from `generator` will be processed further by
            `y_postproc`.
        *gen_args, **gen_kwargs: Passed to `generator` when callable.
    '''

    def __init__(
        self,
        generator,
        data_selector: Optional[Callable] = None,
        data_splitter: Optional[Callable] = None,
        metrics: Optional[Union[str, List[str], Dict[str, Callable]]] = None,
        preproc: Optional[Pipeline] = None,
        y_preproc: Optional[Pipeline] = None,
        y_postproc: Optional[Pipeline] = None,
        *gen_args,
        **gen_kwargs
    ):
        super().__init__(data_selector, data_splitter, metrics)

        assert callable(generator) \
            or (hasattr(generator, 'fit') and hasattr(generator, 'predict'))

        self._generator = generator
        self._preproc = preproc
        self._y_preproc = y_preproc
        self._y_postproc = y_postproc
        self._gen_args = gen_args
        self._gen_kwargs = gen_kwargs

    def _create_model(self, curr_time, state):
        if callable(self._generator):
            model = self._generator(*self._gen_args, **self._gen_kwargs)

        else:
            assert hasattr(self._generator, 'fit') and hasattr(self._generator, 'predict'), \
                'generator is not callable nor fit_predictable.'

            model = copy.deepcopy(self._generator)

        model = UpdatableModel(
            model,
            self._preproc,
            self._y_preproc,
            self._y_postproc
        )

        return model
