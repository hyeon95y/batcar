import pandas as pd
import numpy as np

from typing import Optional
from typing import Union
from tqdm import tqdm

from .metrics import str_to_agg_func
from .update_trigger import UpdateTrigger
from .update_trigger import PeriodicTrigger
from .model_updater import ModelUpdater

# TODO: allow to run BatCar multiple times
# TODO: allow to set a random seed


class BatCar:
    """
    Attributes:
        _update_trigger: Determine when model is updated.
        _model_updater: Determine how to update a model.
        pred_table:
            Table of predicted values.
            Shape is (n_samples) x (data_dim)
        eval_table:
        Table of performance metrics for `pred_table`.
            Shape is (n_samples) x (data_dim)
        supp_table:
            Table of supplementary information.
            Shape is (n_samples) x (data_dim)
        model_table:
            Table of model update history.
            The number of rows is equal to the number of episodes.
            Here, the episode means the lifetime of a single model.
        _x: `x` of `drive`.
        _y: `y` of `drive`.
        _metrics: `_metrics` of `drive`.
        _batch_size: `_batch_size` of `drive`.
        _batch_update: `_batch_update` of `drive`.
        __run:
            Prevent to run `drive` more than once.
            This will be removed when `drive` is updated to run multiple times.
    """
    def __init__(
        self,
        update_trigger: Union[UpdateTrigger, str],
        model_updater: ModelUpdater
    ):
        if isinstance(update_trigger, str):
            update_trigger = PeriodicTrigger(pd.Timedelta(update_trigger))

        self._update_trigger = update_trigger
        self._model_updater = model_updater

        self.pred_table = None
        self.eval_table = None
        self.supp_table = None
        self.model_table = None

        self._x = None
        self._y = None

        self._metrics = None
        self._batch_size = None
        self._batch_update = None

        # The current version allows to run `drive` only one time.
        # This will be updated
        self.__run = False

    @staticmethod
    def _mean_absolute_error(y_true: pd.DataFrame, y_pred: pd.DataFrame):
        return (y_true - y_pred).abs().mean(axis=1)

    class State:
        def __init__(self, x, y, pred_table, eval_table, supp_table, model_table):
            self.x = x
            self.y = y
            self.pred_table = pred_table
            self.eval_table = eval_table
            self.supp_table = supp_table
            self.model_table = model_table

    def drive(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        metrics: Optional[dict] = None,
        batch_size: Union[int, pd.Timedelta, str] = 100,
        batch_update: bool = False,
        curr_time: Optional[pd.Timestamp] = None,
        build_init_model: bool = False,
        pbar: bool = True,
    ):
        '''Make prediction while running through x and y.

        Args:
            x: Features.
            y: Targets.
            metrics: Performance metrics to evaluate the predicted values.
            batch_size:
                Amount of data samples in a batch, which is a unit that prediction is made.
                As long as `batch_update` is False, this affects only a way of internal execution
                and does not affect the result.
            batch_update:
                If True, model update can happen only at the beginning of a batch. 
                If False, it can happen at the time of any sample.
            curr_time: Time to drive. If None, it is set to the time of the first sample.
            build_init_model:
                If True, `model_updater` is called before driving.
                If False, the first model is built at the first trigger of `update_trigger`.
            pbar:
                If True, a progress bar will be shown.
        '''
        assert len(x) == len(y)
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(x.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
        assert all(x.index == y.index)
        assert y.shape[1] == 1
        assert isinstance(metrics, dict) or (metrics is None)        
        assert isinstance(batch_size, (int, pd.Timedelta, str))
        assert \
            (isinstance(batch_size, int) and (batch_size > 0)) or \
            (isinstance(batch_size, pd.Timedelta) and (batch_size > pd.Timedelta(0))) or \
            isinstance(batch_size, str)
        assert isinstance(batch_update, bool)
        assert isinstance(curr_time, pd.Timestamp) or (curr_time is None)
        assert isinstance(build_init_model, bool)
        assert self.__run is False

        self._x = x
        self._y = y
        self._batch_size = batch_size
        self._batch_update = batch_update
        self._metrics = metrics or {'mae': self._mean_absolute_error}

        self.pred_table = pd.DataFrame(columns=y.columns)
        self.eval_table = pd.DataFrame(columns=self._metrics.keys())
        self.supp_table = pd.DataFrame(columns=['batch_id', 'episode_id'])
        self.model_table = pd.DataFrame()

        curr_time = curr_time or x.index[0]

        if build_init_model is True:
            state = self.get_state(curr_time)
            update_log = self.update_model(curr_time, state)
            self.model_table = self.model_table.append(update_log)

        if isinstance(batch_size, int):
            seq_ids = pd.Series(np.arange(len(x)), index=x.index)
            batch_ids = seq_ids.divide(batch_size).apply(np.floor)
        else:
            batch_size = pd.Timedelta(batch_size)
            batch_edges = pd.date_range(x.index[0], x.index[-1] + batch_size, freq=batch_size)
            batch_ids = pd.cut(x.index, batch_edges, include_lowest=True, right=False)
            assert batch_ids.notnull().all()

        if pbar is True:
            for_iter = tqdm(x.index.groupby(batch_ids).items())
        else:
            for_iter = x.index.groupby(batch_ids).items()

        for batch_id, batch_index in for_iter:
            x_batch = x.loc[batch_index]
            y_batch = y.loc[batch_index]

            self.__run = True

            if batch_update:
                self._simulate_forward(x_batch, y_batch, len(x_batch))
            else:
                self._simulate_forward(x_batch, y_batch, 1)

            self.supp_table.loc[x_batch.index, 'batch_id'] = batch_id

        self.print_drive_summary()

    def print_drive_summary(self):
        print()
        print()
        print(self.model_table['build_time'].rename_axis('episode').apply(str).to_markdown())

    def _simulate_forward(self, x_batch, y_batch, step):
        #
        # batch evaluation for efficienty
        #
        assert step > 0

        if self.model_table.empty:
            episode_id = -1
            y_batch_pred = pd.DataFrame(np.nan, index=y_batch.index, columns=y_batch.columns)
        else:
            episode_id = self.model_table.index[-1]
            model = self.model_table.iloc[-1]['model']
            y_batch_pred = model.predict(x_batch)

        batch_eval = self.evaluate(y_batch, y_batch_pred)

        batch_episode = pd.DataFrame(
            np.repeat([[None, episode_id]], len(x_batch), axis=0),
            index=x_batch.index,
            columns=self.supp_table.columns
        )

        self.pred_table = self.pred_table.append(y_batch_pred)
        self.eval_table = self.eval_table.append(batch_eval)
        self.supp_table = self.supp_table.append(batch_episode)

        for curr_time in x_batch.index[::step]:

            state = self.get_state(curr_time)

            if self._update_trigger.is_triggered(curr_time, state):
                update_log = self.update_model(curr_time, state)
                self.model_table = self.model_table.append(update_log, ignore_index=True)

                #
                # correction for data samples after `curr_time`
                #
                new_model = update_log['model']

                x_part = x_batch.loc[curr_time:]
                y_part_true = y_batch.loc[curr_time:]
                y_part_pred = new_model.predict(x_part)

                part_eval = self.evaluate(y_part_true, y_part_pred) 

                self.pred_table.update(y_part_pred)
                self.eval_table.update(part_eval)
                self.supp_table.loc[curr_time:, 'episode_id'] = self.model_table.index[-1]

    def update_model(
        self,
        curr_time: pd.Timestamp,
        state: State
    ):
        #
        # model update
        #
        update_start_time = pd.Timestamp.now()
        new_model = self._model_updater.update(curr_time, state)
        update_end_time = pd.Timestamp.now()

        #
        # update log
        #
        update_log = {
            "build_time": curr_time,
            'model': new_model,
            "update_elapsed_stime": update_end_time - update_start_time,
            "update_start_stime": update_start_time,
            "update_end_stime": update_end_time,
        }
        update_log.update(self._model_updater.log())            

        return update_log

    def evaluate(
        self,
        true: pd.DataFrame,
        pred: pd.DataFrame,
        metrics: Optional[dict] = None
    ) -> pd.DataFrame:

        if metrics is None:
            metrics = self._metrics

        eval_dict = {}

        for name, func in metrics.items():
            eval_dict[name] = func(true, pred)

        return pd.DataFrame(eval_dict)

    def get_state(self, curr_time: pd.Timestamp):
        curr_eps_time = curr_time - pd.Timedelta('1ns')

        x = self._x.loc[:curr_eps_time]
        y = self._y.loc[:curr_eps_time]
        pred_table = self.pred_table.loc[:curr_eps_time]
        eval_table = self.eval_table.loc[:curr_eps_time]
        supp_table = self.supp_table.loc[:curr_eps_time]
        model_table = self.model_table

        return self.State(x, y, pred_table, eval_table, supp_table, model_table)

    def imagine_what_if(self):
        asif_pred = pd.DataFrame().reindex_like(self.pred_table)
        asif_eval = pd.DataFrame().reindex_like(self.eval_table)

        grouped = self._x.index.groupby(self.supp_table['episode_id'])

        for episode_id, episode_indices in grouped.items():
            x_episode = self._x.loc[episode_indices]
            y_episode = self._y.loc[episode_indices]

            if episode_id <= 0:
                y_episode_pred = pd.DataFrame().reindex_like(y_episode)
            else:
                episode_ix = self.model_table.index.get_loc(episode_id)
                model = self.model_table.iloc[episode_ix - 1]['model']

                y_episode_pred = model.predict(x_episode)

            episode_eval = self.evaluate(y_episode, y_episode_pred)

            asif_pred.update(y_episode_pred)
            asif_eval.update(episode_eval)

        return asif_pred, asif_eval

    def get_test_performance(
        self,
        max_period: Optional[pd.Timedelta] = None,
        agg: Optional[dict] = None,
        eval_table: Optional[pd.DataFrame] = None
    ):
        assert isinstance(max_period, pd.Timedelta) or (max_period is None)
        assert isinstance(agg, dict) or (agg is None)

        agg = agg or {'mean': np.mean}
        eval_table = eval_table if eval_table is not None else self.eval_table

        for agg_name, agg_func in agg.items():
            if isinstance(agg_func, str):
                agg[agg_name] = str_to_agg_func(agg_func)

        test_agg_dict = {}

        eval_grouped = eval_table.groupby(self.supp_table['episode_id'])

        for episode_id, episode_eval in eval_grouped:
            test_start_time = episode_eval.index[0]

            if max_period is None:
                test_end_time = episode_eval.index[-1]
            else:
                test_end_time = test_start_time + max_period

            test_eval = episode_eval.loc[test_start_time:test_end_time]

            agg_result = [func(test_eval).rename(name) for name, func in agg.items()]
            test_agg_dict[episode_id] = pd.concat(agg_result)

        return pd.DataFrame(test_agg_dict).T.rename_axis('episode_id')
