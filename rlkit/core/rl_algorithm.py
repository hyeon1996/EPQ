import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        if not self.trainer.discrete:
            snapshot = self._get_snapshot()
            logger.save_itr_params(epoch+1, snapshot)
            gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _set_snapshot(self, snapshot):
        trainer_snapshot = {}
        exploration_snapshot = {}
        evaluation_snapshot = {}
        replay_buffer_snapshot = {}
        for k, v in snapshot.items():
            if k.startswith('trainer/'):
                trainer_snapshot[k[8:]] = v
            elif k.startswith('exploration/'):
                exploration_snapshot[k[12:]] = v
            elif k.startswith('evaluation/'):
                evaluation_snapshot[k[11:]] = v
            elif k.startswith('replay_buffer/'):
                replay_buffer_snapshot[k[14:]] = v
        self.trainer.set_snapshot(trainer_snapshot)
        self.expl_data_collector.set_snapshot(exploration_snapshot)
        self.eval_data_collector.set_snapshot(evaluation_snapshot)
        self.replay_buffer.set_snapshot(replay_buffer_snapshot)

    def _log_with_wandb(self, diagnostics, prefix=''):
        if isinstance(diagnostics, tuple):
            diag_log, diag_wandb = diagnostics[0], diagnostics[1]
            logger.record_dict(diag_log, prefix=prefix)
            return diag_wandb
        elif isinstance(diagnostics, OrderedDict):
            logger.record_dict(diagnostics, prefix=prefix)
            return diagnostics
        else:
            logger.record_dict(diagnostics, prefix=prefix)
            return {}

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
        wandb_log = {}

        """
        Replay Buffer
        """
        wandb_log.update(self._log_with_wandb(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        ))

        """
        Trainer
        """
        wandb_log.update(self._log_with_wandb(
            self.trainer.get_diagnostics(),
            prefix='trainer/'
        ))

        """
        Exploration
        """
        wandb_log.update(self._log_with_wandb(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        ))

        expl_paths = self.expl_data_collector.get_epoch_paths()
        # import ipdb; ipdb.set_trace()
        if hasattr(self.expl_env, 'get_diagnostics'):
            wandb_log.update(self._log_with_wandb(
                self.expl_env.get_diagnostics(expl_paths),
                prefix='exploration/',
            ))
        if not self.batch_rl or self.eval_both:
            wandb_log.update(self._log_with_wandb(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            ))
        """
        Evaluation
        """
        wandb_log.update(self._log_with_wandb(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        ))
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            wandb_log.update(self._log_with_wandb(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            ))
        if hasattr(self.eval_env, 'ref_max_score') and hasattr(self.eval_env, 'ref_min_score'):
            min_return, max_return = self.eval_env.ref_min_score, self.eval_env.ref_max_score
        else:
            min_return, max_return = None, None
        wandb_log.update(self._log_with_wandb(
            eval_util.get_generic_path_information(eval_paths, min_return=min_return, max_return=max_return),
            prefix="evaluation/",
        ))

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        if logger.get_wandb_run() is not None:
            logger.get_wandb_run().log(wandb_log, step=epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass