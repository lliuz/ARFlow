import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        self._log = _log

        self.cfg = config
        self.save_root = save_root
        self.summary_writer = SummaryWriter(str(save_root))

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])

        self.model = self._init_model(model)
        self.optimizer = self._create_optimizer()
        self.loss_func = loss_func

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def train(self):
        for epoch in range(self.cfg.epoch_num):
            self._run_one_epoch()

            if self.i_epoch % self.cfg.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)

    def _init_model(self, model):
        model = model.to(self.device)
        if self.cfg.pretrained_model:
            self._log.info("=> using pre-trained weights {}.".format(
                self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)
        else:
            self._log.info("=> Train from scratch.")
            model.init_weights()
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _create_optimizer(self):
        self._log.info('=> setting Adam solver')
        param_groups = [
            {'params': bias_parameters(self.model.module),
             'weight_decay': self.cfg.bias_decay},
            {'params': weight_parameters(self.model.module),
             'weight_decay': self.cfg.weight_decay}]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)
        return optimizer

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def save_model(self, error, name):
        is_best = error < self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best)
