import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

import basic_train
from logger import init_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/sintel_ft.json')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-m', '--model', default=None)
    parser.add_argument('--n_gpu', type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    if args.evaluate:
        cfg.train.update({
            'epochs': 1,
            'epoch_size': -1,
            'valid_size': 0,
            'workers': 1,
            'val_epoch_size': 1,
        })

    if args.model is not None:
        cfg.train.pretrained_model = args.model
    cfg.train.n_gpu = args.n_gpu

    # store files day by day
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    cfg.save_root = Path('./outputs/checkpoints') / curr_time[:6] / curr_time[6:]
    cfg.save_root.makedirs_p()

    # init logger
    _log = init_logger(log_dir=cfg.save_root, filename=curr_time[6:] + '.log')
    _log.info('=> will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations \n ' + cfg_str)

    basic_train.main(cfg, _log)
