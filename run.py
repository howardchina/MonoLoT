import argparse
import json

from depthnet import setup_runtime, Trainer
from depthnet.model import EstimateDepth
from depthnet.model_sc import EstimateDepthSC
from depthnet.model_supervised import EstimateDepthSupervised

import os


if __name__ == "__main__":
    os.system("nvidia-smi")

    ## runtime arguments
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
    parser.add_argument('--gpu', default=None, type=int, help='Specify a GPU device')
    parser.add_argument('--gpu_any', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    parser.add_argument('--cfg_params', default='{}', type=str, help='Manually add entries to the config dict')
    args = parser.parse_args()

    ## set up
    cfgs = setup_runtime(args)
    print(args.cfg_params)
    cfg_params = json.loads(args.cfg_params)
    cfgs.update(cfg_params)

    if 'model' in cfgs:
        model = globals().get(cfgs.get('model'))
    else:
        model = EstimateDepth
    trainer = Trainer(cfgs, model)
    run_train = cfgs.get('run_train', False)
    # run_test = cfgs.get('run_test', False)

    ## run
    if run_train:
        trainer.train()
    # if run_test:
    #     trainer.test()
