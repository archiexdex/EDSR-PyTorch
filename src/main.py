import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import random
import torch
import numpy as np
from datetime import datetime
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
if args.lms == 1:
    torch.cuda.set_enabled_lms(True)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            st_time = datetime.now()
            while not t.terminate():
                t.train()
                t.test()
            print(f"Cost time: {datetime.now()-st_time}")
            checkpoint.done()

if __name__ == '__main__':
    main()
