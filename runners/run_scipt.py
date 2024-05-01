
import os

cuda_id = 2
cfg_path = "configs/pcdnet/pcdnet_c64n7_8xb1-600k_reds4.py"
model_parameters = {
        'pyramid_depth': "SPyNet"
    }

os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
import torch  # Assuming PyTorch is the backend
from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile(cfg_path)

cfg.model['generator'].update(**model_parameters)

runner = Runner.from_cfg(cfg)
runner.train()
