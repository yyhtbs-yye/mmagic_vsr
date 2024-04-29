
import os

cuda_id = 3
cfg_path = "/workspace/mmagic/configs/vrt/vrt_c64n7_8xb1-600k_reds4.py"
model_parameters = {}

os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
import torch  # Assuming PyTorch is the backend
from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile(cfg_path)

cfg.model['generator'].update(**model_parameters)

runner = Runner.from_cfg(cfg)
runner.train()
