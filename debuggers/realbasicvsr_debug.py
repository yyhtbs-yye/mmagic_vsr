import os
import sys

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# It's assumed that your train.py script processes command line arguments.
# We'll mimic the command line arguments here.
sys.argv = [
    'tools/train.py',  # The actual script path to run.
    '/workspace/mmagic/configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py',  # The config file path.
    '--work-dir', 'work_dirs/realbasicvsr_c64b20'  # Additional arguments.
]

# Now we import the train script. This should be done AFTER setting sys.argv.
# Replace 'tools.train' with the correct module path to your train.py script.
# If 'tools/train.py' is not a module, you might need to adjust the path
# or the way you import and execute it.

# Example assuming 'train.py' can be imported like a module:
from tools import train

if __name__ == "__main__":
    train.main()  # Or however the main function is called within your train.py script.
