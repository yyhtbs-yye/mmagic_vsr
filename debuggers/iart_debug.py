import os
import sys

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# It's assumed that your train.py script processes command line arguments.
# We'll mimic the command line arguments here.
sys.argv = [
    'tools/train.py',  # The actual script path to run.
    '/workspace/mmagic/configs/iart/iart_c64n7_8xb1-600k_reds4.py',  # The config file path.
    '--work-dir', 'work_dirs/iart_c64n7_8xb1'  # Additional arguments.
]

# Now we import the train script. This should be done AFTER setting sys.argv.
# Replace 'tools.train' with the correct module path to your train.py script.
# If 'tools/train.py' is not a module, you might need to adjust the path
# or the way you import and execute it.

# Example assuming 'train.py' can be imported like a module:
from tools import train

if __name__ == "__main__":
    train.main()  # Or however the main function is called within your train.py script.
