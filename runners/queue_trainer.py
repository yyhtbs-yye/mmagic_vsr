from multiprocessing import Process, set_start_method
import subprocess

from flask import Flask, request, jsonify
import psutil
import os, sys
import time
from tabulate import tabulate
import hashlib

wait_cfgs = {}
set_start_method('spawn', force=True)
app = Flask(__name__)


def get_idle_gpu():
    try:
        # Run nvidia-smi command to get GPU usage
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode()

        # Parse the output to find the most idle GPU (based on memory usage)
        gpu_stats = [line.split(',') for line in nvidia_smi_output.strip().split('\n')]
        # Example line: ['0', ' 10', ' 500']
        # Sorting GPUs by memory used first and then by GPU utilization
        gpu_stats.sort(key=lambda x: (int(x[2].strip()), int(x[1].strip())))
        
        # Return the index of the most idle GPU (should use smaller than 256MB)
        if int(gpu_stats[0][2]) < 256:
            return gpu_stats[0][0].strip()
        else:
            return None
    except Exception as e:
        print(f"Failed to read GPU stats: {e}")
        return None  # Default to GPU 0 if unable to fetch or parse
    
def train_in_process(cfg_path, model_parameters, cuda_id):

    # Redirect stdout and stderr to /dev/null or a file
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    import torch  # Assuming PyTorch is the backend
    from mmengine.runner import Runner
    from mmengine.config import Config

    cfg = Config.fromfile(cfg_path)

    cfg.model['generator'].update(**model_parameters)

    runner = Runner.from_cfg(cfg)
    runner.train()

    # Restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def train_a_model(cfg_path, model_parameters={}):

    cuda_id = get_idle_gpu()

    if cuda_id:
        # Create a new process targeting the train_in_process function
        train_process = Process(target=train_in_process, args=(cfg_path, model_parameters, cuda_id))
    
        # Start the new process
        train_process.start()

        time.sleep(5)

        # Return the process ID of the newly started process
        return train_process, cuda_id
    else:
        return None, None

@app.route('/train', methods=['POST'])
def receive_config():
    global cnt
    data = request.json
    cfg_path = data['cfg_path']
    model_parameters = data.get('model_parameters', {})
    
    if cfg_path:
        wait_cfgs[cnt] = (cfg_path, model_parameters)
        cnt += 1
        return jsonify({'message': 'Configuration added successfully'}), 200
    else:
        return jsonify({'error': 'Invalid configuration'}), 400

@app.route('/print_run', methods=['GET'])
def get_running_trainers():
    # Convert the dictionary of running trainers to a list of dictionaries for easier JSON serialization
    trainers_list = [
        {'pid': pid, 'config_path': details[0], 'model_parameters': details[1], 'cuda': details[2]}
        for pid, details in running_trainers.items()
    ]

    # Convert the list of dictionaries to a tabular format
    table_headers = ['PID', 'Config Path', 'Model Parameters', 'GPU Device ID']
    table_data = [[trainer['pid'], trainer['config_path'], trainer['model_parameters'], trainer['cuda']] for trainer in trainers_list]
    trainers_table = tabulate(table_data, headers=table_headers, tablefmt='grid')

    return trainers_table

@app.route('/print_wait', methods=['GET'])
def get_waiting_trainers():
    # Convert the dictionary of waiting trainers to a list of dictionaries for easier JSON serialization
    waiting_trainers_list = [
        {'index': index, 'run_cfg': details[0], 'model_parameters': details[1]}
        for index, details in wait_cfgs.items()
    ]

    # Convert the list of dictionaries to a tabular format
    table_headers = ['Task ID', 'Run Config', 'Model Parameters']
    table_data = [[trainer['index'], trainer['run_cfg'], trainer['model_parameters']] for trainer in waiting_trainers_list]
    waiting_trainers_table = tabulate(table_data, headers=table_headers, tablefmt='grid')

    return waiting_trainers_table

@app.route('/kill_run/<int:pid>', methods=['DELETE'])
def kill_trainer(pid):
    try:
        if pid in running_trainers:
            # Retrieve the Process object from the dictionary
            process = psutil.Process(running_trainers[pid][-1].pid)
            # Terminate all child processes
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()  # Terminate the main process
            process.wait()       # Wait for the main process to terminate
            del running_trainers[pid]  # Remove from the tracking dictionary
            return jsonify({'message': f'Trainer with PID {pid} has been terminated.'}), 200
        else:
            return jsonify({'error': 'Trainer not found or already terminated.'}), 404
    except psutil.NoSuchProcess:
        return jsonify({'error': 'Process does not exist.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/kill_wait/<int:tid>', methods=['DELETE'])
def delete_wait_task(tid):
    try:
        if tid in wait_cfgs:
            del wait_cfgs[tid]  # Remove from the tracking dictionary
            return jsonify({'message': f'Wait Trainer with Task Id {tid} has been terminated.'}), 200
        else:
            return jsonify({'error': 'Trainer not found or already terminated.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def train_from_queue():
    
    while True:
        if len(wait_cfgs) > 0:
            first_key = list(wait_cfgs.keys())[0]
            run_cfg, model_parameters = wait_cfgs.pop(first_key)
            train_process, cuda_id = train_a_model(run_cfg, model_parameters)
            
            if not train_process:
                print(f"No Idle GPU for task {run_cfg}")
                wait_cfgs[first_key] = (run_cfg, model_parameters)  # Reinsert the cfg
            else:
                train_pid = train_process.pid
                running_trainers[train_pid] = [run_cfg, model_parameters, cuda_id, train_pid, train_process]

        # Clean finished processes from the running_trainers dictionary
        for pid, details in list(running_trainers.items()):
            _, _, _, _, process = details
            if not process.is_alive():
                del running_trainers[pid]

        print(wait_cfgs)
        time.sleep(5)  # Sleep briefly to prevent a tight loop


if __name__ == '__main__':
    from threading import Thread
    global cnt
    global running_trainers
    cnt = 0
    running_trainers = {}
    training_thread = Thread(target=train_from_queue)
    training_thread.start()
    app.run(host='0.0.0.0', port=5000)


    # wait_cfgs = ["/workspace/mmagic/configs/d2dunet/d2dunet_c64n7_8xb1-600k_reds4.py"]

    # pid_queue = []

    # while 1:

    #     # Pick the first one
    #     run_cfg = wait_cfgs[0]

    #     if isinstance(run_cfg, str):
    #         pid = train_a_model(run_cfg)
    #     else: 
    #         pid = train_a_model(*run_cfg)

    #     if not pid:
    #         print(f"No Idle GPU for task {run_cfg}")
    #         time.sleep(5)
    #     else:
    #         pid_queue.append(pid)
    #         wait_cfgs.pop(0)
