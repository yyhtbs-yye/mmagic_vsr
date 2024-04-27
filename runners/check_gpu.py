import subprocess

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

if __name__=="__main__":
    idle_gpu = get_idle_gpu()
