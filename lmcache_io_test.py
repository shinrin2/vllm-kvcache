import argparse
import os.path
from time import sleep
import subprocess
import signal
import sys
import time
from openai import OpenAI
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory configurations
OFFLOAD_DIR = '/samsung-data1/kvcache'
RESULT_DIR = 'lmcache-kv-offload-{model}-trace'

def exec_cmd_background(cmd, run=True):
    """Execute a command in background and return the process."""
    if not run:
        print(f"Would run: {cmd}")
        return None
    return subprocess.Popen(cmd, shell=True)

def kill_process_and_children(pid):
    """Kill a process and all its children."""
    try:
        parent = subprocess.Popen(f'pstree -p {pid} | grep -o "([0-9]*)" | grep -o "[0-9]*"',
                                shell=True, stdout=subprocess.PIPE)
        children = parent.communicate()[0].decode()
        for child_pid in children.split('\n'):
            if child_pid:
                try:
                    os.kill(int(child_pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    except Exception as e:
        print(f"Error killing process {pid}: {e}")

def query_and_measure_ttft(client, model, prompt):
    """Query the vLLM server and measure Time To First Token (TTFT)."""
    start = time.perf_counter()
    ttft = None

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.7,
        stream=True,
    )

    for chunk in chat_completion:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            print(chunk_message, end="", flush=True)

    print("\n")  # New line after streaming
    return ttft - start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', default=False,
                        help='Execute the commands')
    args = parser.parse_args()

    # Model configuration
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MAX_MODEL_LEN = 8192

    # Create result directory
    result_dir = RESULT_DIR.format(model=MODEL_NAME.split('/')[-1])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Define output files
    results_file_inference = os.path.join(result_dir, f'{MODEL_NAME.split("/")[-1]}-kv-offload.txt')
    results_file_gpu = os.path.join(result_dir, f'{MODEL_NAME.split("/")[-1]}-kv-offload-gpu.txt')
    results_file_bpftrace = os.path.join(result_dir, f'{MODEL_NAME.split("/")[-1]}-kv-offload-bpftrace-block.txt')

    # Create LMCache config file
    config_file = 'disk-offload.yaml'

    # Command templates
    gpu_monitor_cmd = f'nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv -lms 200 > {results_file_gpu} 2>&1'
    bpftrace_cmd = f'sudo bpftrace ../bpftrace-scripts/bio-bite-size.bt > {results_file_bpftrace} 2>&1'
    
    # Start monitoring processes
    print("Starting monitoring processes...")
    p_gpu_util = exec_cmd_background(gpu_monitor_cmd, run=args.run)
    p_bpftrace = exec_cmd_background(bpftrace_cmd, run=args.run)

    try:
        if args.run:
            # Initialize OpenAI client
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),  # required by OpenAI client even for local servers
                base_url="http://localhost:8000/v1"
            )

            # Get model ID
            models = client.models.list()
            model = models.data[0].id

            # Prepare test prompt
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            question = "Summarize bash in 2 sentences."
            prompt = f"Here's a question: {question}"

            print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

            # Run cold query
            print("\nQuerying vLLM server with cold LMCache Disk Offload")
            cold_ttft = query_and_measure_ttft(client, model, prompt)
            print(f"Cold TTFT: {cold_ttft:.3f} seconds")

            # Run warm query
            print("\nQuerying vLLM server with warm LMCache Disk Offload")
            warm_ttft = query_and_measure_ttft(client, model, prompt)
            print(f"Warm TTFT: {warm_ttft:.3f} seconds")

            print(f"\nTTFT Improvement: {(cold_ttft - warm_ttft):.3f} seconds \
                ({(cold_ttft/warm_ttft):.1f}x faster)")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        if args.run:
            print("Cleaning up monitoring processes...")
            if p_gpu_util:
                kill_process_and_children(p_gpu_util.pid)
            if p_bpftrace:
                os.system(f'sudo kill {p_bpftrace.pid}')
                p_bpftrace.kill()

    print("Test completed. Results are in:", result_dir)

if __name__ == "__main__":
    main() 