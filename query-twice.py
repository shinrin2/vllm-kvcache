import time
from openai import OpenAI
from transformers import AutoTokenizer
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # required by OpenAI client even for local servers
    base_url="http://localhost:8000/v1"
)

models = client.models.list()
model = models.data[0].id

# 119512 characters total
# 26054 tokens total
long_context = ""
with open("man-bash.txt", "r") as f:
    long_context = f.read()

# a truncation of the long context for the --max-model-len 16384
# if you increase the --max-model-len, you can decrease the truncation i.e.
# use more of the long context
long_context = long_context[:1000]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
question = "Summarize bash in 2 sentences."

prompt = f"{long_context}\n\n{question}"

print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

def query_and_measure_ttft():
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

print("Querying vLLM server with cold LMCache Disk Offload")
cold_ttft = query_and_measure_ttft()
print(f"Cold TTFT: {cold_ttft:.3f} seconds")

print("\nQuerying vLLM server with warm LMCache Disk Offload")
warm_ttft = query_and_measure_ttft()
print(f"Warm TTFT: {warm_ttft:.3f} seconds")

print(f"\nTTFT Improvement: {(cold_ttft - warm_ttft):.3f} seconds \
    ({(cold_ttft/warm_ttft):.1f}x faster)")