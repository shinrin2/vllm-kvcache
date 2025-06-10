

# LMCACHE_CHUNK_SIZE=256 \
# LMCACHE_LOCAL_CPU=False \
# LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
# LMCACHE_LOCAL_DISK="file://local/disk_test/local_disk/" \
# LMCACHE_MAX_LOCAL_DISK_SIZE=5.0 \
CPU_OFFLOAD_GB=30
LMCACHE_CONFIG_FILE="disk-offload.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve \
    meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 8192 \
    # --cpu-offload-gb $CPU_OFFLOAD_GB \
    --kv-transfer-config  \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'  > vllm-serve.log 2>&1