LMCACHE_CONFIG_FILE="disk-offload.yaml" \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve \
    meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 16384 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

#    --cpu-offload-gb 10 \

# 2>&1 &