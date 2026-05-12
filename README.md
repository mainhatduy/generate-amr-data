pip uninstall -y ninja && pip install ninja
uv pip install flash-attn --no-build-isolation


# Mặc định (3 samples/input, toàn bộ dataset)
python vllm_pipeline.py

# Tùy chỉnh
python vllm_pipeline.py \
  --config configs/vllm_generation_config.json \
  --output data/vllm_synthetic_data.jsonl \
  --max-samples 100
