#!/bin/bash
# 4D LiDAR Context Extraction Step.
# Requires the nuScenes dataset and an OpenAI API key.
#   export OPENAI_API_KEY=...   and pass your local --nuscenes_root path.

python3 generate_description.py \
    --api_key "$OPENAI_API_KEY" \
    --nuscenes_root /path/to/nuscenes \
    --dataroot ./data \
    --start_index 0 \
    --end_index 5100
