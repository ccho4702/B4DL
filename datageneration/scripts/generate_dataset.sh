#!/bin/bash
# Context-to-QA Transformation Step.
# Generates QA pairs for every task, then merges and preprocesses them
# into the training conversation format.
#   export OPENAI_API_KEY=... before running.

for task in existence binary description temporal_understanding comprehensive; do
    python3 generate_dataset.py \
        --api_key "$OPENAI_API_KEY" \
        --dataroot ./data \
        --task "$task" \
        --start_index 0 \
        --end_index 5100
done

python3 merge_json.py --dataroot ./data
python3 preprocess_dataset.py --dataroot ./data
