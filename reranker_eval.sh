#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # see issue #152
export CUDA_VISIBLE_DEVICES=2
# CSV_PATH="/home/jinho/beir/beir_eval_results.csv"

# Create CSV header if the file doesn't exist
# test -f "$CSV_PATH" || echo "search_top_k,rerank_top_k,ndcg_at_10,recall_at_100" > "$CSV_PATH"

# for k in $(seq 3 1 10); do
OUTPUT_DIR="/home/jinho/FlagEmbedding_Aizip/beir/aizip/results_hybrid/tesla_manual"

rank_depth=5

alpha_values=(0.0 0.1)

for alpha in "${alpha_values[@]}"; do
    OUTPUT_PATH="${OUTPUT_DIR}/alpha_${alpha}_rank_${rank}.md"
    
    python -m FlagEmbedding_Aizip.evaluation.beir \
        --eval_name beir \
        --dataset_dir /home/jinho/FlagEmbedding_Aizip/FlagEmbedding_Aizip/beir/aizip/data \
        --dataset_names tesla_manual \
        --splits test dev \
        --corpus_embd_save_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/corpus_embd \
        --output_dir /home/jinho/beir/aizip/search_results \
        --search_top_k $rank \
        --rerank_top_k $rank \
        --cache_path /home/jinho/.cache/huggingface/hub \
        --overwrite True \
        --k_values 3 4 5 \
        --eval_output_method markdown \
        --eval_output_path "$OUTPUT_PATH" \
        --eval_metrics recall_at_3 recall_at_4 recall_at_5 \
        --ignore_identical_ids True \
        --embedder_name_or_path BAAI/bge-large-en-v1.5 \
        --reranker_name_or_path BAAI/bge-reranker-large \
        --embedder_batch_size 1024 \
        --reranker_batch_size 1024 \
        --devices cuda:0 \
        --alpha $alpha

alpha_args="${alpha_values[*]}"

python process_alpha_sweep.py --alpha_values $alpha_args --output_dir "$output_dir" --rank_depth $rank_depth
done