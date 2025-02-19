#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # see issue #152
export CUDA_VISIBLE_DEVICES=2
# CSV_PATH="/home/jinho/beir/beir_eval_results.csv"

# Create CSV header if the file doesn't exist
# test -f "$CSV_PATH" || echo "search_top_k,rerank_top_k,ndcg_at_10,recall_at_100" > "$CSV_PATH"

# for k in $(seq 3 1 10); do

timestamp=$(date +"%Y%m%d_%H%M")

dataset_name="tesla_manual" # Add more dataset names as needed

OUTPUT_DIR="/home/jinho/FlagEmbedding_Aizip/beir/aizip/results_hybrid/${dataset_name}/${timestamp}"

metadata_file="${OUTPUT_DIR}/metadata.txt"

# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

rank_depths=(3 4 5 6 7 8)

alpha=0.5

embedder_name_or_path="BAAI/bge-large-en-v1.5"   # Define embedder name/path
reranker_name_or_path="BAAI/bge-reranker-large"  # Define reranker name/path

echo "Sweeping with alpha: $alpha" > "$metadata_file"
echo "Rank Depths: ${rank_depths[*]}" >> "$metadata_file"
echo "Timestamp: $timestamp" >> "$metadata_file"
echo "Results will be saved in: $OUTPUT_DIR" >> "$metadata_file"
echo "--dataset_names $dataset_name" >> "$metadata_file"
echo "" >> "$metadata_file"  # Add a blank line for readability

# Add all the parameters used in the Python command to metadata
echo "Command parameters:" >> "$metadata_file"
echo "--eval_name beir" >> "$metadata_file"
echo "--dataset_dir /home/jinho/FlagEmbedding_Aizip/FlagEmbedding_Aizip/beir/aizip/data" >> "$metadata_file"
echo "--splits test dev" >> "$metadata_file"
echo "--corpus_embd_save_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/corpus_embd" >> "$metadata_file"
echo "--output_dir /home/jinho/beir/aizip/search_results" >> "$metadata_file"
echo "--search_top_k \$rank_depth" >> "$metadata_file"
echo "--rerank_top_k \$rank_depth" >> "$metadata_file"
echo "--cache_path /home/jinho/.cache/huggingface/hub" >> "$metadata_file"
echo "--overwrite True" >> "$metadata_file"
echo "--k_values 3 4 5" >> "$metadata_file"
echo "--eval_output_method markdown" >> "$metadata_file"
echo "--eval_output_path \$OUTPUT_PATH" >> "$metadata_file"
echo "--eval_metrics recall_at_3 recall_at_4 recall_at_5" >> "$metadata_file"
echo "--ignore_identical_ids True" >> "$metadata_file"
echo "--embedder_name_or_path $embedder_name_or_path" >> "$metadata_file"
echo "--reranker_name_or_path $reranker_name_or_path" >> "$metadata_file"
echo "--embedder_batch_size 1024" >> "$metadata_file"
echo "--reranker_batch_size 1024" >> "$metadata_file"
echo "--devices cuda:0" >> "$metadata_file"
echo "--alpha \$alpha" >> "$metadata_file"
echo "" >> "$metadata_file"  # Add a blank line after parameters

echo "Metadata written to: $metadata_file"


for rank_depth in "${rank_depths[@]}"; do
    OUTPUT_PATH="${OUTPUT_DIR}/results/alpha_${alpha}_rank_${rank_depth}.md"

    echo "Sweeping alpha: ${alpha}, rank: ${rank_depth}"
    
    # Run the evaluation command with all parameters
    python -m FlagEmbedding_Aizip.evaluation.beir \
        --eval_name beir \
        --dataset_dir /home/jinho/FlagEmbedding_Aizip/FlagEmbedding_Aizip/beir/aizip/data \
        --dataset_names $dataset_name \
        --splits test dev \
        --corpus_embd_save_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/corpus_embd \
        --output_dir /home/jinho/beir/aizip/search_results \
        --search_top_k $rank_depth \
        --rerank_top_k $rank_depth \
        --cache_path /home/jinho/.cache/huggingface/hub \
        --overwrite True \
        --k_values 3 4 5 \
        --eval_output_method markdown \
        --eval_output_path "$OUTPUT_PATH" \
        --eval_metrics recall_at_3 recall_at_4 recall_at_5 \
        --ignore_identical_ids True \
        --embedder_name_or_path $embedder_name_or_path \
        --reranker_name_or_path $reranker_name_or_path \
        --embedder_batch_size 1024 \
        --reranker_batch_size 1024 \
        --devices cuda:0 \
        --alpha $alpha

done

python process_rank_sweep.py --rank_depths ${rank_depths[*]} --output_dir "$OUTPUT_DIR" --alpha $alpha