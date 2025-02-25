#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # see issue #152
export CUDA_VISIBLE_DEVICES=2

timestamp=$(date +"%Y%m%d_%H%M")

# timestamp="20250215_0116"

dataset_name="sb_qna"  # Add more dataset names as needed
OUTPUT_DIR="/home/jinho/FlagEmbedding_Aizip/beir/aizip/sweep_results/$dataset_name/${timestamp}"

metadata_file="${OUTPUT_DIR}/metadata.txt"

# Create the directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/results"

# Rank depths and alpha values
rank_depths=(3)
alpha_values=(0.1)

# embedder_name_or_path="BAAI/bge-m3"   # Define embedder name/path
embedder_name_or_path="intfloat/multilingual-e5-large"
reranker_name_or_path="BAAI/bge-reranker-v2-m3"  # Define reranker name/path

# Write metadata
echo "Dataset: $dataset_name" >> "$metadata_file"
echo "Embedding Model: $embedder_name_or_path" >> "$metadata_file"
echo "Reranker Model: $reranker_name_or_path" >> "$metadata_file"
echo "Sweeping with Rank Depths: ${rank_depths[*]}" >> "$metadata_file"
echo "Alpha Values: ${alpha_values[*]}" >> "$metadata_file"
echo "Timestamp: $timestamp" >> "$metadata_file"
echo "" >> "$metadata_file"  # Add a blank line for readability

# # Add all the parameters used in the Python command to metadata
# echo "Command parameters:" >> "$metadata_file"
# echo "--eval_name beir" >> "$metadata_file"
# echo "--dataset_dir /home/jinho/FlagEmbedding_Aizip/beir/data" >> "$metadata_file"
# echo "--splits test dev" >> "$metadata_file"
# echo "--corpus_embd_save_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/corpus_embd" >> "$metadata_file"
# echo "--output_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/search_results" >> "$metadata_file"
# echo "--search_top_k \$rank_depth" >> "$metadata_file"
# echo "--rerank_top_k \$rank_depth" >> "$metadata_file"
# echo "--cache_path /home/jinho/.cache/huggingface/hub" >> "$metadata_file"
# echo "--overwrite True" >> "$metadata_file"
# echo "--k_values 3 4 5" >> "$metadata_file"
# echo "--eval_output_method markdown" >> "$metadata_file"
# echo "--eval_output_path \$OUTPUT_PATH" >> "$metadata_file"
# echo "--eval_metrics recall_at_3 recall_at_4 recall_at_5" >> "$metadata_file"
# echo "--ignore_identical_ids True" >> "$metadata_file"
# echo "--embedder_batch_size 1024" >> "$metadata_file"
# echo "--reranker_batch_size 1024" >> "$metadata_file"
# echo "--devices cuda:0" >> "$metadata_file"
# echo "--alpha \$alpha" >> "$metadata_file"
# echo "" >> "$metadata_file"  # Add a blank line after parameters

echo "Metadata written to: $metadata_file"

# Loop through alpha values first, then rank_depths
for alpha in "${alpha_values[@]}"; do
    for rank_depth in "${rank_depths[@]}"; do
        OUTPUT_PATH="${OUTPUT_DIR}/results/alpha_${alpha}_rank_${rank_depth}.json"

        echo "Sweeping alpha: ${alpha}, rank: ${rank_depth}"

        # Run the evaluation command with all parameters
        python -m FlagEmbedding_Aizip.evaluation.beir \
            --eval_name beir \
            --dataset_dir /home/jinho/FlagEmbedding_Aizip/dataset \
            --dataset_names $dataset_name \
            --splits test \
            --corpus_embd_save_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/corpus_embd \
            --output_dir /home/jinho/FlagEmbedding_Aizip/beir/aizip/search_result/$dataset_name/${timestamp}/alpha_${alpha}_rank_${rank_depth} \
            --search_top_k $rank_depth \
            --rerank_top_k $rank_depth \
            --cache_path /home/jinho/.cache/huggingface/hub \
            --overwrite True \
            --pooling_method mean \
            --k_values 3 4 5 \
            --eval_output_method json \
            --eval_output_path "$OUTPUT_PATH" \
            --eval_metrics recall_at_3 recall_at_4 recall_at_5 \
            --ignore_identical_ids True \
            --embedder_name_or_path $embedder_name_or_path \
            --reranker_name_or_path $reranker_name_or_path \
            --embedder_batch_size 1024 \
            --reranker_batch_size 1024 \
            --devices cuda:0 \
            --alpha $alpha \
            --sparse_dataset_dir /home/jinho/FlagEmbedding_Aizip/dataset/sudachi_split_b

    done
done

# if [ ${#rank_depths[@]} -eq 1 ]; then
#     alpha_args="${alpha_values[*]}"
#     # Run the process_alpha_sweep.py when only rank_depths has a length of 1
#     python process_alpha_sweep.py --alpha_values $alpha_args --output_dir "$OUTPUT_DIR" --rank_depth "${rank_depths[0]}" \
#     --dataset $dataset_name --embed_model $embedder_name_or_path --rank_model $reranker_name_or_path

# elif [ ${#alpha_values[@]} -eq 1 ]; then
#     rank_depth_args="${rank_depths[*]}"
#     # Run the process_rank_sweep.py when only alpha_values has a length of 1
#     python process_rank_sweep.py --rank_depths $rank_depth_args --output_dir "$OUTPUT_DIR" --alpha "${alpha_values[0]}" \
#     --dataset $dataset_name --embed_model $embedder_name_or_path --rank_model $reranker_name_or_path
# fi