import re
import argparse
from typing import List
import matplotlib.pyplot as plt
import os

def parse_markdown_tables(md_text):
    sections = re.split(r'##\s+', md_text)
    tables = {}
    
    for section in sections[1:]:  # Skip the first split part before first '##'
        lines = section.split("\n")
        title = lines[0].strip()
        table_lines = [line for line in lines[1:] if '|' in line]
        
        if table_lines:
            headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]  # Remove empty first/last from split
            data = [[cell.strip() for cell in row.split('|')[1:-1]] for row in table_lines[2:]]  # Skip headers and separators
            
            tables[title] = {"headers": headers, "rows": data}
    
    return tables

def main(alpha_values: List[float], output_dir: str, rank_depth: int, dataset: str, embed_model: str, rank_model: str):
    recall_3 = []
    no_rank_3 = []
    recall_4 = []
    no_rank_4 = []
    recall_5 = []
    no_rank_5 = []

    embed_model = embed_model.split("/")[-1]

    rank_model = rank_model.split("/")[-1]

    # alpha_values = [0.1 * i for i in range(1)]

    # rank_depth = 5

    # OUTPUT_DIR = "/home/jinho/FlagEmbedding_Aizip/beir/aizip/results_hybrid/tesla_manual"

    OUTPUT_DIR = output_dir

    for alpha in alpha_values:
        with open(f"{OUTPUT_DIR}/results/alpha_{alpha}_rank_{rank_depth}.md", "r", encoding="utf-8") as file:
            md_text = file.read()

            tables = parse_markdown_tables(md_text)

            rows = tables['recall_at_3']['rows']

            for r in rows:
                if embed_model in r and rank_model in r:
                    recall_3.append(float(r[-1].replace("*", "")))
                elif embed_model in r:
                    no_rank_3.append(float(r[-1].replace("*", "")))

            rows = tables['recall_at_4']['rows']

            for r in rows:
                if embed_model in r and rank_model in r:
                    recall_4.append(float(r[-1].replace("*", "")))
                elif embed_model in r:
                    no_rank_4.append(float(r[-1].replace("*", "")))

            rows = tables['recall_at_5']['rows']

            for r in rows:
                if embed_model in r and rank_model in r:
                    recall_5.append(float(r[-1].replace("*", "")))
                elif embed_model in r:
                    no_rank_5.append(float(r[-1].replace("*", "")))

    os.makedirs(f"{OUTPUT_DIR}/sweep_figures", exist_ok=True)

    # Plot the results
    plt.figure(figsize=(10, 5))
    # plt.plot(depths, recall_10, marker='o', linestyle='-', label='Recall@10')
    plt.plot(alpha_values, recall_3, marker='o', linestyle='-', label='Ranker')
    plt.plot(alpha_values, no_rank_3, marker='o', linestyle='-', label='No Ranker')
    plt.xlabel('Alpha Values')
    plt.ylabel('Recall@3')
    plt.title(f'{dataset} Recall@3 vs Alpha Values rank_depth {rank_depth}\nEmbed Model: {embed_model}, Rank Model: {rank_model}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/sweep_figures/rank_{rank_depth}_recall@3.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    # plt.plot(depths, recall_10, marker='o', linestyle='-', label='Recall@10')
    plt.plot(alpha_values, recall_4, marker='o', linestyle='-', label='Ranker')
    plt.plot(alpha_values, no_rank_4, marker='o', linestyle='-', label='No Ranker')
    plt.xlabel('Alpha Values')
    plt.ylabel('Recall@4')
    plt.title(f'{dataset} Recall@4 vs Alpha Values rank_depth {rank_depth}\nEmbed Model: {embed_model}, Rank Model: {rank_model}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/sweep_figures/rank_{rank_depth}_recall@4.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the results
    plt.figure(figsize=(10, 5))
    # plt.plot(depths, recall_10, marker='o', linestyle='-', label='Recall@10')
    plt.plot(alpha_values, recall_5, marker='o', linestyle='-', label='Ranker')
    plt.plot(alpha_values, no_rank_5, marker='o', linestyle='-', label='No Ranker')
    plt.xlabel('Alpha Values')
    plt.ylabel('Recall@5')
    plt.title(f'{dataset} Recall@5 vs Alpha Values rank_depth {rank_depth}\nEmbed Model: {embed_model}, Rank Model: {rank_model}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/sweep_figures/rank_{rank_depth}_recall@5.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Completed Alpha Sweep Processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown files based on alpha values.")
    parser.add_argument("--alpha_values", type=float, nargs="+", required=True, help="List of alpha values (e.g., --alpha_values 0.1 0.2 0.3)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where markdown files are stored")
    parser.add_argument("--rank_depth", type=int, default=5, help="Rank depth value (default: 5)")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--embed_model", type=str, help="Embedding model")
    parser.add_argument("--rank_model", type=str, help="Reranker model")

    args = parser.parse_args()
    main(args.alpha_values, args.output_dir, args.rank_depth, args.dataset, args.embed_model, args.rank_model)






