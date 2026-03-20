import os
import argparse
import datasets


def make_map_fn(source="math"):
    def process_fn(example, idx):
        # Extract core fields
        data_source = example.get("data_source", source)
        question = example.pop("prompt")
        solution = example.pop("solution")
        
        # Build output structure
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question+"\nPlease reason step by step, and put your final answer within \boxed{}.",
                }
            ],
            "ability": example.get("ability", "math").lower(),
            "reward_model": {
                "style": example["reward_model"]["style"],
                "ground_truth": solution
            },
            "extra_info": {
                "index": example["extra_info"]["index"],
            },
            "source_prompt": example.get("source_prompt", []),
            "messages": example.get("messages", []),
            "dataset": example.get("dataset", []),
            "ground_truth": example.get("ground_truth", "")
        }
        return data

    return process_fn

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert math_dapo JSONL to Parquet")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    args = parser.parse_args()

    # Load JSONL dataset
    print(f"Loading JSONL file: {args.input}")
    dataset = datasets.load_dataset("json", data_files=args.input, split="train")
    
    print("Processing data format...")
    processed_dataset = dataset.map(
        function=make_map_fn(),
        with_indices=True
    )
    
    # Save as Parquet file
    print(f"Saving Parquet file: {args.output}")
    processed_dataset.to_parquet(args.output)
    
    print("Conversion completed!")