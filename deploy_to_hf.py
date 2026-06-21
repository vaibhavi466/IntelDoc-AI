import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Deploy IntelDoc-AI to Hugging Face Spaces")
    parser.add_argument("--repo_id", type=str, required=True, help="Your HF space ID (e.g. username/space-name)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face Write Token")
    args = parser.parse_args()
    
    if not args.token:
        print("Error: Hugging Face token not provided. Set the HF_TOKEN environment variable or pass --token.")
        return

    api = HfApi(token=args.token)
    
    print(f"Creating or matching Hugging Face Space: {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
    
    print("Uploading project directory to Hugging Face Space (this may take a few minutes as it uploads model files)...")
    api.upload_folder(
        folder_path=".",
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=[
            ".git",
            ".git/**/*",
            ".venv",
            ".venv/**/*",
            "**/__pycache__",
            "**/__pycache__/**/*",
            "mlruns",
            "mlruns/**/*",
            "experiments",
            "experiments/**/*",
            "notebooks",
            "notebooks/**/*",
            "data/raw",
            "data/raw/**/*",
            "data/processed_text",
            "data/processed_text/**/*",
            "dvc.lock",
            "dvc.yaml",
            ".dvc",
            ".dvc/**/*",
            ".dvcignore",
            "documind.db",
            "deploy_to_hf.py"
        ]
    )
    print(f"Deployment completed successfully! Check your Space at https://huggingface.co/spaces/{args.repo_id}")

if __name__ == "__main__":
    main()
