from huggingface_hub import create_repo, snapshot_download, upload_folder
import os

# 1. Define your model and organization
original_model_id = "varb15/PerfectPhotonV2.1"
new_model_id = "daydreamlive/PerfectPhotonV2.1"

# 2. Create the new repo on the Hub
# You'll need to be logged in (`huggingface-cli login`)
try:
    create_repo(new_model_id, repo_type="model")
    print(f"Created repo: {new_model_id}")
except Exception as e:
    print(f"Repo may already exist: {e}")

# 3. Download the original model to a local cache
print(f"Downloading {original_model_id}...")
local_dir = snapshot_download(repo_id=original_model_id)
print(f"Downloaded to: {local_dir}")

# 4. Upload the files to the new repository
print(f"Uploading to {new_model_id}...")
upload_folder(
    folder_path=local_dir,
    repo_id=new_model_id,
    repo_type="model"
)

print("Done!")