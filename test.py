from huggingface_hub import snapshot_download

snapshot_download("microsoft/Phi-3-mini-4k-instruct", force_download=True)
