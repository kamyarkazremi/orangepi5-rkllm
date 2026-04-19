#!/bin/bash
/opt/rkllm-env/bin/python3 -c "
from huggingface_hub import HfApi, login
import subprocess, sys

# Try whoami first
try:
    api = HfApi()
    info = api.whoami()
    print('Already logged in as:', info['name'])
    sys.exit(0)
except Exception:
    pass

print('Not logged in. Run: huggingface-cli login')
print('Or set HF_TOKEN environment variable.')
sys.exit(1)
"
