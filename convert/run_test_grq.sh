#!/bin/bash
/opt/rkllm-env/bin/python3 /mnt/c/Users/kamyar/Documents/PowerInfer/convert/test_grq.py 2>&1 | grep -E "^(Load|OK|FAIL|Found|ERROR)"
