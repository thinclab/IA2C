import os
import neptune

if os.path.exists("my_secrets.py"):
    from my_secrets import project, api_token
else:
    project, api_token = None, None

run = neptune.init_run(api_token=api_token, project=project, with_id='MIAC-11', mode='read-only')

run[f"train"].download(destination='/home/lzeng/Thinclab Code/mIA2C/IA2C')