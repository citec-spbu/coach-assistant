import subprocess
import sys

import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

import yaml


import logging

# Configure basic logging to a file named 'app.log'
# The filemode='w' will overwrite the file each time the script runs.
# Use filemode='a' (default) to append to the file.
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def dir_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        return Path(cfg["output_dir"])


YAML_FILE = "dancepose/configs/default.yaml"
PROCESSED_DIR = dir_from_yaml(YAML_FILE)
PROCESSED_DIR.mkdir(exist_ok=True)
KOA_WEBHOOK_GET = "http://localhost:3000/api/get"
KOA_WEBHOOK_RESULT = "http://localhost:3000/api/result"


class SendBody(BaseModel):
    """
    :status ["done", "in progress", "failed"]
    """
    status: str
    upload_url: str
    download_url: str | None


class UrlBody(BaseModel):
    upload_url: str


app = FastAPI()


async def process_video(path: str, yaml_file: str):
    from ruamel.yaml import YAML
    r_yaml = YAML()
    r_yaml.preserve_quotes = True
    with open(yaml_file, 'r') as file:
        data = r_yaml.load(file)
    data['video_path'] = path
    with open(yaml_file, 'w') as file:
        r_yaml.dump(data, file)
    result = subprocess.run([sys.executable, "dancepose/scripts/run_pose.py"], capture_output=True, text=True)
    output_lines = result.stdout.strip().split('\n')
    last_line = output_lines[-1] if output_lines else ""
    return last_line

async def safe_process(upload_url: str):
    try:
        result = await process_video(upload_url, YAML_FILE)  # model_simulation.main(upload_url)
        async with httpx.AsyncClient() as client:
            await client.post(KOA_WEBHOOK_RESULT, json={
                "status": "done",
                "upload_url": upload_url,
                "download_url": str(result),
            })
    except Exception as e:
        logging.error(str(e))
        async with httpx.AsyncClient() as client:
            await client.post(KOA_WEBHOOK_RESULT, json={
                "status": "failed",
                "upload_url": upload_url,
                "download_url": None,
                "error": str(e)
            })


@app.post("/api/send/", status_code=204)  # No content
async def post_path(url: UrlBody, background_task: BackgroundTasks):
    print("URL for processing:", url.upload_url)
    background_task.add_task(safe_process, url.upload_url)
    async with httpx.AsyncClient() as client:
        # Отправляем статус 'in progress' на KOA_WEBHOOK_RESULT
        await client.post(KOA_WEBHOOK_RESULT,
                          json=SendBody(
                              status="in progress",
                              upload_url=url.upload_url,
                              download_url=None
                          ).model_dump())


"""
@app.get("/api/get")
async def get_path(download_url: str):
    return {"download_url": PROCESSED_DIR / download_url}

"""
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
