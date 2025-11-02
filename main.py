import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path

import yaml

import model_simulation

def dir_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        return Path(cfg["output_dir"])


PROCESSED_DIR = dir_from_yaml("dancepose/configs/default.yaml")
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


async def safe_process(upload_url: str):
    try:
        result = await model_simulation.main(upload_url)
        async with httpx.AsyncClient() as client:
            await client.post(KOA_WEBHOOK_RESULT, json={
                "status": "done",
                "upload_url": upload_url,
                "download_url": str(PROCESSED_DIR / result),
            })
    except Exception as e:
        print(str(e))
        async with httpx.AsyncClient() as client:
            await client.post(KOA_WEBHOOK_RESULT, json={
                "status": "failed",
                "upload_url": upload_url,
                "download_url": None,
                "error": str(e)
            })

@app.post("/api/send/", status_code=204)  # No content
async def post_path(url: UrlBody, background_task: BackgroundTasks):
    print("Получен URL для обработки:", url.upload_url)
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
