import subprocess
import sys

import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from dancepose.scripts import run_pose_async as run_pose
import yaml

import ffmpeg
import os
import logging

# Configure basic logging to a file named 'app.log'
# The filemode='w' will overwrite the file each time the script runs.
# Use filemode='a' (default) to append to the file.
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def dir_from_yaml(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        return Path(cfg["output_dir"])


YAML_FILE = "dancepose/configs/default.yaml"
PROCESSED_DIR = dir_from_yaml(YAML_FILE)
PROCESSED_DIR.mkdir(exist_ok=True)
KOA_WEBHOOK_GET = "http://localhost:3000/api/get"
KOA_WEBHOOK_RESULT = "http://localhost:3000/api/result"
VIDEO_DIR = "D:\\coach-assistant\\uploads\\"

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


def convert_to_h264(input_path):
    """
    Быстро перекодирует видео из mpeg4video в H.264 (AVC)

    Args:
        input_path: путь к входному видео файлу

    Returns:
        путь к выходному файлу (processed_<имя_входного_файла>.mp4)
    """
    # Получаем директорию и имя входного файла
    input_dir = os.path.dirname(input_path) or '.'
    input_filename = os.path.basename(input_path)

    # Формируем путь к выходному файлу с префиксом "processed_"
    output_filename = f'processed_{input_filename}'
    output_path = os.path.join(input_dir, output_filename)

    try:
        # Перекодируем видео
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec='libx264',  # Кодек H.264
                preset='fast',  # Быстрое кодирование
                crf=23,  # Качество
                acodec='copy'  # Копируем аудио без перекодирования
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return output_path

    except ffmpeg.Error as e:
        print('Ошибка при конвертации:')
        print('stderr:', e.stderr.decode('utf8'))
        raise
    except Exception as e:
        print(str(e))
async def process_video(path: str):
    result = await run_pose.main(video_path=path)
    if not result["success"]:
        raise Exception(result["error"])
    print(result)
    overlay_file = str(result["overlay_file"])
    converted_file = convert_to_h264(overlay_file)
    filename = os.path.basename(converted_file)
    submit_dir = str(Path(result["video_name"]) / filename)
    return submit_dir.replace("\\", "/")

async def safe_process(upload_url: str):
    try:
        process_url = VIDEO_DIR + upload_url.split("/")[-1]
        result = await process_video(process_url)
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
