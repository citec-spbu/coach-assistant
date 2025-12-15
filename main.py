import subprocess
import sys

import httpx
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from dancepose.scripts import run_pose_async as run_pose
import yaml

import numpy as np
import ffmpeg
import os
import logging
import json
import torch
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
VIDEO_DIR = "uploads/"

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
    
    video_name = result["video_name"] 
    poses_file = result["poses_file"]
    overlay_file = str(result["overlay_file"])
    
    data = {"confidence": 0, "figures": ["NotPerforming"], "spatial_similarity": 0, "timing": 0, "balance": 0, "classifier_clarity": 0, "error_details": {}}
    
    try:
        from dance_classifier.inference.predict import DanceClassifierPredictor
        # Предпочтительнее использовать CUDA, в противном случае используется ЦП.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing predictor on: {device}")  # Удобно для подтверждения в журнале.
        predictor = DanceClassifierPredictor(
            model_path="best_model_20pct.pth",
            metadata_path="dance_classifier/dataset/metadata.json",
            scaler_path="dance_classifier/dataset/scaler.pkl",
            label_encoder_path="dance_classifier/dataset/label_encoder.pkl",
            device=device
        )

        predictor_result = predictor.predict_from_poses(
            poses_file,
            video_path=overlay_file,                    
            overlay_video_path=overlay_file,            
            output_dir=f"outputs/{video_name}",        
            create_analyzed_video=True
        )
        
        print("=== predictor_result ===", json.dumps(predictor_result, indent=2))
        
        if predictor_result.get('success'):
            data.update({
                "confidence": predictor_result.get("confidence", 0),
                "figures": [predictor_result.get("predicted_figure", "NotPerforming")],
                "spatial_similarity": predictor_result.get("spatial_similarity", {}).get("score", 0),
                "timing": predictor_result.get("timing", {}).get("score", 0),
                "balance": predictor_result.get("balance", {}).get("score", 0),
                "classifier_clarity": predictor_result.get("classifier_clarity", {}).get("score", 0),
                "error_details": generate_error_details(predictor_result)
            })
            
            # Берем analyzed видео
            analyzed_video_path = predictor_result.get('analyzed_video_path')
            if analyzed_video_path and os.path.exists(analyzed_video_path):
                print(f" Analyzed видео: {analyzed_video_path}")
            else:
                print(" Analyzed не найдено, fallback overlay")
                analyzed_video_path = overlay_file
        else:
            print(f" Predictor: {predictor_result.get('error')}")
            analyzed_video_path = overlay_file
            
    except Exception as e:
        print(f" Predictor ошибка: {e}")
        analyzed_video_path = overlay_file
    
    # 2. Конвертируем analyzed (или fallback overlay)
    converted_file = convert_to_h264(analyzed_video_path)
    filename = os.path.basename(converted_file)
    download_url = f"{video_name}/{filename}".replace("\\", "/")
    
    print(f" Финал: {download_url}")
    print(f" {data['figures']}")
    return download_url, data


async def safe_process(upload_url: str):
    try:
        process_url = VIDEO_DIR + upload_url.split("/")[-1]
        download_url, data = await process_video(process_url)
        print("durl, data = ", download_url, data)
        async with httpx.AsyncClient() as client:
            await client.post(KOA_WEBHOOK_RESULT, json={
                "status": "done",
                "upload_url": upload_url,
                "download_url": download_url,
                "metadata": json.dumps(data)
            })
    except Exception as e:
        print(str(e))
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

def generate_error_details(predictor_result):
    error_details = {
        "spatial_similarity": [],
        "timing": [],
        "balance": [],
        "classifier_clarity": []
    }
    
    try:
        spatial = predictor_result.get('spatial_similarity', {})
        if spatial.get('error_segments') and spatial['error_segments']:
            for seg in spatial['error_segments']:
                error_details["spatial_similarity"].append({
                    "time": f"{seg.get('start_time', 0):.2f}",
                    "issue": f"DTW {seg.get('distance', 0):.1f}",
                    "score": int(spatial.get('score', 0))
                })
        elif spatial.get('note'):
            error_details["spatial_similarity"] = [{
                "time": "—",
                "issue": spatial['note'][:50] + "...",
                "score": int(spatial.get('score', 0))
            }]
        
        balance = predictor_result.get('balance', {})
        if balance.get('error_segments') and balance['error_segments']:
            for seg in balance['error_segments']:
                error_details["balance"].append({
                    "time": f"{seg.get('start_time', 0):.2f}",
                    "issue": f"Смещение {getattr(seg, 'mean_com_offset_norm', 0)*100:.0f}%",
                    "score": int(balance.get('score', 0))
                })
                if hasattr(seg, 'mean_tilt_deg'):
                    error_details["balance"].append({
                        "time": f"{seg.get('end_time', 0):.2f}",
                        "issue": f"Наклон {seg.mean_tilt_deg:.1f}°",
                        "score": int(balance.get('score', 0))
                    })
        
        clarity = predictor_result.get('classifier_clarity', {})
        if clarity.get('error_segments') and clarity['error_segments']:
            for seg in clarity['error_segments']:
                error_details["classifier_clarity"].append({
                    "time": f"{seg.get('start_time', 0):.2f}",
                    "issue": f"Нечеткость {seg.get('mean_confidence', 0):.0%}",
                    "score": int(clarity.get('score', 0))
                })
        
        timing = predictor_result.get('timing', {})
        if timing.get('error'):
            error_details["timing"] = [{
                "time": "—",
                "issue": timing['error'][:40] + "...",
                "score": int(timing.get('score', 0))
            }]
        elif timing.get('error_segments'):
            for seg in timing['error_segments']:
                error_details["timing"].append({
                    "time": f"{seg.get('start_time', 0):.2f}",
                    "issue": f"Опоздание {seg.get('delay', 0)*1000:.0f}мс",
                    "score": int(timing.get('score', 0))
                })
    
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        error_details["spatial_similarity"] = [{"time": "—", "issue": "Ошибка анализа", "score": 0}]
    
    total = sum(len(lst) for lst in error_details.values())
    print(f" TOOLTIP'Ы: {total} ошибок для uploadpage.vue")
    for key, errors in error_details.items():
        if errors:
            print(f"  {key}: {len(errors)}x | {errors[0]['time']}: {errors[0]['issue']}")
    
    return error_details

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
