"""
Асинхронная обёртка для run_pose.py
Позволяет обрабатывать видео асинхронно и параллельно (FIXED VERSION)
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

# Добавляем путь к dancepose в sys.path
script_dir = Path(__file__).parent
dancepose_root = script_dir.parent
if str(dancepose_root) not in sys.path:
    sys.path.insert(0, str(dancepose_root))

# !!! ИСПРАВЛЕНИЕ: Импортируем синхронную функцию run_pose, а не main !!!
from scripts.run_pose import run_pose


def _sync_worker(video_path: str, output_dir: str):
    """
    Вспомогательная функция для запуска run_pose с правильными параметрами.
    Выполняется в ThreadPoolExecutor.
    """
    # Путь к конфигу относительно корня проекта
    # Если скрипт запускается из корня (python main.py), то dancepose/configs/...
    cfg_path = "dancepose/configs/default.yaml"

    # Если файла нет (например, запуск из другой папки), пробуем найти
    if not Path(cfg_path).exists():
        # Fallback: пробуем абсолютный путь относительно этого скрипта
        cfg_path = str(dancepose_root / "configs" / "default.yaml")

    return run_pose(
        video_path=video_path,
        cfg_path=cfg_path,
        overrides={"output_dir": output_dir}
    )


async def main(video_path: str, output_dir: str = None) -> Dict:
    """
    Асинхронная обёртка.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return {
            "success": False,
            "error": f"Видео не найдено: {video_path}",
            "video_path": str(video_path),
        }

    # Определяем output_dir
    if output_dir is None:
        output_dir = Path("outputs") / video_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Запускаем синхронную функцию в отдельном потоке
        loop = asyncio.get_running_loop()

        # Получаем результат (словарь), а не просто путь
        result_dict = await loop.run_in_executor(
            None,
            _sync_worker,
            str(video_path),
            str(output_dir)
        )

        # Извлекаем путь к overlay видео из результата
        overlay_file = result_dict.get("overlay_mp4")

        return {
            "success": True,
            "video_path": str(video_path),
            "video_name": video_path.stem,
            "poses_file": str(output_dir / "poses.jsonl"),
            "overlay_file": overlay_file,
            "output_dir": str(output_dir)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "video_path": str(video_path),
            "video_name": video_path.stem,
            "output_dir": str(output_dir)
        }


# Сохраняем совместимость с остальными функциями
async def run(path: str) -> Dict:
    return await main(path)


async def process_multiple_videos(video_paths: List[str], output_base_dir: str = "outputs", device: str = "cpu") -> \
List[Dict]:
    import time
    start_time = time.time()
    tasks = []
    for video_path in video_paths:
        video_name = Path(video_path).stem
        output_dir = Path(output_base_dir) / video_name
        tasks.append(main(video_path, str(output_dir)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "error": str(result),
                "video_path": video_paths[i],
                "frames_processed": 0
            })
        else:
            if result.get("success"):
                result["processing_time"] = time.time() - start_time
            processed_results.append(result)
    return processed_results

