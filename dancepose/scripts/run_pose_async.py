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


# """
# Асинхронная обёртка для run_pose.py
# Позволяет обрабатывать видео асинхронно и параллельно
# """
# import asyncio
# import sys
# from pathlib import Path
# from typing import Dict, List, Optional
#
# # Добавляем путь к dancepose в sys.path
# script_dir = Path(__file__).parent
# dancepose_root = script_dir.parent
# sys.path.insert(0, str(dancepose_root))
#
# from run_pose import main as sync_main
#
#
# async def main(video_path: str, output_dir: str = None) -> Dict:
#     """
#     Асинхронная обёртка для функции run_pose.main.
#     Запускает синхронную функцию в отдельном потоке.
#
#     Args:
#         video_path: путь к видеофайлу
#         output_dir: директория для сохранения (опционально, по умолчанию outputs/{video_name})
#
#     Returns:
#         Dict с результатами:
#         {
#             "success": bool,
#             "video_path": str,
#             "video_name": str,
#             "poses_file": str,
#             "overlay_file": str or None,
#             "output_dir": str
#         }
#     """
#     video_path = Path(video_path)
#     if not video_path.exists():
#         return {
#             "success": False,
#             "error": f"Видео не найдено: {video_path}",
#             "video_path": str(video_path),
#         }
#
#     # Определяем output_dir
#     if output_dir is None:
#         output_dir = Path("outputs") / video_path.stem
#     else:
#         output_dir = Path(output_dir)
#
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     try:
#         # Запускаем синхронную функцию в отдельном потоке
#         loop = asyncio.get_running_loop()
#         overlay_file = await loop.run_in_executor(
#             None,
#             sync_main,
#             None,  # cfg_path
#             str(video_path),  # video_path
#             str(output_dir)   # output_dir
#         )
#
#         return {
#             "success": True,
#             "video_path": str(video_path),
#             "video_name": video_path.stem,
#             "poses_file": str(output_dir / "poses.jsonl"),
#             "overlay_file": overlay_file,
#             "output_dir": str(output_dir)
#         }
#
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "video_path": str(video_path),
#             "video_name": video_path.stem,
#             "output_dir": str(output_dir)
#         }
#
#
# async def run(path: str) -> Dict:
#     """
#     Простая обёртка - как в примере пользователя.
#
#     Args:
#         path: путь к видеофайлу
#
#     Returns:
#         Dict с результатами
#     """
#     result = await main(path)
#     return result
#
#
# async def process_multiple_videos(
#     video_paths: List[str],
#     output_base_dir: str = "outputs",
#     device: str = "cpu"
# ) -> List[Dict]:
#     """
#     Обрабатывает несколько видео параллельно.
#
#     Args:
#         video_paths: список путей к видео
#         output_base_dir: базовая директория для всех результатов
#         device: устройство для обработки (cpu/cuda)
#
#     Returns:
#         Список словарей с результатами для каждого видео
#     """
#     import time
#
#     start_time = time.time()
#
#     tasks = []
#     for video_path in video_paths:
#         video_name = Path(video_path).stem
#         output_dir = Path(output_base_dir) / video_name
#         tasks.append(main(video_path, str(output_dir)))
#
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     # Обрабатываем исключения
#     processed_results = []
#     for i, result in enumerate(results):
#         if isinstance(result, Exception):
#             processed_results.append({
#                 "success": False,
#                 "error": str(result),
#                 "video_path": video_paths[i],
#                 "video_name": Path(video_paths[i]).stem,
#                 "processing_time": 0,
#                 "frames_processed": 0
#             })
#         else:
#             # Добавляем время обработки
#             if result.get("success"):
#                 result["processing_time"] = time.time() - start_time
#                 result["frames_processed"] = result.get("frames_processed", 0)
#                 result["output_video"] = result.get("overlay_file", "")
#                 result["output_json"] = result.get("poses_file", "")
#             processed_results.append(result)
#
#     return processed_results
#
#
# # ============================================================================
# # ПРИМЕР ИСПОЛЬЗОВАНИЯ
# # ============================================================================
#
# async def demo():
#     """Пример использования асинхронной версии"""
#     print("🎬 Демонстрация асинхронной обработки видео\n")
#
#     # ПРИМЕР 1: Обработка одного видео
#     print("=" * 60)
#     print("ПРИМЕР 1: Обработка одного видео")
#     print("=" * 60)
#
#     # Найдём первое видео в папке "Видео движение"
#     video_dir = Path(__file__).parent.parent.parent / "Видео движение" / "Видео движение"
#     videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MOV"))
#
#     if videos:
#         test_video = videos[0]
#         print(f"Обрабатываем: {test_video.name}")
#
#         result = await run(str(test_video))
#
#         if result["success"]:
#             print(f"\nУспешно обработано!")
#             print(f"   Видео: {result['video_name']}")
#             print(f"   Позы: {result['poses_file']}")
#             print(f"   Наложение: {result['overlay_file']}")
#             print(f"   Папка: {result['output_dir']}")
#         else:
#             print(f"\nОшибка: {result.get('error', 'Неизвестная ошибка')}")
#     else:
#         print("Не найдено видео для обработки")
#         print(f"   Проверьте папку: {video_dir}")
#
#     # ПРИМЕР 2: Параллельная обработка нескольких видео
#     if len(videos) >= 2:
#         print("\n" + "=" * 60)
#         print("ПРИМЕР 2: Параллельная обработка 2 видео")
#         print("=" * 60)
#
#         test_videos = [str(v) for v in videos[:2]]
#         print(f"Обрабатываем {len(test_videos)} видео параллельно...")
#
#         import time
#         start = time.time()
#         results = await process_multiple_videos(test_videos, "outputs_async_demo")
#         elapsed = time.time() - start
#
#         print(f"\nОбработка заняла {elapsed:.1f} секунд")
#         print("\nРезультаты:")
#         for i, res in enumerate(results, 1):
#             status = "Успех" if res["success"] else "Ошибка"
#             print(f"  {i}. {res['video_name']}: {status}")
#             if not res["success"]:
#                 print(f"     Причина: {res.get('error', 'Неизвестно')}")
#
#
# if __name__ == "__main__":
#     # Запуск демо
#     asyncio.run(demo())
#
#
#
#
#
#
