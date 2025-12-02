"""
Скрипт построения эталонных траекторий фигур
на основе уже подготовленного датасета (metadata.json + JSON с позами).

Запуск (из папки dance_classifier/):

    python data_preparation/build_references.py \
        --dataset-dir data/BDD_dataset

В результате в <dataset-dir>/reference_trajectories
появятся файлы <class_name>.npy с усреднённой нормализованной траекторией.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List
import importlib.util

import numpy as np
import sys

# Добавляем путь к корню модуля dance_classifier
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем напрямую без использования пакета
import importlib.util

# Загрузка feature_extraction напрямую
fe_path = Path(__file__).parent / "feature_extraction.py"
spec = importlib.util.spec_from_file_location("feature_extraction", fe_path)
fe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_module)
FeatureExtractor = fe_module.FeatureExtractor

# Загрузка spatial_similarity напрямую
ss_path = Path(__file__).parent.parent / "utils" / "spatial_similarity.py"
spec = importlib.util.spec_from_file_location("spatial_similarity", ss_path)
ss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss_module)
build_reference_from_examples = ss_module.build_reference_from_examples


# Имена полей в metadata["videos"][i]
# Если у тебя в metadata.json другие ключи – просто поменяй тут.
LABEL_KEY_CANDIDATES = ["label", "class", "figure", "figure_name", "target"]
POSES_KEY_CANDIDATES = ["poses_path", "poses_json_path", "poses_file", "poses", "json_path"]


def _detect_keys(example: Dict) -> (str, str):
    """Пытается угадать имена полей с меткой и путём к позам."""
    label_key = None
    poses_key = None

    for k in LABEL_KEY_CANDIDATES:
        if k in example:
            label_key = k
            break

    for k in POSES_KEY_CANDIDATES:
        if k in example:
            poses_key = k
            break

    if label_key is None or poses_key is None:
        raise ValueError(
            f"Не удалось угадать ключи для label/poses. "
            f"Доступные поля: {list(example.keys())}. "
            f"Отредактируй LABEL_KEY_CANDIDATES и POSES_KEY_CANDIDATES в build_references.py."
        )

    return label_key, poses_key


def _interpolate_missing(sequence: np.ndarray) -> np.ndarray:
    """Копия логики DanceClassifierPredictor._interpolate_missing."""
    sequence = sequence.copy()

    for feature_idx in range(sequence.shape[1]):
        feature_values = sequence[:, feature_idx]
        valid_mask = ~np.isnan(feature_values)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            sequence[:, feature_idx] = 0.0
            continue

        if len(valid_indices) < len(feature_values):
            invalid_indices = np.where(~valid_mask)[0]
            sequence[invalid_indices, feature_idx] = np.interp(
                invalid_indices, valid_indices, feature_values[valid_indices]
            )

    return sequence


def build_references_for_dataset(
    dataset_dir: Path,
    output_dir: Path,
    min_examples_per_class: int = 1,
) -> None:
    """
    Строит эталонные траектории для всех классов из metadata.json.

    Args:
        dataset_dir: директория датасета (в ней лежит metadata.json, scaler.pkl, label_encoder.pkl ...)
        output_dir: куда складывать <class_name>.npy
        min_examples_per_class: минимальное число примеров, чтобы строить эталон
    """
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / "metadata.json"
    scaler_path = dataset_dir / "scaler.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Не найден metadata.json по пути: {metadata_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Не найден scaler.pkl по пути: {scaler_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    sequence_length = metadata["sequence_length"]
    videos: List[Dict] = metadata["videos"]

    if not videos:
        raise ValueError("В metadata['videos'] пусто – нечего использовать как эталон.")

    # Определяем ключи по первому примеру
    label_key, poses_key = _detect_keys(videos[0])
    print(f"Использую ключи: label='{label_key}', poses='{poses_key}'")

    # Загружаем scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_extractor = FeatureExtractor(min_confidence=0.3)

    # Сбор последовательностей по классам
    sequences_by_class: Dict[str, List[np.ndarray]] = {}

    for video_meta in videos:
        cls_name = str(video_meta[label_key])
        poses_rel = Path(str(video_meta[poses_key]))
        poses_path = poses_rel if poses_rel.is_absolute() else (dataset_dir / poses_rel)

        if not poses_path.exists():
            print(f"[WARN] Поза-файл не найден: {poses_path} – пропускаю")
            continue

        with open(poses_path, "r", encoding="utf-8") as f:
            poses_data = json.load(f)

        # Извлекаем фичи
        feature_list, valid_mask = feature_extractor.extract_sequence_features(poses_data)
        feature_array, used_feature_names, valid_indices = feature_extractor.features_to_array(
            feature_list, feature_names
        )

        if len(valid_indices) < sequence_length:
            print(
                f"[INFO] Видео {poses_path.name}: "
                f"валидных кадров {len(valid_indices)} < sequence_length {sequence_length} – пропускаю"
            )
            continue

        # Берём последнюю последовательность нужной длины
        sequence = feature_array[-sequence_length:]
        sequence = _interpolate_missing(sequence)
        sequence_norm = scaler.transform(sequence)

        sequences_by_class.setdefault(cls_name, []).append(sequence_norm)

    # Строим эталоны
    for cls_name, seqs in sequences_by_class.items():
        if len(seqs) < min_examples_per_class:
            print(
                f"[WARN] Класс '{cls_name}': всего {len(seqs)} примеров "
                f"< min_examples_per_class={min_examples_per_class} – пропускаю"
            )
            continue

        print(f"[INFO] Класс '{cls_name}': строю эталон по {len(seqs)} примерам")
        ref_seq = build_reference_from_examples(seqs)
        out_path = output_dir / f"{cls_name}.npy"
        np.save(out_path, ref_seq)
        print(f"  -> сохранено: {out_path}")

    print("\nГотово. Эталонные траектории сохранены в:", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Построение эталонных траекторий фигур (reference_trajectories/*.npy)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Директория датасета (metadata.json, scaler.pkl, JSON с позами)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Куда сохранять эталоны (по умолчанию <dataset-dir>/reference_trajectories)",
    )
    parser.add_argument(
        "--min-examples-per-class",
        type=int,
        default=1,
        help="Минимальное число примеров на класс для построения эталона",
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (dataset_dir / "reference_trajectories")

    build_references_for_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        min_examples_per_class=args.min_examples_per_class,
    )


if __name__ == "__main__":
    main()

