from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "openai/clip-vit-base-patch32"
# 수정: 2026-10-27 16:25 KST - 참고 사진 상위 3장 평균으로 캐릭터 순위를 계산합니다.
REFERENCE_SCORES_TO_AVERAGE = 3


class InvalidImageError(ValueError):
    pass


def find_reference_images(directory: Path) -> dict[str, tuple[Path, ...]]:
    """Find image files grouped by the name of their immediate parent folder."""
    if not directory.is_dir():
        return {}

    characters: dict[str, tuple[Path, ...]] = {}
    for character_dir in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if not character_dir.is_dir():
            continue
        images = tuple(
            sorted(
                (
                    path
                    for path in character_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                ),
                key=lambda path: path.name.casefold(),
            )
        )
        if images:
            characters[character_dir.name] = images
    return characters


def rank_character_scores(
    reference_vectors: Mapping[str, Sequence[Sequence[float]]],
    query_vector: Sequence[float],
    limit: int = 3,
) -> list[tuple[str, float]]:
    """Rank characters by the average of their strongest reference-image scores."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not reference_vectors:
        raise ValueError("At least one reference vector is required")
    if not query_vector:
        raise ValueError("The query vector cannot be empty")
    if not all(math.isfinite(value) for value in query_vector):
        raise ValueError("Vectors must contain only finite values")
    query_norm = math.sqrt(math.fsum(value * value for value in query_vector))
    if query_norm == 0:
        raise ValueError("Vectors cannot have zero magnitude")

    ranked: list[tuple[str, float]] = []
    for character, vectors in reference_vectors.items():
        if not vectors:
            continue
        scores: list[float] = []
        for vector in vectors:
            if len(vector) != len(query_vector):
                raise ValueError("Reference and query vectors must have the same dimensions")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Vectors must contain only finite values")
            vector_norm = math.sqrt(math.fsum(value * value for value in vector))
            if vector_norm == 0:
                raise ValueError("Vectors cannot have zero magnitude")
            dot_product = math.fsum(
                left * right for left, right in zip(vector, query_vector)
            )
            scores.append(dot_product / (vector_norm * query_norm))
        # 수정: 2026-10-27 16:25 KST - 사진 한 장의 우연한 고득점을 줄이도록 상위 3장을 평균냅니다.
        best_scores = sorted(scores, reverse=True)[:REFERENCE_SCORES_TO_AVERAGE]
        ranked.append((character, math.fsum(best_scores) / len(best_scores)))

    return sorted(ranked, key=lambda item: (-item[1], item[0].casefold()))[:limit]


class ImageSimilarityRecommender:
    def __init__(self, reference_images: Mapping[str, Sequence[Path]]) -> None:
        if not reference_images:
            raise ValueError("No reference images were provided")

        import torch
        from PIL import Image, ImageOps
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._image_ops = ImageOps
        self._image_type = Image
        self._processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self._model = CLIPModel.from_pretrained(MODEL_NAME)
        self._model.eval()
        self._reference_images = {
            name: tuple(paths) for name, paths in reference_images.items()
        }
        self._reference_vectors: dict[str, list[list[float]]] = {
            name: [] for name in reference_images
        }

        indexed_paths = [
            (name, path)
            for name, paths in reference_images.items()
            for path in paths
        ]
        for offset in range(0, len(indexed_paths), 8):
            batch = indexed_paths[offset : offset + 8]
            images = [self._open_reference(path) for _, path in batch]
            vectors = self._encode(images).tolist()
            for (name, _), vector in zip(batch, vectors):
                self._reference_vectors[name].append(vector)

    @property
    def character_count(self) -> int:
        return len(self._reference_vectors)

    def _open_reference(self, path: Path) -> Image.Image:
        try:
            with self._image_type.open(path) as image:
                return self._image_ops.exif_transpose(image).convert("RGB")
        except (OSError, self._image_type.UnidentifiedImageError) as error:
            raise ValueError(f"Could not read reference image: {path}") from error

    def _encode(self, images: Sequence[Image.Image]):
        inputs = self._processor(images=list(images), return_tensors="pt")
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            return self._torch.nn.functional.normalize(features, p=2, dim=-1).cpu()

    def recommend(self, image_bytes: bytes, limit: int = 3) -> list[dict[str, str | float]]:
        try:
            with self._image_type.open(BytesIO(image_bytes)) as image:
                query_image = self._image_ops.exif_transpose(image).convert("RGB")
        except (OSError, self._image_type.UnidentifiedImageError) as error:
            raise InvalidImageError("사진을 읽을 수 없습니다. 다시 촬영해 주세요.") from error

        query_vector = self._encode([query_image])[0].tolist()
        ranked = rank_character_scores(self._reference_vectors, query_vector, limit)
        results: list[dict[str, str | float]] = []
        for character, score in ranked:
            matched_image = max(
                zip(
                    self._reference_images[character],
                    self._reference_vectors[character],
                ),
                key=lambda item: math.fsum(
                    left * right for left, right in zip(item[1], query_vector)
                ),
            )[0]
            results.append(
                {
                    "character": character,
                    "score": round(score, 4),
                    "image_filename": matched_image.name,
                }
            )
        return results
