from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from recommender import ImageSimilarityRecommender, InvalidImageError, find_reference_images


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_images"
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    reference_images = find_reference_images(REFERENCE_DIR)
    app.state.reference_images = reference_images
    app.state.recommender = (
        ImageSimilarityRecommender(reference_images) if reference_images else None
    )
    yield


app = FastAPI(title="닮은 캐릭터 이미지 추천", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, bool | int]:
    recommender = app.state.recommender
    return {
        "ready": recommender is not None,
        "character_count": recommender.character_count if recommender else 0,
    }


@app.get("/api/reference-images/{character}/{filename}")
async def reference_image(character: str, filename: str) -> FileResponse:
    reference_images = app.state.reference_images
    image_path = next(
        (
            path
            for path in reference_images.get(character, ())
            if path.name == filename and path.is_file()
        ),
        None,
    )
    if image_path is None:
        raise HTTPException(status_code=404, detail="참고 이미지를 찾을 수 없습니다.")
    return FileResponse(image_path)


@app.post("/api/recommend")
async def recommend(request: Request) -> dict[str, list[dict[str, str | float]]]:
    recommender = app.state.recommender
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="reference_images 폴더에 캐릭터별 이미지 폴더와 참고 이미지를 추가한 뒤 앱을 다시 실행하세요.",
        )
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0]
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="이미지 파일만 업로드할 수 있습니다.")

    image_bytes = bytearray()
    async for chunk in request.stream():
        image_bytes.extend(chunk)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="사진은 10MB 이하로 업로드해 주세요.")

    try:
        results = await run_in_threadpool(recommender.recommend, bytes(image_bytes))
    except InvalidImageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    for result in results:
        character = str(result["character"])
        filename = str(result["image_filename"])
        result["image_url"] = (
            f"/api/reference-images/{quote(character, safe='')}/{quote(filename, safe='')}"
        )
        del result["image_filename"]

    return {"recommendations": results}
