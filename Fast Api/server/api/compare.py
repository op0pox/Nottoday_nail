import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import List

from classification.contour_compare import SAMPLE_CHOICES, nearest_cuticle, points_from_labelme

router = APIRouter(prefix="/api")


@router.post("/compare")
async def compare_labels(
    left_json: List[UploadFile] = File(...),
    right_json: UploadFile = File(...),
    n_samples: int = Form(160),
):
    if not left_json:
        raise HTTPException(status_code=400, detail="왼쪽 라벨 JSON이 없습니다")

    try:
        templates = []
        for item in left_json:
            raw = json.loads((await item.read()).decode("utf-8"))
            name = Path(item.filename or "left").stem
            templates.append((name, points_from_labelme(raw)))
        right_data = json.loads((await right_json.read()).decode("utf-8"))
        right_pts = points_from_labelme(right_data)
        result = nearest_cuticle(right_pts, templates, int(n_samples))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {exc}") from exc

    return {
        "n_choices": SAMPLE_CHOICES,
        **result,
    }
