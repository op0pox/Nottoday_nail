import json

from fastapi import APIRouter, File, HTTPException, UploadFile

from classification.contour_compare import (
    CATALOG_TEMPLATES,
    extract_shape_code,
    leave_one_out_fingers,
    nearest_cuticle,
    points_from_labelme,
    visualize_compare_steps,
    visualize_full_shell_steps,
)

router = APIRouter(prefix="/api")
DEFAULT_SAMPLES = 160


# 업로드 JSON에서 폴리곤 좌표
async def read_polygon(upload: UploadFile):
    raw = json.loads((await upload.read()).decode("utf-8"))
    return points_from_labelme(raw)


# 카탈로그 전체와 비교해 P/S 판정
@router.post("/classify-label")
async def classify_label(file: UploadFile = File(...)):
    try:
        query_pts = await read_polygon(file)
        result = nearest_cuticle(query_pts, CATALOG_TEMPLATES, DEFAULT_SAMPLES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {exc}") from exc

    template = result["predicted_shape"]
    return {
        "shape": extract_shape_code(template),
        "template": template,
        "distance": result["distance"],
    }


# fingers 전체 leave-one-out 채점
@router.post("/classify-loo")
async def classify_loo():
    try:
        return leave_one_out_fingers(DEFAULT_SAMPLES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"검증 실패: {exc}") from exc


# 큐티클 비교 4단계 그림
@router.post("/classify-viz")
async def classify_viz(
    green_json: UploadFile = File(...),
    red_json: UploadFile = File(...),
):
    try:
        green_pts = await read_polygon(green_json)
        red_pts = await read_polygon(red_json)
        stages = visualize_compare_steps(red_pts, green_pts, DEFAULT_SAMPLES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"시각화 실패: {exc}") from exc

    return {
        "green_name": green_json.filename or "1번째 JSON",
        "red_name": red_json.filename or "2번째 JSON",
        "stages": stages,
    }


# 전체 쉘 비교 3단계 그림
@router.post("/classify-viz-shell")
async def classify_viz_shell(
    green_json: UploadFile = File(...),
    red_json: UploadFile = File(...),
):
    try:
        green_pts = await read_polygon(green_json)
        red_pts = await read_polygon(red_json)
        result = visualize_full_shell_steps(red_pts, green_pts, DEFAULT_SAMPLES)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"시각화 실패: {exc}") from exc

    return {
        "green_name": green_json.filename or "1번째 JSON",
        "red_name": red_json.filename or "2번째 JSON",
        "distance": result["distance"],
        "stages": result["stages"],
    }
