# -*- coding: utf-8 -*-
"""
Jetson 촬영 사진을 사람별로 저장한다. utils/JetsonCheck.py 와 같은 폴더·명단을 쓴다.

명단 : utils/data/dataset_jetson.csv (순번, 이름). 사람 추가는 JetsonCheck.py 에서 한다.
폴더 : utils/jetson_capture/<순번 또는 이름>/  (없으면 순번 폴더를 만든다)
파일 : 순번_손_손가락_side.jpg / 순번_손_손가락_front.jpg
       손 L(왼손) R(오른손), 손가락 01 엄지 02 검지 03 중지 04 약지 05 소지
       예: 001_L_01_side.jpg, 001_L_01_front.jpg
       관리자 페이지에서 손·손가락을 골라 저장한다. 이미 있으면 overwrite 일 때만 덮어쓴다.

docker 로 띄우면 docker-compose.yml 이 ../utils 를 /app/utils 로 붙이고 UTILS_DIR 로 알려 준다.
"""
import csv
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(prefix="/api/capture")

UTILS_DIR = Path(os.getenv("UTILS_DIR") or Path(__file__).resolve().parents[3] / "utils")
PEOPLE_CSV = UTILS_DIR / "data" / "dataset_jetson.csv"
CAPTURE_DIR = UTILS_DIR / "jetson_capture"

HANDS = ("L", "R")
FINGERS = ("01", "02", "03", "04", "05")
FINGER_KEYS = ["%s_%s" % (hand, finger) for hand in HANDS for finger in FINGERS]
VIEWS = ("side", "front")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class Person(BaseModel):
    id: str
    name: str
    saved: List[str]


class SaveResult(BaseModel):
    id: str
    key: str
    overwritten: bool
    files: List[str]
    saved: List[str]


def read_people():
    if not PEOPLE_CSV.is_file():
        return []
    with open(PEOPLE_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    people = []
    for row in rows:
        pid = (row.get("순번") or "").strip()
        if pid:
            people.append((pid, (row.get("이름") or "").strip()))
    return people


def person_dirs(pid, name):
    return [CAPTURE_DIR / key for key in (pid, name) if key and (CAPTURE_DIR / key).is_dir()]


def scan_views(pid, name):
    """손가락별로 저장된 view 집합. {'L_01': {'side', 'front'}, ...}"""
    prefixes = {key.upper() for key in (pid, name) if key}
    found = {}
    for folder in person_dirs(pid, name):
        for path in folder.iterdir():
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            parts = path.stem.split("_")
            if len(parts) < 4 or "_".join(parts[:-3]).upper() not in prefixes:
                continue
            key = "%s_%s" % (parts[-3].upper(), parts[-2])
            view = parts[-1].lower()
            if key in FINGER_KEYS and view in VIEWS:
                found.setdefault(key, set()).add(view)
    return found


def saved_keys(pid, name):
    """정면·측면이 둘 다 있는 손가락 목록 (L_01 ... R_05 순서)."""
    found = scan_views(pid, name)
    return [key for key in FINGER_KEYS if set(VIEWS) <= found.get(key, set())]


def find_person(pid):
    for person_id, name in read_people():
        if person_id == pid:
            return person_id, name
    raise HTTPException(status_code=404, detail="명단에 없는 순번입니다: %s" % pid)


@router.get("/people", response_model=List[Person])
def list_people():
    return [Person(id=pid, name=name, saved=saved_keys(pid, name)) for pid, name in read_people()]


@router.post("/save", response_model=SaveResult)
async def save_shot(
    person: str = Form(...),
    hand: str = Form(...),
    finger: str = Form(...),
    overwrite: bool = Form(False),
    side: UploadFile = File(...),
    front: UploadFile = File(...),
):
    pid, name = find_person(person.strip())
    hand = hand.strip().upper()
    finger = finger.strip().zfill(2)
    if hand not in HANDS or finger not in FINGERS:
        raise HTTPException(status_code=400, detail="손은 L/R, 손가락은 01~05 여야 합니다.")
    key = "%s_%s" % (hand, finger)

    exists = bool(scan_views(pid, name).get(key))
    if exists and not overwrite:
        raise HTTPException(status_code=409, detail="%s_%s 사진이 이미 있습니다." % (pid, key))

    dirs = person_dirs(pid, name)
    folder = dirs[0] if dirs else CAPTURE_DIR / pid
    folder.mkdir(parents=True, exist_ok=True)

    files = []
    for view, upload in (("side", side), ("front", front)):
        data = await upload.read()
        if not data:
            raise HTTPException(status_code=400, detail="%s 사진이 비어 있습니다." % view)
        path = folder / ("%s_%s_%s.jpg" % (pid, key, view))
        path.write_bytes(data)
        files.append(str(path.relative_to(UTILS_DIR)).replace("\\", "/"))

    return SaveResult(id=pid, key=key, overwritten=exists, files=files, saved=saved_keys(pid, name))
