# -*- coding: utf-8 -*-
"""
collect.py
===========
[역할]
    이미지 쌍(정면/측면 원본 고해상도) + 자동 검출 마스크(PNG, 세그멘테이션
    학습 표준 포맷: 이미지와 동일 크기의 단일 채널 0/255 PNG) + 키포인트/
    메타데이터(JSON)를 세션 폴더에 저장한다.

    main.py의 `collect` 서브커맨드와 `measure --save-data` 옵션이 공통으로
    이 모듈의 CollectSession을 사용한다 (측정이 곧 데이터 수집이 되도록).

    저장된 마스크는 나중에 PC에서 SAM(Segment Anything) 기반 반자동
    라벨링의 초기 후보로 쓰거나, 사람이 직접 검수/수정해서 YOLOv8n-seg /
    U-Net 학습 데이터로 바로 쓸 수 있는 포맷이다 (README의 로드맵 참고).

[폴더/파일명 규칙]
    data/collect/session_YYYYMMDD_HHMMSS/
        R_thumb_front.jpg
        R_thumb_front_mask.png
        R_thumb_side.jpg
        R_thumb_side_mask.png
        R_thumb_meta.json
        R_index_front.jpg
        ...

[실행 위치] 직접 실행하는 파일이 아니다. main.py가 불러와서 쓴다.
"""

import json
import os
from datetime import datetime

import cv2


class CollectSession(object):
    """세션 하나(보통 한 사람, 한 회차) 동안의 데이터 수집을 관리한다."""

    def __init__(self, config, session_name=None):
        self.config = config
        base_dir = config["paths"]["collect_dir"]
        self.session_name = session_name or ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.session_dir = os.path.join(base_dir, self.session_name)
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

    def save_finger(
        self,
        finger_id,
        front_image,
        side_image,
        front_mask=None,
        side_mask=None,
        front_keypoints_px=None,
        side_keypoints_px=None,
        meta_extra=None,
    ):
        """
        finger_id: "R_thumb" 등 (config.yaml fingers.order 참고)
        front_image/side_image: 원본 고해상도 BGR (numpy array)
        front_mask/side_mask: HxW uint8 0/255 또는 None (없으면 마스크 파일은 저장 안 함)
        front_keypoints_px/side_keypoints_px: {"1": (x,y), ...} 또는 None
        meta_extra: 결과 JSON에 추가로 합칠 dict (계산 결과 등)

        반환: 저장된 파일 경로들을 담은 dict
        """
        paths = {}

        front_path = os.path.join(self.session_dir, "%s_front.jpg" % finger_id)
        side_path = os.path.join(self.session_dir, "%s_side.jpg" % finger_id)
        cv2.imwrite(front_path, front_image)
        cv2.imwrite(side_path, side_image)
        paths["front_image"] = front_path
        paths["side_image"] = side_path

        if front_mask is not None:
            front_mask_path = os.path.join(self.session_dir, "%s_front_mask.png" % finger_id)
            cv2.imwrite(front_mask_path, front_mask)
            paths["front_mask"] = front_mask_path

        if side_mask is not None:
            side_mask_path = os.path.join(self.session_dir, "%s_side_mask.png" % finger_id)
            cv2.imwrite(side_mask_path, side_mask)
            paths["side_mask"] = side_mask_path

        meta = {
            "finger_id": finger_id,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "front_keypoints_px": {k: list(v) for k, v in (front_keypoints_px or {}).items()},
            "side_keypoints_px": {k: list(v) for k, v in (side_keypoints_px or {}).items()},
            "files": {k: os.path.basename(v) for k, v in paths.items()},
        }
        if meta_extra:
            meta.update(meta_extra)

        meta_path = os.path.join(self.session_dir, "%s_meta.json" % finger_id)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        paths["meta"] = meta_path

        return paths

    def list_saved_fingers(self):
        """이 세션에서 지금까지 저장된 finger_id 목록 (meta.json 파일 기준)."""
        if not os.path.exists(self.session_dir):
            return []
        fingers = []
        for name in os.listdir(self.session_dir):
            if name.endswith("_meta.json"):
                fingers.append(name[: -len("_meta.json")])
        return sorted(fingers)
