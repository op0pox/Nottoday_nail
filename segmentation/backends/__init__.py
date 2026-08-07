"""
backends 패키지
================
손톱 자동 검출용 세그멘테이션 백엔드 모음이다. 모든 백엔드는
common.py의 SegmentationBackend 인터페이스(입력: BGR 이미지,
출력: NailMask 리스트)를 따른다.

- dl_yolo.py           : (메인) 사전학습 YOLOv8-seg 모델 기반 자동 검출
- classical.py         : 피부색/명도 기반 폴백 백엔드
- measure_from_mask.py : 마스크에서 길이/폭 후보점 추출 + ChArUco 호모그래피로 mm 계산
"""
