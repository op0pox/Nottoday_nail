from classification.nail_classification import chamfer_distance

OVERLAY_TITLE = None  # 단계 그림 외에 따로 보여줄 그림이 없다


# 정렬·절단된 두 점집합의 양방향 평균 최근접 거리
def score(prepared):
    # region이 shell이면 band가 곧 전체 윤곽이라 쉘 전체를 재게 된다
    return chamfer_distance(prepared["band_a"], prepared["band_b"])


# 외곽선 방식은 마지막 단계 그림이 곧 비교 대상이라 더 그릴 게 없다
def overlay_png(prepared):
    return None
