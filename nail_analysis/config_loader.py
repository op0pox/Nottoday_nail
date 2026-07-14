# -*- coding: utf-8 -*-
"""
config_loader.py
=================
[역할] config.yaml(모든 파라미터/상수/사이즈테이블)을 읽어서 dict로 반환한다.
Python 3.6 호환을 위해 dataclass 대신 plain dict를 그대로 사용한다.

이 파일은 직접 실행하지 않는다. 다른 모듈에서
    from config_loader import load_config
형태로 불러와서 쓴다.
"""

import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(path=None):
    """
    config.yaml을 읽어 dict로 반환한다. path를 생략하면 nail_analysis/config.yaml을 쓴다.
    """
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise RuntimeError(
            "설정 파일을 찾을 수 없습니다: %s\n"
            "config.yaml이 nail_analysis/ 폴더에 있는지, --config 경로가 맞는지 확인하세요." % path
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise RuntimeError("설정 파일이 비어있습니다: %s" % path)
    return config


def config_snapshot(config):
    """
    결과 JSON에 "이 측정에 어떤 설정으로 계산했는지"를 함께 남기기 위한
    스냅샷. 지금은 config 전체를 그대로 복사해서 반환한다(참조 안전).
    """
    import copy

    return copy.deepcopy(config)
