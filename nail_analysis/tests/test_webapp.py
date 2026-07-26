# -*- coding: utf-8 -*-
"""
tests/test_webapp.py
======================
[역할] webapp/(읽기 전용 대시보드)의 라우트와 path-traversal 방어를 검증한다.

실행 명령어:
    cd nail_analysis
    python -m unittest tests.test_webapp -v

포함 내용:
    1. 정상 라우트(index/collect/measure/analyze + 이미지 서빙)가 200을 반환하고
       기대하는 내용을 담고 있는지 확인.
    2. 존재하지 않는 세션/finger_id/timestamp, 그리고 상위 경로 조작(../, 인코딩된
       ../)이 전부 404로 막히는지 확인 (읽기 전용 대시보드가 data/ 바깥 파일을
       절대 서빙하지 않아야 한다).

config.yaml 전체(tip_size_table, fingers.order 등)를 그대로 쓰되 paths만
tempfile로 격리해서, 실제 data/ 디렉터리를 건드리지 않는다.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config  # noqa: E402
import generate_fake_dashboard_data as fakegen  # noqa: E402
from webapp.app import create_app  # noqa: E402


class WebappRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="nail_analysis_webapp_test_")

        cls.config = load_config()
        cls.config["paths"] = dict(cls.config["paths"])
        cls.config["paths"]["collect_dir"] = os.path.join(cls.tmp_dir, "collect")
        cls.config["paths"]["results_dir"] = os.path.join(cls.tmp_dir, "results")

        fakegen.random.seed(1)
        cls.session_name = fakegen.generate_collect_session(cls.config, session_offset=0, num_fingers=3)
        fakegen.generate_measure_summary(cls.config, num_fingers=3)
        fakegen.generate_analyze_result(cls.config, "analyze_R_thumb")

        cls.finger_id = cls.config["fingers"]["order"][0]

        # measure_*.json의 실제 timestamp를 파일명에서 그대로 읽어온다.
        results_dir = cls.config["paths"]["results_dir"]
        measure_files = [f for f in os.listdir(results_dir) if f.startswith("measure_")]
        cls.measure_timestamp = measure_files[0][len("measure_") : -len(".json")]

        app = create_app(cls.config)
        app.testing = True
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # -- 정상 라우트 ---------------------------------------------------
    def test_index_lists_all_sections(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn(self.session_name, body)
        self.assertIn(self.measure_timestamp, body)
        self.assertIn("analyze_R_thumb", body)

    def test_collect_session_detail_and_image(self):
        resp = self.client.get("/collect/%s" % self.session_name)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn(self.finger_id, body)
        self.assertIn("%s_front.jpg" % self.finger_id, body)

        img_resp = self.client.get(
            "/collect/%s/img/%s_front.jpg" % (self.session_name, self.finger_id)
        )
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.content_type, "image/jpeg")
        img_resp.close()

    def test_measure_detail_has_no_images(self):
        resp = self.client.get("/measure/%s" % self.measure_timestamp)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertNotIn("<img", body)

    def test_analyze_detail_and_overlay(self):
        resp = self.client.get("/analyze/analyze_R_thumb")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        self.assertIn("overlay.jpg", body)

        img_resp = self.client.get("/analyze/analyze_R_thumb/overlay.jpg")
        self.assertEqual(img_resp.status_code, 200)
        self.assertEqual(img_resp.content_type, "image/jpeg")
        img_resp.close()

    # -- path-traversal / 404 방어 --------------------------------------
    def test_nonexistent_session_is_404(self):
        resp = self.client.get("/collect/does_not_exist")
        self.assertEqual(resp.status_code, 404)

    def test_nonexistent_measure_run_is_404(self):
        resp = self.client.get("/measure/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_nonexistent_analyze_result_is_404(self):
        resp = self.client.get("/analyze/nonexistent")
        self.assertEqual(resp.status_code, 404)

    def test_session_name_traversal_is_404(self):
        resp = self.client.get("/collect/..%2f..%2f..%2fmain.py")
        self.assertEqual(resp.status_code, 404)

    def test_image_filename_traversal_is_404(self):
        resp = self.client.get(
            "/collect/%s/img/..%%2f..%%2fmain.py" % self.session_name
        )
        self.assertEqual(resp.status_code, 404)

    def test_analyze_finger_id_traversal_is_404(self):
        resp = self.client.get("/analyze/..%2f..%2frequirements")
        self.assertEqual(resp.status_code, 404)

    def test_image_route_rejects_file_outside_session(self):
        # 다른 세션에 실제 존재하는 이미지라도, 이 세션 디렉터리 안에 없으면 404.
        resp = self.client.get("/collect/%s/img/nonexistent_file.jpg" % self.session_name)
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
