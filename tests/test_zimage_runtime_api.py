import unittest
from unittest import mock

import app as meetnote_app


class ZImageRuntimeApiTests(unittest.TestCase):
    def setUp(self):
        meetnote_app.app.config.update(TESTING=True)
        self.client = meetnote_app.app.test_client()

    def test_runtime_endpoint_reports_current_state(self):
        runtime = {
            "is_loaded": True,
            "loaded_model": "qwenimg",
            "model_label": "Qwen-Image 2.1 (Unsloth GGUF)",
            "busy": False,
            "unloading": False,
        }
        with mock.patch.object(meetnote_app, "zimage_runtime", return_value=runtime):
            response = self.client.get("/api/zimage/runtime")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, runtime)

    def test_unload_endpoint_returns_conflict_when_generation_is_active(self):
        busy = {"status": "busy", "busy": True, "message": "generation active"}
        with mock.patch.object(meetnote_app, "zimage_unload", return_value=busy):
            response = self.client.post("/api/zimage/unload")

        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json["busy"])

    def test_generate_endpoint_rejects_while_model_is_unloading(self):
        with mock.patch.object(meetnote_app, "zimage_generate", return_value="__busy"):
            response = self.client.post(
                "/api/zimage/generate",
                json={"prompt": "a calm mountain lake"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json["code"], "MODEL_UNLOADING")

    def test_text_to_image_page_exposes_manual_unload_control(self):
        response = self.client.get("/text-to-image")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="zi-unload"', response.data)
        self.assertIn(b'id="zi-runtime-status"', response.data)


if __name__ == "__main__":
    unittest.main()
