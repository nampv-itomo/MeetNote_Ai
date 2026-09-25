import base64
import os
import tempfile
import unittest
from unittest import mock

import zimage


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakeResponse:
    status_code = 200

    def __init__(self, payload=None):
        self.payload = payload or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class ZImageServerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.output_dir.cleanup)
        self.process = FakeProcess()
        self.patches = [
            mock.patch.object(zimage, "ZIMAGE_SD_SERVER", "/fake/sd-server"),
            mock.patch.object(zimage, "ZIMAGE_SERVER_PORT", 1234),
            mock.patch.object(zimage, "ZIMAGE_SERVER_START_TIMEOUT", 1),
            mock.patch.object(zimage, "ZIMAGE_OUTPUT_FOLDER", self.output_dir.name),
            mock.patch.object(zimage.subprocess, "Popen", return_value=self.process),
            mock.patch.object(zimage.requests, "get", return_value=FakeResponse()),
            mock.patch.object(
                zimage.requests,
                "post",
                return_value=FakeResponse({
                    "images": [base64.b64encode(b"fake-png").decode("ascii")]
                }),
            ),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        with zimage._lock:
            zimage._jobs.clear()
            zimage._server_proc = None
            zimage._loaded_model_key = None
            zimage._unloading = False

    def tearDown(self):
        zimage.shutdown_server()

    def _run_worker(self, job_id):
        with zimage._lock:
            zimage._jobs[job_id] = {"status": "queued", "progress": 0}
        zimage._worker(
            job_id, "a quiet mountain lake", 512, 512, 20, 6.0, -1,
            "/models/qwen.gguf", "qwenimg",
        )

    def test_successful_generations_reuse_server_until_manual_unload(self):
        self._run_worker("first")
        self._run_worker("second")

        self.assertEqual(zimage.subprocess.Popen.call_count, 1)
        self.assertEqual(zimage.requests.post.call_count, 2)
        self.assertEqual(zimage._jobs["first"]["status"], "completed")
        self.assertEqual(zimage._jobs["second"]["status"], "completed")
        self.assertTrue(os.path.isfile(zimage._jobs["first"]["output_path"]))
        self.assertTrue(zimage.get_runtime_status()["is_loaded"])
        self.assertFalse(self.process.terminated)

        result = zimage.unload_model()

        self.assertEqual(result["status"], "unloaded")
        self.assertTrue(self.process.terminated)
        self.assertFalse(zimage.get_runtime_status()["is_loaded"])

    def test_unload_refuses_while_a_job_is_queued_or_running(self):
        with zimage._lock:
            zimage._jobs["active"] = {"status": "queued", "progress": 0}

        result = zimage.unload_model()

        self.assertTrue(result["busy"])
        self.assertFalse(self.process.terminated)

    def test_switching_models_releases_previous_server_before_loading_next(self):
        second_process = FakeProcess()
        zimage.subprocess.Popen.side_effect = [self.process, second_process]
        self._run_worker("qwen")
        with zimage._lock:
            zimage._jobs["zimage"] = {"status": "queued", "progress": 0}

        zimage._worker(
            "zimage", "a quiet mountain lake", 512, 512, 8, 1.0, -1,
            "/models/zimage.gguf", "zimage",
        )

        self.assertTrue(self.process.terminated)
        self.assertEqual(zimage.subprocess.Popen.call_count, 2)
        self.assertEqual(zimage.get_runtime_status()["loaded_model"], "zimage")

    def test_failed_generation_keeps_server_loaded_for_retry(self):
        zimage.requests.post.return_value = FakeResponse({"images": []})

        self._run_worker("failed")

        self.assertEqual(zimage._jobs["failed"]["status"], "error")
        self.assertTrue(zimage.get_runtime_status()["is_loaded"])
        self.assertFalse(self.process.terminated)


if __name__ == "__main__":
    unittest.main()
