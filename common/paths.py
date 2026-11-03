"""꽃 저장소 공용 경로와 opencv 우회.

cv2, ultralytics보다 먼저 가져온다. 서버의 opencv 4.7/4.11 혼합 설치에서 cv2.typing을 읽으면 AttributeError가 나서,
새 프로세스의 import ultralytics가 실패한다. 이 모듈이 그 하위 모듈을 건너뛰게 한다.
"""
import importlib.abc
import sys
from pathlib import Path


class _BlockCv2Typing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cv2.typing":
            raise ImportError("cv2.typing skipped")
        return None


if not any(type(f).__name__ == "_BlockCv2Typing" for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockCv2Typing())

WEIGHTS = Path(__file__).resolve().parent / "weights"  # 사전학습 가중치 (yolo26s-seg, yolo26n, sam2.1_s, efficientnet_v2_s)
