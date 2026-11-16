from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLM_ID = "Qwen/Qwen3-VL-2B-Instruct"
VLM_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
VLM_DIR = ROOT / "models/qwen3-vl-2b"
SAM2_ID = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM2_DIR = ROOT / "models/sam2.1-hiera-tiny"
YOLO_WORLD_ID = "yolov8s-worldv2.pt"
YOLO_WORLD_DIR = ROOT / "models/yolov8s-worldv2.pt"
DINO_V2_ID = "facebook/dinov2-small"
DINO_V2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINO_V2_DIR = ROOT / "models/dinov2-small"
