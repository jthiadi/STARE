"""Optional SAM2 workpiece focus for the sampled-video recognition path.

SAM2 is only a promptable segmenter/tracker. It does not classify SOP steps and
its masks are never treated as completion evidence. An SOP-guided VLM proposal,
or a manual correction box, starts a generation-scoped streaming session; the
resulting crop is passed to the VLM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from PIL import Image

from .settings import SAM2_DIR, SAM2_ID, SAM2_REVISION


def validate_prompt_box(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Draw one SAM2 box around the assembly workpiece.')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('SAM2 box coordinates must be finite numbers.')
    box = [float(v) for v in value]
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('SAM2 box must be inside the image.')
    if x2 - x1 < .03 or y2 - y1 < .03:
        raise ValueError('Draw a larger SAM2 box around the workpiece.')
    return box


def mask_focus(image: Image.Image, mask: np.ndarray, padding=.75):
    """Return a context-preserving crop and normalized full-frame mask bounds."""
    image = image.convert('RGB')
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (image.height, image.width):
        raise ValueError('SAM2 returned a mask with the wrong image size.')
    ys, xs = np.where(mask)
    if not len(xs):
        return None, None, 0.0
    coverage = float(mask.mean())
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    pad_x = max(8, int((x2 - x1) * padding))
    pad_y = max(8, int((y2 - y1) * padding))
    crop_box = (max(0, x1-pad_x), max(0, y1-pad_y), min(image.width, x2+pad_x), min(image.height, y2+pad_y))

    # Keep the crop unmasked. An SOP check often depends on a relationship
    # between the tracked base and nearby braces, fasteners, hands, or tools;
    # erasing those pixels would turn SAM2 into a source of false negatives.
    focused = image.crop(crop_box)
    box = [x1/image.width, y1/image.height, x2/image.width, y2/image.height]
    return focused, box, coverage


def automatic_mask_too_broad(prompt_box, tracked_box):
    """Reject a first automatic mask that simply floods its candidate window."""
    if not tracked_box:
        return True
    prompt_area = (prompt_box[2]-prompt_box[0])*(prompt_box[3]-prompt_box[1])
    tracked_area = (tracked_box[2]-tracked_box[0])*(tracked_box[3]-tracked_box[1])
    near = .03
    touched = sum((abs(tracked_box[0]-prompt_box[0]) <= near,
                   abs(tracked_box[1]-prompt_box[1]) <= near,
                   abs(tracked_box[2]-prompt_box[2]) <= near,
                   abs(tracked_box[3]-prompt_box[3]) <= near))
    return tracked_area >= prompt_area*.9 and touched >= 3


@dataclass
class FocusResult:
    image: Image.Image | None
    box: list[float] | None
    coverage: float = 0.0
    reason: str = ''
    latency_ms: int = 0


class Sam2Focus:
    """Streaming SAM2.1 Hiera-tiny session owned by the single GPU worker."""
    def __init__(self, device='cuda'):
        if not SAM2_DIR.is_dir():
            raise ValueError('SAM2 is not downloaded. Run: .venv/Scripts/python.exe -m scripts.download_sam2')
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('SAM2 focus requires CUDA in this local prototype.')
        self.torch = torch
        self.device = device
        self.dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.model = Sam2VideoModel.from_pretrained(
            SAM2_DIR, local_files_only=True, dtype=self.dtype).to(device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(SAM2_DIR, local_files_only=True)
        self.session = None
        self.prompt_id = None
        self.last_timestamp = None
        self.last_reason = ''

    def status(self):
        return {'ready': True, 'name': 'SAM2.1 Hiera-tiny focus tracker', 'model_id': SAM2_ID,
                'revision': SAM2_REVISION, 'device': self.device,
                'purpose': 'automatically or manually prompted segmentation only; not step completion',
                'tracking': self.session is not None, 'last_reason': self.last_reason}

    def reset(self):
        self.session = None
        self.prompt_id = None
        self.last_timestamp = None
        self.last_reason = ''

    def _new_session(self):
        self.session = self.processor.init_video_session(
            inference_device=self.device, inference_state_device='cpu', video_storage_device='cpu',
            max_vision_features_cache_size=1, dtype=self.dtype)

    def focus(self, image: Image.Image, timestamp: float, prompt: dict, cancel_event=None) -> FocusResult:
        started = time.perf_counter()
        try:
            box = validate_prompt_box(prompt.get('box'))
            prompt_id = str(prompt.get('id', ''))
            anchor = float(prompt.get('timestamp'))
            if not prompt_id or not math.isfinite(anchor) or anchor < 0:
                raise ValueError('SAM2 prompt is invalid; draw the workpiece box again.')
            if self.prompt_id != prompt_id:
                self.reset()
                self.prompt_id = prompt_id
            if cancel_event and cancel_event.is_set():
                return FocusResult(None, None, reason='SAM2 analysis was cancelled.')
            first = self.session is None
            if first and not anchor - .25 <= timestamp <= anchor + 3:
                return FocusResult(None, None, reason='SAM2 needs a frame near the drawn box timestamp. Draw the box again.')
            if self.last_timestamp is not None and timestamp <= self.last_timestamp:
                self.reset()
                return FocusResult(None, None, reason='SAM2 tracking was reset after non-forward media time. Draw the box again.')
            if first:
                self._new_session()

            inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            if first:
                w, h = image.size
                pixel_box = [[[box[0]*w, box[1]*h, box[2]*w, box[3]*h]]]
                original_size = tuple(inputs.original_sizes[0].tolist())
                self.processor.add_inputs_to_inference_session(
                    self.session, frame_idx=0, obj_ids=1, input_boxes=pixel_box,
                    original_size=original_size)
            if cancel_event and cancel_event.is_set():
                return FocusResult(None, None, reason='SAM2 analysis was cancelled.')
            with self.torch.inference_mode():
                output = self.model(inference_session=self.session, frame=inputs.pixel_values[0])
            masks = self.processor.post_process_masks([output.pred_masks.cpu()], inputs.original_sizes)[0]
            mask = masks[0, 0].numpy().astype(bool)
            focused, tracked_box, coverage = mask_focus(image, mask)
            self.last_timestamp = timestamp
            if focused is None or coverage < .0005:
                reason = 'SAM2 lost the selected workpiece in this frame; no step claim was made.'
                self.last_reason = reason
                return FocusResult(None, None, coverage, reason, round((time.perf_counter()-started)*1000))
            if first and prompt.get('source') == 'automatic_vlm' and automatic_mask_too_broad(box, tracked_box):
                reason = 'Automatic SAM2 mask covered the candidate window instead of one clear workpiece.'
                self.reset()
                self.last_reason = reason
                return FocusResult(None, tracked_box, coverage, reason,
                                   round((time.perf_counter()-started)*1000))
            if coverage > .80:
                reason = 'SAM2 selected most of the frame; draw a tighter box around the workpiece.'
                self.last_reason = reason
                return FocusResult(None, tracked_box, coverage, reason, round((time.perf_counter()-started)*1000))
            self.last_reason = ''
            return FocusResult(focused, tracked_box, coverage, latency_ms=round((time.perf_counter()-started)*1000))
        except Exception as error:
            self.last_reason = str(error)[:350] or type(error).__name__
            return FocusResult(None, None, reason=self.last_reason,
                               latency_ms=round((time.perf_counter()-started)*1000))
