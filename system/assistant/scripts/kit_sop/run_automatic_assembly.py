"""No-click SOP/text -> SAM3 instance masks + causal-window Qwen observations.

Stages are separate GPU processes. No camera labels, instance clicks, reference
photos or evaluation annotation files are read. Rendering never fabricates masks
between inference frames: the exported FPS is the actual detection sample rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from monitor.settings import ROOT


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_frame(cap, timestamp):
    # Seek slightly before EOF; OpenCV often fails exactly on the last timestamp.
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(
        float(cap.get(cv2.CAP_PROP_FPS) or 1), 1e-6)
    if duration > 0:
        timestamp = min(float(timestamp), max(0.0, duration - 1e-3))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ok, frame = cap.read()
    if not ok and duration > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, duration - 0.05) * 1000)
        ok, frame = cap.read()
    if not ok:
        raise ValueError(f'Cannot decode frame at {timestamp:.2f}s')
    return frame


def step_context(sop):
    # Whitelist fields: evaluation_sources and visual-reference paths never reach inference.
    return [{k: step.get(k, []) if k in ('parts', 'tools', 'prerequisites')
             else step.get(k, '') for k in
             ('id', 'instruction', 'parts', 'tools', 'prerequisites', 'completion_check')}
            for step in sop['steps']]


def make_plan(args):
    sop = json.loads(args.sop.read_text(encoding='utf-8'))
    from monitor.sop_objects import detection_plan_from_sop

    # Prefer deterministic SOP-derived objects whenever parts/visual_plans exist.
    # This is the factory path: the SOP names the station objects; detectors only
    # search for those names. VLM drafting remains a fallback for bare instructions.
    try:
        plan = detection_plan_from_sop(sop, sop_path=str(args.sop))
        plan['sop_sha256'] = digest(args.sop)
        write_json(args.output / 'plan.json', plan)
        print(json.dumps(plan['classes']), flush=True)
        return
    except ValueError:
        pass

    from monitor.general import GeneralPredictor
    predictor = GeneralPredictor(None)
    schema = {'type': 'object', 'required': ['classes'], 'additionalProperties': False,
              'properties': {'classes': {'type': 'array', 'minItems': 6, 'maxItems': 9,
                'items': {'type': 'string', 'maxLength': 40}}}}
    prompt = ('Create a compact object vocabulary for automatic text-prompted segmentation from this SOP. '
              'Return 6 to 9 SHORT STRINGS in the classes array. Cover the chassis, structural parts, '
              'wheels, fasteners and tools that the SOP actually mentions. '
              'List distinct visually distinguishable physical object types. '
              'Names/prompts must be short noun phrases (2-5 words). Collapse duplicates and '
              'front/rear synonyms that look identical. Include tools mentioned in the SOP. '
              'Use ordinary names, not step IDs. For generic retaining hardware include nut and washer '
              'as SEARCH HYPOTHESES, not confirmed presence. Do not use actions, whole scenes, '
              'piles or count phrases. Do not invent colors absent from the SOP. '
              'These are search queries, not evidence that any object is visible. SOP DATA:\n'
              + json.dumps(step_context(sop)))
    raw = predictor.generate(prompt, max_tokens=160, schema=schema)
    classes = []
    for row in raw['classes']:
        name, query = str(row).strip()[:50], str(row).strip()[:80]
        if name and query and name.casefold() not in {r['name'].casefold() for r in classes}:
            classes.append({'name': name, 'prompt': query, 'source': 'SOP language model draft'})
    classes.append({'name': 'hand', 'prompt': 'human hand', 'source': 'generic interaction context'})
    for tool in ('screwdriver', 'wrench', 'paper'):
        if tool not in {row['name'] for row in classes}:
            classes.append({'name': tool, 'prompt': tool,
                            'source': 'generic tool search hypothesis, not SOP-required or confirmed present'})
    write_json(args.output / 'plan.json', {
        'classes': classes, 'sop': str(args.sop), 'sop_sha256': digest(args.sop),
        'steps': step_context(sop), 'runtime_clicks': 0, 'reference_images': 0,
        'ground_truth_used': False, 'planner': predictor.status()['model_id']})
    print(json.dumps(classes), flush=True)


def load_sam3(checkpoint):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache/ultralytics_auto'))
    import torch
    from ultralytics.models.sam import build_sam3
    from ultralytics.models.sam.predict import SAM3SemanticPredictor

    checkpoint = Path(checkpoint)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = os.environ.get('SAM3_CUDNN_BENCHMARK', '1') == '1'

    if checkpoint.suffix.lower() == '.safetensors':
        # Local SAM3 safetensors mirror: validate every inference weight.
        from safetensors.torch import load_file

        def load_checked(model, path, interactive=False):
            state = load_file(str(path), device='cpu')
            if any(key.startswith('detector.') for key in state):
                state = {key.removeprefix('detector.'): value for key, value in state.items()
                         if key.startswith('detector.')}
            expected = model.state_dict()
            # SAM 3.1 multiplex has no detector sam2_convs (interactive point prompts only) or convs.3
            # (level removed by scalp=1); the native loader leaves both at init too.
            unused = ('.sam2_convs.', 'vision_backbone.convs.3.')
            missing = {k for k in set(expected) - set(state) if not any(u in k for u in unused)}
            mismatched = [key for key in set(state) & set(expected)
                          if state[key].shape != expected[key].shape]
            if missing or mismatched:
                raise ValueError(
                    f'Incomplete SAM3 checkpoint: missing={list(missing)[:8]}, '
                    f'shapes={mismatched[:8]}')
            model.load_state_dict({k: state[k] for k in expected if k in state}, strict=False)
            return model.half()

        # Build in fp16 so the randomly initialised model never needs fp32 host memory.
        dtype = torch.get_default_dtype()
        if torch.cuda.is_available():
            torch.set_default_dtype(torch.float16)
        try:
            with patch.object(build_sam3, '_load_checkpoint', load_checked):
                model = build_sam3.build_sam3_image_model(str(checkpoint))
        finally:
            torch.set_default_dtype(dtype)
    else:
        # Official Meta .pt checkpoints (SAM 3 / SAM 3.1 multiplex): native loader.
        model = build_sam3.build_sam3_image_model(str(checkpoint))
        if torch.cuda.is_available():
            model = model.half()

    predictor = SAM3SemanticPredictor(overrides={
        'model': str(checkpoint), 'device': '0', 'half': True, 'imgsz': 1008,
        'conf': .4, 'save': False, 'verbose': False, 'task': 'segment', 'mode': 'predict'})
    predictor.setup_model(model=model, verbose=False)
    return predictor


def mask_iou(a, b):
    intersection = np.count_nonzero(a & b)
    return intersection / max(1, np.count_nonzero(a | b))


def latest_at(rows, timestamp):
    """Only already-observed records may contribute to the current prediction."""
    return max((row for row in rows if row['timestamp_s'] <= timestamp),
               key=lambda row: row['timestamp_s'], default=None)


def interaction_crop(frame, record):
    """Focus a VLM crop using automatically found hands, never a stored ROI."""
    hands = [row['box'] for row in (record or {}).get('objects', []) if row['class'] == 'hand']
    if not hands:
        return frame
    width, height = frame.size
    x1 = max(0, min(row[0] for row in hands)-width*.07)
    y1 = max(0, min(row[1] for row in hands)-height*.10)
    x2 = min(width, max(row[2] for row in hands)+width*.07)
    y2 = min(height, max(row[3] for row in hands)+height*.10)
    if x2-x1 < 50 or y2-y1 < 50:
        return frame
    return frame.crop((int(x1),int(y1),int(x2),int(y2)))


def filter_detections(rows, *, frame_shape=None):
    """Size priors + wheel/washer conflict rules + duplicate suppression."""
    from monitor.detection_filters import refine_detections
    frame_area = None
    if frame_shape is not None and len(frame_shape) >= 2:
        frame_area = float(frame_shape[0] * frame_shape[1])
    return refine_detections(rows, frame_area=frame_area)


def detect(args):
    plan = json.loads((args.output / 'plan.json').read_text())
    sop = json.loads(args.sop.read_text(encoding='utf-8'))
    from monitor.sop_detect_scope import scoped_plan_classes
    from monitor.step_rules import (
        advance_progress_memory,
        build_step_rules,
        infer_step_from_detections,
    )

    predictor = load_sam3(args.checkpoint)
    rules = build_step_rules(sop)
    completed: set[str] = set()
    memory: dict = {}
    scoped = bool(getattr(args, 'scoped_detect', True))
    full_classes = list(plan['classes'])
    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(source_count / source_fps, args.start + args.duration)
    records, previous, next_id = [], [], 1
    mask_dir = args.output / 'masks'
    mask_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    samples = int(math.ceil((end - args.start) * args.fps))
    start_index = 0
    detections_path = args.output / 'detections.json'
    if getattr(args, 'resume', False) and detections_path.is_file():
        prior = json.loads(detections_path.read_text(encoding='utf-8'))
        prior_records = list(prior.get('records') or [])
        if prior.get('finalized'):
            print(json.dumps({'resume': 'already_finalized', 'records': len(prior_records)}), flush=True)
            cap.release()
            return
        if prior_records:
            records = prior_records
            start_index = len(records)
            last = records[-1]
            completed = set(last.get('completed_steps') or [])
            step = last.get('inferred_step')
            if step not in {None, 'unknown', ''}:
                memory['active_step'] = step
                memory['active_step_display'] = step
            # Restore ids + previous masks for temporal association.
            max_id = 0
            previous = []
            mask_path = mask_dir / f"{int(last['index']):05d}.npz"
            archive = np.load(mask_path) if mask_path.is_file() else None
            try:
                for obj in last.get('objects') or []:
                    obj_id = int(obj.get('id') or 0)
                    max_id = max(max_id, obj_id)
                    row = dict(obj)
                    if archive is not None and str(obj_id) in archive.files:
                        row['mask'] = archive[str(obj_id)]
                    previous.append(row)
            finally:
                if archive is not None:
                    archive.close()
            next_id = max_id + 1
            print(json.dumps({
                'resume': True,
                'from_index': start_index,
                'of': samples,
                'completed': sorted(completed),
                'active_step': memory.get('active_step'),
            }), flush=True)
    for index in range(start_index, samples):
        timestamp = args.start + index / args.fps
        frame = read_frame(cap, timestamp)
        tick = time.perf_counter()
        if scoped:
            query_classes, scope_meta = scoped_plan_classes(
                sop,
                full_classes,
                completed=completed,
                current_step=memory.get('active_step'),
                # Ready ∪ current only — no successor-part lookahead.
                include_next=False,
                always_parts=tuple(getattr(args, 'always_search_parts', ()) or ()),
            )
        else:
            query_classes, scope_meta = full_classes, {
                'ready': [], 'current': [], 'next': [], 'search': [],
                'parts': [row.get('name') for row in full_classes],
                'class_names': [row.get('name') for row in full_classes],
                'scoped': False,
            }
        if scoped:
            scope_meta = {**scope_meta, 'scoped': True}
        # Expand rare-part prompts (bracket / screw pin) for better recall.
        from monitor.ar_objects import detector_prompt_variants_for_part
        expanded = []
        seen_prompts = set()
        for row in query_classes:
            variants = detector_prompt_variants_for_part(row.get('name') or '')
            prompts = variants or [row.get('prompt') or row.get('name')]
            for prompt in prompts[:2]:
                key = (row.get('name'), prompt)
                if key in seen_prompts:
                    continue
                seen_prompts.add(key)
                expanded.append({**row, 'prompt': prompt[:80]})
        query_classes = expanded or query_classes
        # Cache one image backbone, then bound decoder batches to fit laptop VRAM.
        predictor.set_image(frame)
        rows = []
        batch = 3  # denser GPU use; falls back safely if a batch is empty
        for first in range(0, len(query_classes), batch):
            group = query_classes[first:first + batch]
            result = predictor(text=[row['prompt'] for row in group])[0]
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy().astype(bool)
            for mask, box in zip(masks, result.boxes.data.cpu().numpy()):
                class_info = group[int(box[5])] if 0 <= int(box[5]) < len(group) else None
                if class_info is None:
                    continue
                area = float(mask.mean())
                min_area = float(getattr(args, 'min_mask_area', 0.00004) or 0.00004)
                if area < min_area or (class_info['name'] not in ('hand', 'paper') and area > .18):
                    continue
                score = float(box[4])
                min_score = float(getattr(args, 'min_det_score', 0.0) or 0.0)
                if score < min_score:
                    continue
                rows.append({'class': class_info['name'], 'score': score,
                             'mask': mask, 'box': [float(v) for v in box[:4]]})
        predictor.reset_image()
        rows = filter_detections(rows, frame_shape=frame.shape)
        used = set()
        for row in rows:
            matches = [(mask_iou(row['mask'], old['mask']), old['id'])
                       for old in previous
                       if old.get('class') == row['class'] and old.get('id') not in used
                       and old.get('mask') is not None]
            score, old_id = max(matches, default=(0, None))
            if score >= .25:
                row['id'] = old_id
            else:
                row['id'] = next_id
                next_id += 1
            used.add(row['id'])
        if rows:
            np.savez_compressed(mask_dir / f'{index:05d}.npz',
                                **{str(row['id']): row['mask'] for row in rows})
        else:
            np.savez_compressed(mask_dir / f'{index:05d}.npz')
        objects = [{key: value for key, value in row.items() if key != 'mask'} for row in rows]
        # Online progress drives the next frame's ready∪current∪next part query.
        clean_objects = [row for row in objects if not row.get('ambiguous_classes')]
        inference = infer_step_from_detections(
            [row['class'] for row in clean_objects],
            rules,
            completed=completed,
            objects=clean_objects,
        )
        completed, memory, inference = advance_progress_memory(
            completed=completed,
            rules=rules,
            inference=inference,
            memory=memory,
            timestamp_s=timestamp,
        )
        record = {
            'index': index,
            'timestamp_s': timestamp,
            'latency_ms': round(1000 * (time.perf_counter() - tick)),
            'objects': objects,
            'detect_scope': scope_meta,
            'inferred_step': inference.get('step'),
            'inferred_status': inference.get('status'),
            'completed_steps': sorted(completed),
            'ready_steps': inference.get('ready_steps', []),
        }
        records.append(record)
        previous = rows
        write_json(args.output / 'detections.json', {
            'model': ('SAM3.1' if '3.1' in str(args.checkpoint).replace('\\', '/') else 'SAM3'),
            'checkpoint': str(args.checkpoint), 'source_video': str(args.video),
            'source_fps': source_fps, 'inference_fps': args.fps, 'start_s': args.start, 'end_s': end,
            'ground_truth_used': False, 'runtime_clicks': 0,
            'scoped_detect': scoped,
            'always_search_parts': list(getattr(args, 'always_search_parts', ()) or ()),
            'min_det_score': float(getattr(args, 'min_det_score', 0.0) or 0.0),
            'records': records,
            'finalized': index == samples - 1, 'elapsed_s': time.perf_counter() - started})
        print(json.dumps({
            'sample': index, 'of': samples, 'time': timestamp,
            'objects': len(rows), 'latency_ms': record['latency_ms'],
            'scoped_classes': len(query_classes),
            'step': inference.get('step'),
            'ready': inference.get('ready_steps', []),
        }), flush=True)
    cap.release()


def observe(args):
    """Prefer deterministic SOP object→step rules; optionally enrich with VLM prose."""
    from monitor.step_rules import reason_over_detection_records

    sop = json.loads(args.sop.read_text(encoding='utf-8'))
    detections = json.loads((args.output / 'detections.json').read_text())
    bootstrap = []
    if getattr(args, 'bootstrap_prerequisites', True) and detections.get('start_s', 0) >= 30:
        # Late offline clips usually begin after earlier stations work; treat the
        # first confident step's prerequisites as already done unless disabled.
        from monitor.step_rules import build_step_rules, infer_step_from_detections, close_prerequisites
        rules = build_step_rules(sop)
        for record in detections['records']:
            objects = [row for row in record.get('objects') or []
                       if not row.get('ambiguous_classes')]
            classes = [row['class'] for row in objects]
            guess = infer_step_from_detections(
                classes, rules, completed=set(), objects=objects)
            # Never bootstrap from a mistake claim (e.g. early IR21 bracket).
            if (guess.get('status') == 'step_in_progress'
                    and guess.get('step') not in {None, 'unknown'}
                    and guess.get('score', 0) >= 6
                    and any(row.get('moving_hits') for row in guess.get('scores') or []
                            if row.get('step_id') == guess['step'])):
                bootstrap = list(close_prerequisites(rules, set(), guess['step']))
                break
    observations = reason_over_detection_records(
        detections['records'], sop, initial_completed=bootstrap)
    # Keep a compact observation stream at the action interval for the footer,
    # plus per-frame rows under reasoning.json for audits.
    write_json(args.output / 'reasoning.json', {
        'model': 'sop_step_rules',
        'ground_truth_used': False,
        'completion_verified': False,
        'bootstrap_prerequisites': bootstrap,
        'observations': observations,
    })
    sampled = []
    start, end = detections['start_s'], detections['end_s']
    for timestamp in np.arange(start, end, args.action_interval):
        row = latest_at(observations, float(timestamp))
        if row:
            sampled.append(dict(row))
    if not sampled and observations:
        sampled = [dict(observations[0])]
    write_json(args.output / 'observations.json', {
        'model': 'sop_step_rules',
        'ground_truth_used': False,
        'completion_verified': False,
        'bootstrap_prerequisites': bootstrap,
        'observations': sampled if sampled else observations,
    })
    for row in (sampled if sampled else observations):
        print(json.dumps({key: row[key] for key in (
            'timestamp_s', 'action', 'step', 'status', 'certainty', 'evidence',
            'recovery', 'missing_prerequisites') if key in row}), flush=True)


def color(identity, class_name: str | None = None):
    """Stable overlay color. Prefer class name so the same part never blinks."""
    if class_name:
        from monitor.kit_catalog import kit_class_color
        return kit_class_color(str(class_name))
    # Legacy numeric fallback (avoid for new draws).
    return ((60 + int(identity) * 73) % 200 + 40,
            (40 + int(identity) * 43) % 200 + 40,
            (90 + int(identity) * 97) % 200 + 40)


from monitor.overlay_hud import (
    draw_sop_step_rail as _draw_sop_step_rail,
    fit_text as _fit_text,
    short_step_title as _short_step_title,
    step_labels_from_sop as _step_labels_from_sop,
    step_order_from_sop as _step_order_from_sop,
)


def _wrap_text(text, width=92, maximum_lines=2):
    words = " ".join(str(text or "").split()).split(" ")
    lines, current = [], ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= maximum_lines:
            break
    if current and len(lines) < maximum_lines:
        lines.append(current)
    if not lines:
        return [""]
    if len(" ".join(words)) > sum(len(row) for row in lines):
        lines[-1] = _fit_text(lines[-1], width)
    return lines[:maximum_lines]


def _step_labels(sop_path):
    sop = json.loads(Path(sop_path).read_text(encoding="utf-8"))
    return _step_labels_from_sop(sop)


def _station_detectables(plan_path):
    if not Path(plan_path).is_file():
        return []
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    names = []
    for row in plan.get("classes") or []:
        name = str(row.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _step_order(sop_path):
    sop = json.loads(Path(sop_path).read_text(encoding="utf-8"))
    return _step_order_from_sop(sop)


def _next_after_ready(sop_path, ready_ids, completed_ids):
    """Steps that open once any current ready step finishes (one-step ahead)."""
    from monitor.sop_detect_scope import successor_step_ids

    sop = json.loads(Path(sop_path).read_text(encoding="utf-8"))
    ready = {str(value) for value in ready_ids}
    done = {str(value) for value in completed_ids}
    if not ready:
        return []
    return [
        step_id for step_id in successor_step_ids(sop, ready, completed=done)
        if step_id not in ready and step_id not in done
    ]


def _box_center_xy(box) -> tuple[int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _draw_dashed_line(frame, p1, p2, color, thickness=2, gap=10):
    x1, y1 = p1
    x2, y2 = p2
    length = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    drawn = 0.0
    while drawn < length:
        start = drawn
        end = min(length, drawn + gap)
        cv2.line(
            frame,
            (int(x1 + dx * start), int(y1 + dy * start)),
            (int(x1 + dx * end), int(y1 + dy * end)),
            color, thickness, cv2.LINE_AA)
        drawn += gap * 2


def _relation_pairs_for_overlay(objects, rules, step_ids):
    """Return drawable moving↔target pairs for the active/ready steps."""
    from monitor.step_rules import (
        _as_box,
        _boxes_attached,
        _boxes_for_queries,
    )

    rows = [row for row in (objects or []) if not row.get("ambiguous_classes")]
    by_id = {rule["step_id"]: rule for rule in rules}
    pairs = []
    for step_id in step_ids:
        rule = by_id.get(step_id)
        if not rule:
            continue
        moving_boxes = _boxes_for_queries(rows, list(rule.get("moving_queries") or []))
        target_boxes = _boxes_for_queries(rows, list(rule.get("target_queries") or []))
        fastener_boxes = _boxes_for_queries(rows, list(rule.get("fastener_queries") or []))
        if not target_boxes:
            continue
        # Prefer a true attach pair; otherwise nearest moving/target centers.
        best = None
        for moving in moving_boxes + fastener_boxes:
            for target in target_boxes:
                attached = _boxes_attached(moving, target)
                mc = _box_center_xy(moving)
                tc = _box_center_xy(target)
                if mc is None or tc is None:
                    continue
                dist = (mc[0] - tc[0]) ** 2 + (mc[1] - tc[1]) ** 2
                candidate = {
                    "step_id": step_id,
                    "moving": moving,
                    "target": target,
                    "attached": attached,
                    "dist": dist,
                    "from_fastener": moving in fastener_boxes and moving not in moving_boxes,
                }
                if best is None:
                    best = candidate
                    continue
                if attached and not best["attached"]:
                    best = candidate
                elif attached == best["attached"] and dist < best["dist"]:
                    best = candidate
        if best is not None:
            # Label the moving class if we can name it.
            label = "fastener" if best["from_fastener"] else "part"
            for row in rows:
                box = _as_box(row.get("box"))
                if box == best["moving"]:
                    label = str(row.get("class") or label)
                    break
            best["label"] = label
            pairs.append(best)
    return pairs


def _draw_relation_edges(frame, edges, width: int):
    """Draw RelateAnything-style subject -predicate-> object edges on the frame."""
    if not edges:
        return False
    any_strong = False
    for edge in edges[:8]:
        if edge.get("draw") is False:
            continue
        subject = edge.get("subject_box")
        obj = edge.get("object_box")
        mc = _box_center_xy(subject)
        tc = _box_center_xy(obj)
        if mc is None or tc is None:
            continue
        score = float(edge.get("score") or 0)
        held = bool(edge.get("held"))
        kind = str(edge.get("kind") or "install")
        strong = score >= (0.45 if kind == "hand" else 0.55)
        any_strong = any_strong or strong
        if kind == "hand":
            # Amber for hand↔part (step start); green/cyan for install.
            color = (40, 170, 255) if strong else (80, 140, 220)
            thickness = 3 if strong else 2
        elif strong:
            color = (40, 220, 120) if not held else (40, 200, 220)
            thickness = 4
        else:
            color = (60, 180, 255)
            thickness = 2
        # Highlight both endpoint boxes so edges stay obvious when centers coincide.
        for box in (subject, obj):
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        # Fan the arrow out if subject sits inside the target (common for pins).
        if ((mc[0] - tc[0]) ** 2 + (mc[1] - tc[1]) ** 2) ** 0.5 < 28:
            tc = (min(width - 8, tc[0] + 55), max(100, tc[1] - 40))
        cv2.line(frame, mc, tc, (10, 15, 20), thickness + 3, cv2.LINE_AA)
        if strong and not held:
            cv2.line(frame, mc, tc, color, thickness, cv2.LINE_AA)
        else:
            _draw_dashed_line(frame, mc, tc, color, thickness=thickness, gap=9)
        ang = math.atan2(tc[1] - mc[1], tc[0] - mc[0])
        ah = 16
        p1 = (
            int(tc[0] - ah * math.cos(ang - 0.45)),
            int(tc[1] - ah * math.sin(ang - 0.45)),
        )
        p2 = (
            int(tc[0] - ah * math.cos(ang + 0.45)),
            int(tc[1] - ah * math.sin(ang + 0.45)),
        )
        cv2.fillConvexPoly(frame, np.array([tc, p1, p2], dtype=np.int32), color)
        cv2.circle(frame, mc, 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, tc, 8, (240, 240, 240), -1, cv2.LINE_AA)
        cv2.circle(frame, tc, 8, color, 2, cv2.LINE_AA)
        predicate = str(edge.get("predicate") or "near")
        subj = str(edge.get("subject") or "part")
        obj_name = str(edge.get("object") or "target")
        step_id = str(edge.get("step_id") or "")
        mark = "HOLD" if held else f"{score:.2f}"
        tag = f"{step_id}  {subj} -{predicate}-> {obj_name}  [{mark}]"
        text_w = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
        mid = ((mc[0] + tc[0]) // 2, (mc[1] + tc[1]) // 2)
        x = max(8, min(width - text_w - 12, mid[0] - text_w // 2))
        y = max(110, mid[1] - 12)
        cv2.rectangle(frame, (x - 4, y - 20), (x + text_w + 8, y + 8), (12, 16, 22), -1)
        cv2.rectangle(frame, (x - 4, y - 20), (x + text_w + 8, y + 8), color, 2)
        cv2.putText(frame, tag, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
    return any_strong


def render(args):
    import imageio_ffmpeg
    from monitor.step_rules import build_step_rules

    detection = json.loads((args.output / 'detections.json').read_text())
    if not detection['finalized']:
        raise ValueError('Detection pass is incomplete.')
    reasoning_path = args.output / 'reasoning.json'
    if reasoning_path.is_file():
        observations = json.loads(reasoning_path.read_text(encoding='utf-8'))['observations']
    else:
        observations = json.loads((args.output / 'observations.json').read_text())['observations']
    sop = json.loads(Path(args.sop).read_text(encoding='utf-8'))
    rules = build_step_rules(sop)
    step_labels = _step_labels(args.sop)
    step_order = _step_order(args.sop)
    detectables = _station_detectables(args.output / 'plan.json')
    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(3)), int(cap.get(4))
    footer = 230
    playback_fps = float(getattr(args, 'playback_fps', 0) or detection.get('source_fps') or 10)
    if playback_fps <= 0:
        raise ValueError('playback_fps must be positive.')
    detect_fps = float(detection['inference_fps'])
    start_s = float(detection['start_s'])
    end_s = float(detection['end_s'])
    # Optional clip window from CLI (--start / --duration), clamped to detections.
    clip_start = float(getattr(args, 'start', start_s) or start_s)
    clip_duration = float(getattr(args, 'duration', 0) or 0)
    if clip_start > start_s:
        start_s = min(clip_start, end_s)
    if 0 < clip_duration < 100000:
        end_s = min(end_s, start_s + clip_duration)
    destination = args.output / 'assembly_overlays.mp4'
    temp_destination = args.output / 'assembly_overlays.rendering.mp4'
    if temp_destination.exists():
        temp_destination.unlink()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{width}x{height+footer}', '-r', str(playback_fps),
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp_destination)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.perf_counter()
    frame_count = 0
    try:
        total = max(1, int(math.floor((end_s - start_s) * playback_fps)))
        for frame_index in range(total):
            timestamp = start_s + frame_index / playback_fps
            if timestamp >= end_s - 1e-9:
                timestamp = max(start_s, end_s - 1e-3)
            record = latest_at(detection['records'], timestamp)
            if record is None:
                continue
            frame = read_frame(cap, timestamp)
            mask_path = args.output / 'masks' / f'{record["index"]:05d}.npz'
            try:
                archive = np.load(mask_path)
            except (OSError, ValueError, zipfile.BadZipFile):
                archive = None
            try:
                files = set(archive.files) if archive is not None else set()
                drawable = [obj for obj in record['objects'] if str(obj['id']) in files]
                for obj in sorted(
                    drawable,
                    key=lambda r: np.count_nonzero(archive[str(r['id'])]),
                    reverse=True,
                ):
                    mask = archive[str(obj['id'])]
                    if mask.shape[:2] != frame.shape[:2]:
                        continue
                    tint = color(obj['id'], obj.get('class'))
                    frame[mask] = (frame[mask]*.68 + np.array(tint)*.32).astype(np.uint8)
                    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(frame, contours, -1, tint, 1)
                    x1,y1,x2,y2 = map(int,obj['box'])
                    label = ('ambiguous part' if obj.get('ambiguous_classes') else obj['class'])
                    label += f" {obj['score']:.2f}"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),tint,2)
                    y = max(17,y1-6)
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,.44,1)[0][0]
                    cv2.rectangle(frame,(x1,max(0,y-17)),(min(width,x1+text_width+6),y+3),(20,25,30),-1)
                    cv2.putText(frame,label,(x1+2,y),cv2.FONT_HERSHEY_SIMPLEX,.44,tint,1,cv2.LINE_AA)
                drawn_ids = {str(obj['id']) for obj in drawable}
                for obj in record['objects']:
                    if str(obj['id']) in drawn_ids:
                        continue
                    box = obj.get('box')
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    tint = color(obj['id'], obj.get('class'))
                    x1,y1,x2,y2 = map(int, box)
                    label = ('ambiguous part' if obj.get('ambiguous_classes') else obj['class'])
                    label += f" {float(obj.get('score') or 0):.2f}"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),tint,2)
                    y = max(17,y1-6)
                    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,.44,1)[0][0]
                    cv2.rectangle(frame,(x1,max(0,y-17)),(min(width,x1+text_width+6),y+3),(20,25,30),-1)
                    cv2.putText(frame,label,(x1+2,y),cv2.FONT_HERSHEY_SIMPLEX,.44,tint,1,cv2.LINE_AA)
            finally:
                if archive is not None:
                    archive.close()

            current = latest_at(observations, timestamp) or {}
            status = current.get('status', 'unknown')
            # Worker CURRENT: sticky help step (hand-holding), else live reasoner step.
            step = current.get('current_step') or current.get('step', 'unknown')
            help_active = bool(current.get('help_active') or (
                step not in {None, 'unknown', ''} and status == 'step_in_progress'))
            action = current.get('action', 'unknown')
            recovery = str(current.get('recovery') or '')
            evidence = str(current.get('evidence') or 'Waiting for step reasoning')
            instruction = str(current.get('instruction') or step_labels.get(step, '') or '')
            if not instruction and step not in {None, 'unknown', ''}:
                instruction = step_labels.get(step, '')
            completed = [sid for sid in step_order if sid in set(current.get('completed_steps') or [])]
            ready = [sid for sid in step_order if sid in set(current.get('ready_steps') or [])]
            # Prefer observation ready list order already; fall back keeps SOP order above.
            if current.get('ready_steps'):
                ready = list(current.get('ready_steps') or [])
            upcoming = _next_after_ready(args.sop, ready, completed)

            # Relation edges: CURRENT step only (hand↔part + its install), not all ready.
            relation_edges = list(current.get("relations") or [])
            current_sid = step if step not in {None, "unknown", ""} else None
            if relation_edges and current_sid:
                relation_edges = [
                    edge for edge in relation_edges
                    if edge.get("step_id") == current_sid
                ]
            if not relation_edges and current_sid:
                live_pairs = _relation_pairs_for_overlay(
                    (record or {}).get("objects") or [], rules, [current_sid])
                relation_edges = [
                    {
                        "step_id": pair["step_id"],
                        "kind": "install",
                        "predicate": "attached to" if pair.get("attached") else "near",
                        "score": 0.8 if pair.get("attached") else 0.4,
                        "subject": pair.get("label") or "part",
                        "object": "partial model",
                        "subject_box": pair["moving"],
                        "object_box": pair["target"],
                        "held": False,
                    }
                    for pair in live_pairs
                ]
            from monitor.sop_relations import refresh_relation_edges_for_frame
            relation_edges = refresh_relation_edges_for_frame(
                relation_edges, (record or {}).get("objects") or [])
            # Prefer hand edges for CURRENT; keep at most one install edge.
            hand_edges = [e for e in relation_edges if e.get("kind") == "hand"]
            install_edges = [e for e in relation_edges if e.get("kind") != "hand"]
            relation_edges = (hand_edges[:2] + install_edges[:1])
            attach_now = _draw_relation_edges(frame, relation_edges, width)

            # Left SOP checklist: tick completed, color current execution.
            current_for_rail = step if (
                help_active and step not in {None, 'unknown', ''}
            ) else (step if status == 'step_in_progress' and step not in {None, 'unknown', ''} else '')
            rail_w = _draw_sop_step_rail(
                frame, step_order, step_labels, completed, current_for_rail, ready=ready,
            )

            # Top banners (right of step rail): CURRENT help, then DONE summary.
            banner_x = rail_w
            banner_w = max(1, width - banner_x)
            cv2.rectangle(frame, (banner_x, 0), (width, 40), (20, 25, 30), -1)
            if status.startswith('mistake'):
                tone = (80, 80, 255)
                banner = f"MISTAKE: {_short_step_title(step, instruction)}  |  {action}"
            elif help_active and step not in {None, 'unknown', ''}:
                tone = (120, 230, 160)
                banner = f"CURRENT (help): {_short_step_title(step, instruction)}  |  {action}"
            else:
                tone = (150, 160, 170)
                banner = "CURRENT (help): (waiting for hand+part)"
            cv2.rectangle(frame, (banner_x, 0), (banner_x + 8, 40), tone, -1)
            max_banner = max(40, banner_w // 8)
            cv2.putText(frame, _fit_text(banner, max_banner), (banner_x + 16, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, tone, 2, cv2.LINE_AA)

            cv2.rectangle(frame, (banner_x, 40), (width, 80), (25, 30, 38), -1)
            done_text = "DONE: " + (", ".join(completed) if completed else "(none yet)")
            cv2.putText(frame, _fit_text(done_text, max_banner), (banner_x + 16, 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 2, cv2.LINE_AA)

            canvas = np.zeros((height + footer, width, 3), np.uint8)
            canvas[:height] = frame
            seen = []
            for obj in (record or {}).get('objects') or []:
                name = 'ambiguous' if obj.get('ambiguous_classes') else obj.get('class')
                if name and name not in seen:
                    seen.append(str(name))
            relation_line = str(current.get("relations_summary") or "")
            if not relation_line:
                from monitor.sop_relations import summarize_relations
                relation_line = summarize_relations(relation_edges)
            lines = [
                f"t={timestamp:.1f}s   detect {detect_fps:g} FPS   playback {playback_fps:g} FPS   top-down SOP monitor",
                _fit_text("DONE: " + (", ".join(completed) if completed else "(none)"), 115),
                _fit_text("READY now: " + (", ".join(ready) if ready else "(none)"), 115),
                _fit_text(
                    "NEXT (after ready): " + (", ".join(upcoming) if upcoming else "(none)"),
                    115,
                ),
                _fit_text(
                    "CURRENT detail: " + (
                        f"{step} — {instruction}" if step not in {None, 'unknown', ''} and instruction
                        else "(no active SOP step)"
                    ),
                    115,
                ),
                _fit_text(relation_line, 115),
                _fit_text("Seen now: " + (", ".join(seen) if seen else "(nothing)"), 115),
            ]
            colors = [
                (200, 210, 220),
                (180, 200, 220),
                (120, 230, 160),
                (120, 200, 255),
                tone,
                (40, 220, 120) if attach_now else (60, 180, 255),
                (180, 220, 255),
            ]
            for index, line in enumerate(lines):
                cv2.putText(canvas, line, (12, height + 24 + index * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors[index], 1, cv2.LINE_AA)
            if recovery:
                cv2.putText(canvas, _fit_text(f"Recovery: {recovery}", 115),
                            (12, height + footer - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (80, 80, 255), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
            frame_count += 1
            if frame_index % max(1, int(playback_fps * 30)) == 0:
                cv2.imwrite(str(args.output / f'preview_{int(timestamp):04d}.jpg'), canvas)
    finally:
        cap.release()
        process.stdin.close()
    if process.wait() != 0:
        if temp_destination.exists():
            temp_destination.unlink()
        raise RuntimeError('Video encoder failed')
    if destination.exists():
        destination.unlink()
    temp_destination.replace(destination)
    write_json(args.output / 'result.json', {
        'video': str(destination), 'frames': frame_count,
        'fps': playback_fps, 'detect_fps': detect_fps,
        'runtime_clicks': 0, 'ground_truth_used': False,
        'accuracy_measured': False, 'video_sha256': digest(args.video),
        'detection_seconds': detection['elapsed_s'],
        'action_seconds': 0,
        'render_seconds': time.perf_counter()-started,
        'step_reasoner': 'sop_step_rules_topdown',
        'station_detectables': detectables,
        'overlay_style': 'left_sop_checklist_done_ready_next_relations',
        'limitations': ['Playback can be 10 FPS while detection stays sparse',
                        'Masks are held between detect samples, not re-segmented every frame',
                        'Relation lines use the same box geometry as the DONE attach gate',
                        'Step candidates are rule-based from detected SOP objects, not verified completions']})
    print(destination, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['plan','detect','observe','render'])
    parser.add_argument('--video',type=Path,default=ROOT/'data/generated/industreal/videos/03_assy_0_1.mp4')
    parser.add_argument('--sop',type=Path,default=ROOT/'data/industreal_assembly.template.json')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/sam3/sam3.safetensors')
    parser.add_argument('--output',type=Path,default=ROOT/'output/automatic_assembly')
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--duration',type=float,default=100000)
    parser.add_argument('--fps',type=float,default=2)
    parser.add_argument('--action-interval',type=float,default=5)
    parser.add_argument('--playback-fps', type=float, default=10,
                        help='Output video FPS for render. Detection can stay sparse; frames hold overlays between samples.')
    parser.add_argument('--bootstrap-prerequisites', dest='bootstrap_prerequisites',
                        action='store_true', default=True,
                        help='For late clips, mark the first confident step prerequisites complete.')
    parser.add_argument('--no-bootstrap-prerequisites', dest='bootstrap_prerequisites',
                        action='store_false',
                        help='Require full prerequisite history; useful to demo out-of-order recovery.')
    parser.add_argument('--scoped-detect', dest='scoped_detect', action='store_true',
                        default=True,
                        help='Query SAM3 only for parts of ready∪current SOP steps (+hand/paper).')
    parser.add_argument('--no-scoped-detect', dest='scoped_detect', action='store_false',
                        help='Query the full plan vocabulary every frame.')
    parser.add_argument(
        '--always-search-parts', nargs='*', default=[],
        help='Extra SOP part names to query every frame (improves small-part recall).')
    parser.add_argument('--min-det-score', type=float, default=0.0,
                        help='Drop SAM boxes below this confidence (0 keeps model default).')
    parser.add_argument('--min-mask-area', type=float, default=0.00004,
                        help='Drop masks covering less than this fraction of the frame.')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an incomplete detections.json from the next sample.')
    args=parser.parse_args()
    if not 0 < args.fps <= 10 or args.start < 0 or args.duration <= 0 or args.action_interval <= 0:
        parser.error('Invalid timing parameters')
    args.output=args.output.resolve()
    if ROOT.resolve() not in args.output.parents:
        parser.error('Output must stay inside this project')
    args.output.mkdir(parents=True,exist_ok=True)
    {'plan':make_plan,'detect':detect,'observe':observe,'render':render}[args.stage](args)


if __name__=='__main__':
    main()
