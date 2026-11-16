# Our changes to the teammate's code

Base: GitHub jennifer-nathania/assembly-task-checking, branch `verb-noun-tcn`, commit 8ce1e81 (2026-11-09).
Edits to their files (full diff in `OUR_CHANGES.diff`):
- `assist/run_engine.py`: state source `asdv14` -> `asd_v14.jsonl` (v14's steps as the engine's state, as their `asdv9`).
- `assist/dashboard/data.py`, `assist/dashboard/static/common.js`: dashboard run list / default run include the v14 runs; the
  Assembly101 table is optional; v14's own step scores on the results page.
New files (no other file of theirs changed):
- `live/`: the on-demand pipeline (`run_video.py`: their causal TCN pipeline, their online V-JEPA 2 label, their action-end
  label, frozen v14 live, their hands + engine), the job service `api.py`, `start_jobs.sh`, `progress.py`.
- `assist/tools/cards_web.py`: the card overlay in the browser, drawn by their overlay_cards / cad_state / end_label_view
  functions (pixel-identical to their MP4 renderer, checked on full frames).
- `scripts/mistakes/compare_v14.py`: v14's saved steps / alerts inside their engine and scorers.
- `meccano_action_end/` + `output/meccano_action_end/` (2026-11-14): the teammate's past-only MECCANO action-end run, copied
  from the teammate's copy on the lab server with its results, logs, features, TCN outputs and the three V-JEPA 2
  checkpoints (`vjft_area_k12.pt`, `vjft_hand_k12.pt` from their `output/meccano/vjepa`). One edit: `data.py` reads the
  dataset from STARE's `data/MECCANO` by default. Test: `tests/test_meccano_action_end.py`.
- Removed (2026-11-14): `scripts/meccano/` (the earlier offline MECCANO recipe, 62.24 %, which read later frames) and its
  README section; replaced by the above.
- Tidy for the public repository (2026-11-16). Moved out (they feed none of the paper's results; kept outside the
  repository): `scripts/evaluate_release.py`, `scripts/run_release_pipeline.py`, `scripts/efficiency/`, `scripts/zeroshot/`,
  `assist/assembly101/`, `assist/dashboard/{app.py, __main__.py}` (the dashboard's `data.py` and `static/sound.js` stay: the
  demo's spoken alerts use them), `assist/tools/transcode_videos.py`, `scripts/features/extract_finetuned_features.py`,
  `scripts/kit_sop/render_ar_asd_psr_gt_overlay.py`, `scripts/mistakes/{crop_prototypes, dual_qformer, execution_qformer,
  render_train, render_wrong, tar_images, val_baseline, warn_vlm}.py`, `scripts/recognition/{boost_accuracy, compare_finals,
  cv_trainval, distill_base, early_eval, online_eval, procedure_recognition, render_online_overlay, score_extra_final}.py`,
  `scripts/segmentation/{fuse_videomae_segmentation, relabel_segments}.py`, their four tests, `TEAM_GPU_RUN.md` and the
  planning notes under `docs/`. Edited: local or server paths replaced by repository-relative defaults in
  `data/sops/13_assy_0_1.sop.json`, `scripts/mistakes/state_match.py`, `scripts/mistakes/v9_checks.py`,
  `live/start_jobs.sh` (comment) and `meccano_action_end/README.md`. New test: `tests/test_meccano_final_labels_score.py`.
