# Third-party code

`third_party/industreal_psr/psr_utils.py` and `third_party/industreal_psr/procedure_info.json` are copied
**unmodified** from the official IndustReal repository (https://github.com/TimSchoonbeek/IndustReal, folder `PSR/`)
and are licensed under the Apache License 2.0 (`third_party/industreal_psr/LICENSE`).
They provide the official PSR evaluation metrics (POS, F1, average delay) and the procedure definition.

`psr/official_eval.py` imports `psr_utils` as-is; it only stubs the `bounding_box` drawing package, which the
metric functions do not use.

`third_party/asr_error_labels/error_labels.json` is copied **unmodified** from
https://github.com/TimSchoonbeek/AssemblyStateRecognition (Schoonbeek et al., RA-L 2024) and is licensed under the
Apache License 2.0 (`third_party/asr_error_labels/LICENSE`). It provides the frame-level error-state labels used to
train and evaluate the mistake detector.

The DINOv2 backbones (`facebook/dinov2-small`, `facebook/dinov2-base`) are downloaded from Hugging Face at run time
and are licensed by their authors (Apache 2.0).

The IndustReal dataset is not included; obtain it from the authors under its own terms.

# Not included

The live action-recognition predictions (`ar.jsonl`) and the `assist` package (task graph and mistake rules, used by
`psr/v11.py` through `TEAMMATE_AR`) are a teammate's work and are not redistributed here.

# Assembly state detector (v14)

v14 uses the official IndustReal ASD model `asd_best_IndustRealandSynthetic.pt` (YOLOv8, Schoonbeek et al.), from
`assembly_state_detection_model_weights.zip` on the IndustReal dataset page (data.4tu.nl, Apache-2.0). The weights are
not included. `cache/asd_val/` and `cache/asd_test/` hold this model's outputs on val and test, computed with
`tools/run_asd.py` (ultralytics 8.3.40).
