"""The six-cases reel: each clip must really contain its alert, and each card must say where the input came from."""
import unittest

from assist.tools.six_cases import (CAD_CASES, CASES, FINAL_CASES, LIVE, TRAIN_LIVE, V9_CASES, V9_RUN, clip_name, missing_alert,
                                    shown_alert, spoken_alerts)


class ClipTest(unittest.TestCase):
    def test_an_alert_inside_the_clip_counts(self):
        alerts = [{"t": 128.5, "type": "wrong_order"}]
        self.assertFalse(missing_alert(alerts, "wrong_order", 128.5, 118, 136))

    def test_an_alert_of_another_type_or_time_does_not(self):
        alerts = [{"t": 128.5, "type": "skipped_step"}, {"t": 150.0, "type": "wrong_order"}]
        self.assertTrue(missing_alert(alerts, "wrong_order", 128.5, 118, 136))

    def test_a_clip_running_to_the_end_of_the_video_contains_the_last_alert(self):
        self.assertFalse(missing_alert([{"t": 239.8, "type": "skipped_step"}], "skipped_step", 239.8, 226, None))


class SoundTest(unittest.TestCase):
    def test_only_the_shown_alert_speaks_when_two_fire_together(self):
        alerts = [{"t": 239.8, "type": "wrong_order"}, {"t": 239.8, "type": "skipped_step"}, {"t": 90.1, "type": "early_warning"}]
        self.assertEqual([a["type"] for a in spoken_alerts(alerts, "skipped_step", 239.8)], ["skipped_step", "early_warning"])

    def test_the_shown_alert_is_found_when_it_is_the_last_one(self):
        alerts = [{"t": 239.8, "type": "wrong_order"}, {"t": 239.8, "type": "skipped_step"}]
        self.assertEqual(shown_alert(alerts, "skipped_step", 239.8)["type"], "skipped_step")


class CasesTest(unittest.TestCase):
    def test_the_six_cases_come_in_the_order_asked_for(self):
        self.assertEqual([c["alert"] for c in CASES], ["skipped_step", "wrong_order", "incorrect_install",
                                                       "possible_incorrect_install", "wrong_part", "early_warning"])

    def test_every_clip_comes_from_a_live_run_never_from_the_labels(self):
        for case in CASES:
            self.assertIn(case["run"], (LIVE, TRAIN_LIVE), case["title"])
            self.assertIn("live system", " ".join(case["lines"]).lower(), case["title"])

    def test_a_training_recording_is_named_as_one_and_says_what_differs(self):
        for case in CASES:
            text = " ".join(case["lines"]).lower()
            on_training = case["footer_split"] == "train"
            self.assertEqual("training recording" in text, on_training, case["title"])
            self.assertEqual("without v-jepa" in text, on_training, case["title"])
            self.assertEqual("trained on this video" in text, on_training, case["title"])
            self.assertEqual(case["run"] == TRAIN_LIVE, on_training, case["title"])


class V9ClipsTest(unittest.TestCase):
    def test_only_the_cases_this_run_really_raised_are_included(self):
        self.assertEqual([(c["alert"], c["t"]) for c in V9_CASES],
                         [("skipped_step", 239.8), ("wrong_order", 130.0), ("early_warning", 90.1)])

    def test_every_clip_comes_from_the_v9_run_and_says_so(self):
        for case in V9_CASES:
            self.assertEqual(case["run"], V9_RUN, case["title"])
            self.assertIn("v9", " ".join(case["lines"]), case["title"])

    def test_every_alert_falls_inside_its_clip(self):
        for case in V9_CASES:
            self.assertFalse(missing_alert([{"t": case["t"], "type": case["alert"]}], case["alert"], case["t"],
                                           case["start"], case["stop"]), case["title"])

    def test_the_v9_clips_never_overwrite_the_existing_ones(self):
        self.assertEqual(clip_name("six_cases", CASES[0]), "six_cases_1_skipped_step.mp4")
        self.assertEqual(clip_name("six_cases_v9", V9_CASES[0]), "six_cases_v9_1_skipped_step.mp4")


class CadClipsTest(unittest.TestCase):
    def test_all_six_cases_come_in_order(self):
        self.assertEqual([c["title"][0] for c in CAD_CASES], list("123456"))

    def test_a_case_uses_the_v9_run_where_one_exists_and_the_released_detector_otherwise(self):
        self.assertEqual([c["run"] for c in CAD_CASES], [V9_RUN, V9_RUN, TRAIN_LIVE, LIVE, TRAIN_LIVE, V9_RUN])

    def test_the_cad_clips_never_overwrite_the_existing_ones(self):
        self.assertEqual(clip_name("six_cases_cad", CAD_CASES[0]), "six_cases_cad_1_skipped_step.mp4")


class FinalClipsTest(unittest.TestCase):
    def test_all_six_cases_stay_on_the_same_runs_and_moments(self):
        self.assertEqual([(c["run"], c["t"]) for c in FINAL_CASES], [(c["run"], c["t"]) for c in CAD_CASES])

    def test_a_training_clip_says_the_final_labels_models_trained_on_that_video(self):
        for case in FINAL_CASES:
            text = " ".join(case["lines"]).lower()
            self.assertEqual("final label's models" in text, case["footer_split"] == "train", case["title"])

    def test_the_final_clips_never_overwrite_the_existing_ones(self):
        self.assertEqual(clip_name("six_cases_final", FINAL_CASES[0]), "six_cases_final_1_skipped_step.mp4")


if __name__ == "__main__":
    unittest.main()
