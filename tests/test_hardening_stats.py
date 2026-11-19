"""The statistics behind eval_P1_T66.md section 10: bootstrap by scene, sign-flip test, scene grouping, wording."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hardening_study as H  # noqa: E402


# ------------------------------------------------------------------------------------------- bootstrap
def test_interval_brackets_the_mean_and_is_reproducible():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 1.5])
    lo, hi = H.bootstrap_ci(d)
    assert lo < d.mean() < hi and H.bootstrap_ci(d) == (lo, hi)           # fixed seed -> identical


def test_a_constant_difference_has_a_zero_width_interval():
    lo, hi = H.bootstrap_ci(np.full(12, 0.7))
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_resampling_by_scene_is_wider_than_by_image_when_scenes_repeat():
    """40 images in 8 scenes of 5 near-identical shots each: by-image intervals pretend there are 40 independent samples."""
    rng = np.random.default_rng(1)
    scene_effect = rng.normal(0, 1, 8)
    d = np.repeat(scene_effect, 5) + rng.normal(0, 0.01, 40)
    groups = np.repeat(np.arange(8), 5)
    by_scene = H.bootstrap_ci(d, groups)
    by_image = H.bootstrap_ci(d)
    assert (by_scene[1] - by_scene[0]) > 1.5 * (by_image[1] - by_image[0])


def test_scene_resampling_of_singletons_matches_image_resampling():
    d = np.random.default_rng(2).normal(0.3, 1, 30)
    assert H.bootstrap_ci(d, np.arange(30)) == H.bootstrap_ci(d, None)


# ------------------------------------------------------------------------------------------- sign-flip test
def test_exact_sign_flip_p_for_all_positive_differences():
    assert H.sign_flip_p(np.arange(1, 11, dtype=float)) == pytest.approx(2 / 2 ** 10)


def test_sign_flip_p_is_one_for_no_difference_and_large_for_symmetric_noise():
    assert H.sign_flip_p(np.zeros(9)) == 1.0
    assert H.sign_flip_p(np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])) == 1.0


def test_sign_flip_uses_scenes_not_images():
    """Ten images of ONE scene are one unit: p cannot be smaller than 1 (two sign patterns, both as extreme)."""
    d = np.full(10, 0.5)
    assert H.sign_flip_p(d, np.zeros(10, int)) == 1.0
    assert H.sign_flip_p(d) < 0.01                                        # treated as 10 independent images it looks significant


def test_monte_carlo_branch_for_many_scenes_agrees_with_intuition():
    d = np.full(40, 1.0)
    assert H.sign_flip_p(d) < 0.001
    noise = np.random.default_rng(3).normal(0, 1, 40)
    assert H.sign_flip_p(noise) > 0.05


# ------------------------------------------------------------------------------------------- comparison and wording
def test_compare_counts_wins_ties_and_losses_with_the_right_direction():
    a = np.array([30.0, 30.0, 30.0, 30.0])
    b = np.array([29.0, 30.0, 31.0, 29.5])
    psnr = H.compare(a, b, "psnr", None)
    assert (psnr["a_better"], psnr["ties"], psnr["a_worse"]) == (2, 1, 1)
    lp = H.compare(a / 100, b / 100, "lpips", None)                       # lower LPIPS is better: the roles swap
    assert (lp["a_better"], lp["ties"], lp["a_worse"]) == (1, 1, 2)
    assert psnr["mean_diff"] == pytest.approx(0.125)


def test_verdict_words_follow_the_interval_and_the_metric_direction():
    ci = {"excludes_zero": True, "mean_diff": 0.2, "metric": "psnr"}
    assert H.verdict(ci) == "better"
    assert H.verdict({**ci, "metric": "lpips"}) == "worse"                # a higher LPIPS is worse
    assert H.verdict({**ci, "excludes_zero": False}) == "not distinguishable"
    assert H.verdict({**ci, "mean_diff": -0.2}) == "worse"


def test_p_values_are_shown_without_false_precision():
    assert H.pvalue(0.00002) == "< 0.001" and H.pvalue(0.0352) == "0.035"


# ------------------------------------------------------------------------------------------- scene grouping
def test_near_duplicates_form_one_scene_and_distinct_images_do_not():
    rng = np.random.default_rng(4)
    base = rng.random((3, 64))
    fps = np.stack([base[0], base[0] + 0.004, base[0] - 0.003, base[1], base[1] + 0.002, base[2]])
    g = H.scene_groups(fps, tol=0.01)
    assert g[0] == g[1] == g[2] and g[3] == g[4] and len({g[0], g[3], g[5]}) == 3


def test_scene_chain_is_connected_transitively():
    fps = np.array([[0.0], [0.008], [0.016], [0.5]])
    g = H.scene_groups(fps, tol=0.01)
    assert g[0] == g[1] == g[2] != g[3]


# ------------------------------------------------------------------------------------------- the generated report
SUMMARY = ROOT / "results" / "phase_f_hardening" / "summary.json"


@pytest.mark.skipif(not SUMMARY.is_file(), reason="run scripts/hardening_study.py first")
def test_eval_report_section_is_exactly_what_the_script_generates_from_the_results():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    text = (ROOT / "eval_P1_T66.md").read_text(encoding="utf-8")
    assert H.START in text and H.END in text
    section = text.split(H.START, 1)[1].split(H.END, 1)[0].strip()
    assert section == H.build_section(s).strip(), "eval_P1_T66.md section 10 is stale: run scripts/hardening_study.py report"


@pytest.mark.skipif(not SUMMARY.is_file(), reason="run scripts/hardening_study.py first")
def test_the_frozen_headline_rows_were_reproduced_exactly_and_the_headline_table_is_untouched():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert max(s["eval15"]["frozen_check_max_abs_diff"].values()) < 1e-6
    text = (ROOT / "eval_P1_T66.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "1RUA24CSE0093,1RUA24CSE0096,1RUA24CSE0258,1RUA24CSE0259"    # USNs stay on line 1
    for row in ("| Classical (tuned) | 17.16 | 0.7241 | 0.2781 | 5.50 |", "| **Deep: Retinexformer** | **25.15** | **0.8441** | **0.1314** | **3.48** |",
                "| **Quality-Guided Hybrid: Retinexformer** | 25.12 |"):
        assert row in text


@pytest.mark.skipif(not SUMMARY.is_file(), reason="run scripts/hardening_study.py first")
def test_summary_numbers_are_consistent_with_the_per_image_files():
    """Spot check: the eval15 mean PSNR in the summary equals the mean of its per-image CSV."""
    import csv

    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(ROOT / "results/phase_f_hardening/per_image_eval15.csv", encoding="utf-8")) if r["method"] == "retinexformer"]
    assert len(rows) == s["eval15"]["n"]
    assert np.mean([float(r["psnr"]) for r in rows]) == pytest.approx(s["eval15"]["means"]["retinexformer"]["psnr"])


@pytest.mark.skipif(not SUMMARY.is_file(), reason="run scripts/hardening_study.py first")
def test_evidence_page_trust_table_is_generated_from_the_study_and_the_page_can_show_it():
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "app/web/static/data/evaluation.json").read_text(encoding="utf-8"))
    assert ev["robustness"] == H.robustness_block(s), "evaluation.json is stale: run scripts/hardening_study.py report"
    assert ev["robustness"]["rows"] and {r["reading"] for r in ev["robustness"]["rows"]} <= {"better", "worse", "not distinguishable"}
    assert [m["name"] for m in ev["methods"]][:1] == ["Quick (classical, tuned)"]              # the frozen table is untouched
    html = (ROOT / "app/web/evidence.html").read_text(encoding="utf-8")       # moved from the details page in the redesign
    js = (ROOT / "app/web/static/js/evidence.js").read_text(encoding="utf-8")
    for element in ("trust", "trustIntro", "trustTable", "trustCaveat"):
        assert f'id="{element}"' in html and f'$("{element}")' in js


def test_the_evaluation_data_file_is_still_valid_json_after_the_block_is_replaced(tmp_path, monkeypatch):
    f = tmp_path / "evaluation.json"
    f.write_text('{\n  "a": 1,\n  "b": [1, 2]\n}\n', encoding="utf-8")
    monkeypatch.setattr(H, "EVAL_JSON", f)
    H.write_evaluation_json({"rows": [1]})
    H.write_evaluation_json({"rows": [2, 3]})                    # replaced, not appended twice
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data == {"a": 1, "b": [1, 2], "robustness": {"rows": [2, 3]}}
