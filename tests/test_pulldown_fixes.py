"""六个修复点与完整 pulldown 管线的行为回归。"""

from fractions import Fraction

import media.pulldown_dedup as pd


def test_adaptive_period_detects_3_to_2_cadence():
    flags = [i in {1, 6, 11, 16, 21} for i in range(26)]
    result = pd.adaptive_period(flags)
    assert result == 5
    assert result["phase"] == 1
    assert result["reliable"]


def test_adaptive_period_tracks_vfr_segment():
    result = pd.adaptive_period(pd.make_vfr_case())
    assert result["segments"][0]["period"] == 5
    assert result["segments"][1]["period"] == 4
    assert result["vfr"] is True


def test_tolerant_match_accepts_grain_but_rejects_motion():
    duplicate_a = pd.make_frame(7, occurrence=100, noise=0.0)
    duplicate_b = pd.make_frame(7, occurrence=101, noise=18.0)
    moving_b = pd.make_frame(8, occurrence=102, noise=4.0)

    assert pd.tolerant_match(duplicate_a, duplicate_b) is True
    assert pd.tolerant_match(duplicate_a, moving_b) is False
    report = pd.tolerant_report(duplicate_a, duplicate_b)
    assert report["exact_hash_equal"] is False
    assert report["match"] is True


def test_field_order_uses_bottom_first_metadata():
    order = pd.field_order_meta({"field_order": "Lower Field First"})
    assert order == "bottom"
    assert order == "bb"
    assert order["top_field_first"] is False
    assert pd.field_order_meta({"top_field_first": 1}) == "top"


def test_weave_fields_places_bottom_field_on_odd_rows():
    first = [[10.0, 10.0], [12.0, 12.0]]
    second = [[20.0, 20.0], [22.0, 22.0]]
    woven = pd.weave_fields(first, second, "bottom")
    # bottom-first：第一场 -> 奇数行；第二场 -> 偶数行。
    assert woven[0][0] == pd._clamp(20.0 / 255.0, 0.0, 1.0)
    assert woven[1][0] == pd._clamp(10.0 / 255.0, 0.0, 1.0)


def test_timestamp_uses_exact_23_976_rate_without_accumulation():
    ts = pd.timestamp_actual_rate(14386, Fraction(24000, 1001))
    assert isinstance(ts.rational, Fraction)
    assert ts.rational == Fraction(7200193, 12000)
    assert float(ts) == 600.0160833333333
    nominal_24p = 14386 / 24
    assert abs(nominal_24p - float(ts)) > 0.5


def test_continue_at_splice_continues_local_phase():
    state = {"period": 5, "phase": 1, "segment_start": 0,
             "phase_kind": "drop_residue"}
    result = pd.continue_at_splice(state, {"start": 13, "length": 13})
    assert result["continued"] is True
    assert result["reset"] is False
    assert result["phase"] == 3


def test_dedup_standard_splice_and_vfr_cases():
    for case in (pd.make_standard_case(), pd.make_splice_case(),
                 pd.make_vfr_case()):
        result = pd.dedup(case, safe_mode=False)
        assert result["drop_indices"] == case["drop_indices"]
        assert result["validation"]["ok"]
        metrics = pd.metrics(result)
        assert metrics["false_drop_count"] == 0
        assert metrics["missed_count"] == 0


def test_safety_gate_rejects_wrong_period_motion_candidate():
    case = pd.make_vfr_case()
    result = pd.dedup(case, drop_indices=[26], safe_mode=False)
    assert result["drop_indices"] == []
    assert result["rejected"][0]["reason"] == "failed_tolerant_match"
    assert result["validation"]["ok"]


def test_validate_output_detects_timestamp_drift_and_false_drop():
    drift = {
        "frames": [[[0.0]]] * 5,
        "kept_indices": list(range(5)),
        "dropped": [],
        "rate": Fraction(24000, 1001),
        "timestamps": [i / 24 for i in range(5)],
    }
    assert not pd.validate_output(drift)["ok"]

    case = pd.make_standard_case()
    false_drop = {
        "frames": [case["frames"][0]],
        "kept_indices": [0],
        "dropped": [2],
        "source_frames": case["frames"],
        "rate": Fraction(24000, 1001),
        "timestamps": [pd.timestamp_actual_rate(0)],
    }
    issues = pd.validate_output(false_drop)["issues"]
    assert any("误删运动帧" in issue for issue in issues)


def test_determinism_and_export_are_stable(tmp_path):
    first = pd.dedup(1, safe_mode=False)
    second = pd.dedup(1, safe_mode=False)
    assert pd.determinism(first, second)["deterministic"]

    path = tmp_path / "pulldown.json"
    returned = pd.export(first, path)
    assert returned == str(path)
    assert path.exists() and path.stat().st_size > 0
