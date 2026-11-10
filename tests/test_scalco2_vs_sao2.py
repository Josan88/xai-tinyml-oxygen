"""Tests for scripts/analyze_scalco2_vs_sao2.py pure helpers (written before the script).

Pre-registration: docs/scalco2-vs-sao2-preregistration.md (456b006). Host only, no model is
trained or loaded. Every test uses small synthetic frames; none opens the real CSVs.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_scalco2_vs_sao2 as sv  # noqa: E402


# ----------------------------------------------------------------------------- draw aggregation

def bloodgas_frame():
    nan = np.nan
    return pd.DataFrame({
        "patient_id":   ["p1", "p1", "p1", "p1", "p1", "p1", "p2", "p2", "p9"],
        "encounter_id": ["e1", "e1", "e1", "e1", "e1", "e1", "e2", "e2", None],
        "sample":       [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        "so2":          ["90", 80.0, nan, 84.0, 70.0, nan, 95.0, 97.0, 88.0],
        "ScalcO2":      [nan, 85.0, "70", 87.0, nan, 80.0, 93.0, 99.0, 87.0],
    })


def test_draw_aggregation_means_over_both_numeric_rows_only():
    draws, counts = sv.aggregate_draws(bloodgas_frame())
    assert list(draws.columns[:6]) == ["encounter_id", "patient_id", "sample", "n_rows", "SaO2_draw", "ScalcO2_draw"]
    e1 = draws[(draws["encounter_id"] == "e1") & (draws["sample"] == 1)].iloc[0]
    assert e1["n_rows"] == 2                                       # rows 2 and 4 only
    assert e1["SaO2_draw"] == pytest.approx(82.0)                  # not (90 + 80 + 84) / 3
    assert e1["ScalcO2_draw"] == pytest.approx(86.0)               # not (85 + 70 + 87) / 3
    e2 = draws[draws["encounter_id"] == "e2"].iloc[0]
    assert (e2["SaO2_draw"], e2["ScalcO2_draw"], e2["n_rows"]) == (96.0, 96.0, 2)


def test_group_without_both_numeric_row_is_not_a_draw():
    draws, counts = sv.aggregate_draws(bloodgas_frame())
    keys = set(zip(draws["encounter_id"], draws["sample"]))
    assert keys == {("e1", 1), ("e2", 1)}                          # e1/2 and e1/3 have one value only
    assert draws["sample"].dtype == np.int64
    assert counts["draws"] == 2 and counts["rows_both_numeric"] == 4


def test_missing_encounter_rows_are_dropped_and_counted():
    draws, counts = sv.aggregate_draws(bloodgas_frame())
    assert counts["rows_no_encounter_id"] == 1
    assert "p9" not in set(draws["patient_id"])


def test_draw_with_two_patient_ids_raises():
    frame = bloodgas_frame()
    frame.loc[3, "patient_id"] = "pX"                              # a both-numeric row of draw e1/1
    with pytest.raises(AssertionError):
        sv.aggregate_draws(frame)


def test_draw_patient_must_match_encounter_table():
    ok = {"e1": "p1", "e2": "p2"}
    sv.aggregate_draws(bloodgas_frame(), encounter_patient=ok)
    with pytest.raises(AssertionError):
        sv.aggregate_draws(bloodgas_frame(), encounter_patient={"e1": "p1", "e2": "p3"})
    with pytest.raises(AssertionError):
        sv.aggregate_draws(bloodgas_frame(), encounter_patient={"e1": "p1"})


def test_non_integer_sample_raises():
    frame = bloodgas_frame()
    frame.loc[0, "sample"] = 1.5
    with pytest.raises(AssertionError):
        sv.aggregate_draws(frame)


# ----------------------------------------------------------------------------- pulse oximeter validity

VALUES = ["text", np.nan, "0", "-1", "4", "9.99", "10", "100", "100.0001", "101", "493"]
PRIMARY = ["non_numeric", "non_numeric", "below_range", "below_range", "below_range", "below_range",
           "valid", "valid", "above_100", "above_100", "above_100"]
PLAN = ["non_numeric", "non_numeric", "below_range", "below_range", "valid", "valid",
        "valid", "valid", "above_100", "above_100", "above_100"]


def test_validity_rule_primary_keeps_10_to_100():
    sat, reason = sv.saturation_validity(pd.Series(VALUES, dtype=object), rule="primary")
    assert list(reason) == PRIMARY
    assert sat[6] == 10.0 and sat[7] == 100.0


def test_validity_rule_plan_sensitivity_keeps_above_0_to_100():
    _, reason = sv.saturation_validity(pd.Series(VALUES, dtype=object), rule="plan")
    assert list(reason) == PLAN


def pulse_frame(saturation):
    n = len(saturation)
    return pd.DataFrame({
        "encounter_id": ["e1"] * n,
        "device": [f"{i}" for i in range(1, n + 1)],
        "probe_location": [np.nan] * n,
        "sample_number": [1.0] * n,
        "saturation": saturation,
        "pi": [1.0] * n,
    })


def test_clean_pulse_drop_order_and_counts():
    frame = pulse_frame(["95", "93", "very unstable", "4", "493", "101", "88", "92"])
    frame.loc[0, "device"] = np.nan                                 # no device: dropped first
    frame.loc[1, "sample_number"] = np.nan                          # NaN sample_number: dropped second
    frame.loc[2, "sample_number"] = np.nan                          # text note on a NaN sample_number row
    clean, counts = sv.clean_pulse(frame, rule="primary")
    assert counts["no_device"] == 1
    assert counts["sample_number_nan"] == 2
    assert counts["invalid_non_numeric"] == 0                       # the text row went with sample_number NaN
    assert counts["invalid_above_100"] == 2
    assert counts["invalid_below_range"] == 1
    assert sorted(clean["spo2"].tolist()) == [88.0, 92.0]
    assert clean["sample_number"].dtype == np.int64
    plan, plan_counts = sv.clean_pulse(frame, rule="plan")
    assert sorted(plan["spo2"].tolist()) == [4.0, 88.0, 92.0]
    assert plan_counts["invalid_below_range"] == 0


def test_clean_pulse_duplicate_key_raises():
    frame = pulse_frame(["95", "93"])
    frame["device"] = ["2", "2.0"]                                  # one device spelled two ways, same sample
    with pytest.raises(AssertionError):
        sv.clean_pulse(frame)


# ----------------------------------------------------------------------------- device label

def test_device_label_normalisation():
    labels = sv.device_label(pd.Series(["2", "2.0", "56 .01", "42.01", "42.02", "42", "59.0"]))
    assert list(labels) == ["2.00", "2.00", "56.01", "42.01", "42.02", "42.00", "59.00"]
    assert len(set(labels[3:6])) == 3
    assert list(sv.integer_model(pd.Series(["42.02", "42.00", "56.01"]))) == [42, 42, 56]


# ----------------------------------------------------------------------------- join

def test_pair_readings_inner_join_on_draws():
    draws = pd.DataFrame({"encounter_id": ["e1", "e1", "e2"], "patient_id": ["p1", "p1", "p2"],
                          "sample": np.array([1, 2, 1], dtype=np.int64), "n_rows": [1, 1, 1],
                          "SaO2_draw": [85.0, 92.0, 97.0], "ScalcO2_draw": [86.0, 91.0, 96.0]})
    pulse = pd.DataFrame({"encounter_id": ["e1", "e1", "e1", "e1", "e2", "e3"],
                          "device": ["2", "3", "2", "2", "2", "2"],
                          "sample_number": [1.0, 1.0, 2.0, 5.0, 1.0, 1.0],
                          "saturation": ["84", "88", "93", "90", "98", "99"]})
    clean, _ = sv.clean_pulse(pulse)
    pairs = sv.pair_readings(clean, draws)
    assert len(pairs) == 4                                          # e1/5 and e3 are not draws
    assert not pairs.duplicated(["encounter_id", "sample", "device_label"]).any()
    row = pairs[(pairs["encounter_id"] == "e1") & (pairs["sample"] == 1) & (pairs["device_label"] == "3.00")].iloc[0]
    assert (row["spo2"], row["SaO2_draw"], row["patient_id"]) == (88.0, 85.0, "p1")


# ----------------------------------------------------------------------------- ARMS / bias / SD / LoA

def test_bland_altman_hand_computed():
    x = np.array([92.0, 88.0, 95.0, 81.0])
    r = np.array([90.0, 89.0, 91.0, 84.0])
    d = np.array([2.0, -1.0, 4.0, -3.0])
    out = sv.bland_altman(x, r)
    assert out["n"] == 4
    assert out["bias"] == pytest.approx(0.5)
    assert out["arms"] == pytest.approx(np.sqrt(30.0 / 4.0))
    sd = np.sqrt(np.sum((d - 0.5) ** 2) / 3.0)
    assert out["sd"] == pytest.approx(sd)
    assert out["loa_low"] == pytest.approx(0.5 - 1.96 * sd)
    assert out["loa_high"] == pytest.approx(0.5 + 1.96 * sd)
    assert sv.arms(x, r) == pytest.approx(np.sqrt(7.5))
    assert sv.bias(x, r) == pytest.approx(0.5)


def test_weighted_arms_and_bias_equal_replicated():
    x = np.array([92.0, 88.0, 95.0, 81.0])
    r = np.array([90.0, 89.0, 91.0, 84.0])
    w = np.array([3, 0, 1, 2])
    xr, rr = np.repeat(x, w), np.repeat(r, w)
    assert sv.arms(x, r, w) == pytest.approx(sv.arms(xr, rr))
    assert sv.bias(x, r, w) == pytest.approx(sv.bias(xr, rr))


# ----------------------------------------------------------------------------- kappa

def test_kappa_equals_sklearn_hand_case():
    x = np.array([85.0, 88.0, 91.0, 95.0, 89.0, 92.0, 80.0, 97.0])
    r = np.array([84.0, 91.0, 92.0, 96.0, 88.0, 89.0, 82.0, 99.0])
    y1, y2 = (x < 90).astype(int), (r < 90).astype(int)
    assert sv.kappa_below(x, r) == pytest.approx(cohen_kappa_score(y1, y2, labels=[0, 1]))
    # hand: 8 units, agree 6 (tp 3, tn 3), x-pos 4, r-pos 4 -> po 0.75, pe 0.5, kappa 0.5
    assert sv.kappa_below(x, r) == pytest.approx(0.5)


def test_kappa_with_integer_weights_equals_replicated():
    x = np.array([85.0, 88.0, 91.0, 95.0, 89.0, 92.0, 80.0, 97.0])
    r = np.array([84.0, 91.0, 92.0, 96.0, 88.0, 89.0, 82.0, 99.0])
    w = np.array([2, 0, 1, 3, 1, 2, 1, 1])
    assert sv.kappa_below(x, r, sample_weight=w) == pytest.approx(sv.kappa_below(np.repeat(x, w), np.repeat(r, w)))


def test_kappa_one_class_returns_nan():
    assert np.isnan(sv.kappa_below(np.array([95.0, 96.0]), np.array([97.0, 98.0])))
    assert np.isnan(sv.kappa_binary(np.array([1, 1]), np.array([1, 1])))


def test_below_90_threshold_is_strict():
    assert list(sv.below90(np.array([89.999, 90.0, 90.0001]))) == [1, 0, 0]


# ----------------------------------------------------------------------------- sensitivity / specificity

def test_sens_spec_with_sao2_truth():
    x = np.array([85.0, 88.0, 91.0, 95.0, 89.0, 92.0, 80.0, 97.0])  # ScalcO2
    r = np.array([84.0, 91.0, 92.0, 96.0, 88.0, 89.0, 82.0, 90.0])  # SaO2 (90.0 is not < 90)
    sens, spec = sv.sens_spec(sv.below90(x), sv.below90(r))
    assert sens == pytest.approx(3 / 4)                             # r<90: 84, 88, 89, 82 -> x<90 on 3
    assert spec == pytest.approx(3 / 4)                             # r>=90: 91, 92, 96, 90 -> x>=90 on 3
    w = np.array([2, 1, 1, 1, 0, 1, 1, 3])
    ws, wp = sv.sens_spec(sv.below90(x), sv.below90(r), w)
    assert ws == pytest.approx(3 / 4) and wp == pytest.approx(5 / 6)


def test_sens_spec_empty_denominator_is_nan():
    sens, spec = sv.sens_spec(np.array([0, 1]), np.array([0, 0]))
    assert np.isnan(sens) and spec == pytest.approx(0.5)


# ----------------------------------------------------------------------------- benchmark devices

def device_pairs():
    rows = []
    def add(device, n, sao2):
        for i in range(n):
            rows.append({"encounter_id": f"e{i}", "patient_id": f"p{i % 7}", "sample": i, "device_label": device,
                         "model": int(float(device)), "spo2": sao2 + (i % 3) - 1, "SaO2_draw": sao2, "ScalcO2_draw": sao2})
    add("1.00", 99, 85.0)          # 99 in range, 99 overall
    add("2.00", 100, 85.0)         # 100 in range
    add("3.00", 50, 85.0)
    add("3.00", 50, 65.0)          # 50 in range, 100 overall
    add("4.00", 99, 101.0)         # 0 in range, 99 overall
    return pd.DataFrame(rows)


def test_benchmark_rule_uses_its_own_draw_set():
    table = sv.device_table(device_pairs(), column="device_label", min_n=100).set_index("device")
    assert table.loc["1.00", "n_p1"] == 99 and not table.loc["1.00", "benchmark_p1"]
    assert not table.loc["1.00", "benchmark_p2"]
    assert table.loc["2.00", "n_p1"] == 100 and table.loc["2.00", "benchmark_p1"] and table.loc["2.00", "benchmark_p2"]
    assert table.loc["3.00", "n_p1"] == 50 and table.loc["3.00", "n_all"] == 100
    assert not table.loc["3.00", "benchmark_p1"] and table.loc["3.00", "benchmark_p2"]
    assert table.loc["4.00", "n_p1"] == 0 and np.isnan(table.loc["4.00", "arms_d"])
    assert list(table.index) == sorted(table.index)


def test_p1_range_is_inclusive():
    assert list(sv.in_p1(np.array([69.99, 70.0, 85.0, 100.0, 100.01]))) == [False, True, True, True, False]


# ----------------------------------------------------------------------------- gate

def test_gate_pass_pass():
    g = sv.gate(arms_s=2.0, device_arms=[2.5, 3.0, 4.0], kappa_s=0.8, device_kappa=[0.5, 0.6, 0.7])
    assert g["P1"]["pass"] and g["P2"]["pass"]
    assert g["verdict"] == "PASS" and g["failed_parts"] == []
    assert g["P1"]["device_median"] == 3.0 and g["P1"]["n_devices"] == 3


def test_gate_p1_fail():
    g = sv.gate(arms_s=3.5, device_arms=[2.5, 3.0, 4.0], kappa_s=0.8, device_kappa=[0.5, 0.6, 0.7])
    assert not g["P1"]["pass"] and g["P2"]["pass"]
    assert g["verdict"] == "FAIL" and g["failed_parts"] == ["P1"]


def test_gate_p2_fail():
    g = sv.gate(arms_s=2.0, device_arms=[2.5, 3.0, 4.0], kappa_s=0.55, device_kappa=[0.5, 0.6, 0.7])
    assert g["P1"]["pass"] and not g["P2"]["pass"]
    assert g["verdict"] == "FAIL" and g["failed_parts"] == ["P2"]


def test_gate_both_fail():
    g = sv.gate(arms_s=9.0, device_arms=[2.5, 3.0, 4.0], kappa_s=0.1, device_kappa=[0.5, 0.6, 0.7])
    assert g["verdict"] == "FAIL" and g["failed_parts"] == ["P1", "P2"]


def test_gate_ties_pass():
    g = sv.gate(arms_s=3.0, device_arms=[2.5, 3.0, 4.0], kappa_s=0.6, device_kappa=[0.5, 0.6, 0.7])
    assert g["P1"]["pass"] and g["P2"]["pass"] and g["verdict"] == "PASS"


def test_gate_even_count_median():
    g = sv.gate(arms_s=2.75, device_arms=[2.0, 2.5, 3.0, 4.0], kappa_s=0.55, device_kappa=[0.4, 0.5, 0.6, 0.9])
    assert g["P1"]["device_median"] == pytest.approx(2.75) and g["P1"]["pass"]
    assert g["P2"]["device_median"] == pytest.approx(0.55) and g["P2"]["pass"]


def test_gate_nonfinite_kappa_excluded_and_counted():
    g = sv.gate(arms_s=2.0, device_arms=[2.5, 3.0, 4.0], kappa_s=0.65, device_kappa=[0.5, np.nan, 0.6, 0.7, np.inf])
    assert g["P2"]["n_devices"] == 3 and g["P2"]["n_excluded_nonfinite"] == 2
    assert g["P2"]["device_median"] == pytest.approx(0.6) and g["P2"]["pass"]


def test_rank_among_devices():
    assert sv.rank_among(2.0, [1.0, 2.0, 3.0], higher_is_better=False) == 2   # only 1.0 strictly better
    assert sv.rank_among(0.9, [0.5, 0.95, 0.9], higher_is_better=True) == 2
    assert sv.rank_among(0.1, [np.nan, 0.5], higher_is_better=True) == 2      # NaN devices never count


# ----------------------------------------------------------------------------- bootstrap indices

def test_bootstrap_counts_shape_seed_and_groups():
    patients = np.array(["p10", "p2", "p10", "p1", "p2", "p2"], dtype=object)
    groups, group_index, counts = sv.bootstrap_counts(patients, n_boot=50, seed=20261108)
    assert list(groups) == ["p1", "p10", "p2"]                      # np.unique string order
    assert list(group_index) == [1, 2, 1, 0, 2, 2]
    assert counts.shape == (50, 3)
    assert (counts.sum(axis=1) == 3).all()
    draws = np.random.default_rng(20261108).integers(0, 3, size=(50, 3))
    for b in range(50):
        assert np.array_equal(counts[b], np.bincount(draws[b], minlength=3))
    _, _, again = sv.bootstrap_counts(patients, n_boot=50, seed=20261108)
    assert np.array_equal(counts, again)
    _, _, other = sv.bootstrap_counts(patients, n_boot=50, seed=1)
    assert not np.array_equal(counts, other)


def test_bootstrap_group_ids_are_strings():
    groups, group_index, _ = sv.bootstrap_counts(np.array([10, 9, 10]), n_boot=5, seed=0)
    assert list(groups) == ["10", "9"] and list(group_index) == [0, 1, 0]


# ----------------------------------------------------------------------------- S1 draw time

def hz2_frame():
    return pd.DataFrame({
        "Sample":    [np.nan, np.nan, np.nan, 3.0, 3.0, np.nan, 4.0, 4.0, 4.0, 4.0, 5.0],
        "ScalcO2":   [95.0, np.nan, 94.0, 93.0, np.nan, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0],
        "Timestamp": [None, "2218-10-03 14:16:30.000", "2218-10-03 14:16:30.500", "2218-10-03 14:16:31.000",
                      "2218-10-03 14:16:31.500", "2218-10-03 14:16:32.000", "2218-10-03 14:16:40.000",
                      "2218-10-03 14:16:41.000", "2218-10-03 14:16:43.000", "2218-10-03 14:16:50.000", "garbage"],
    })


def test_elapsed_from_first_valid_timestamp_row_even_without_scalco2():
    hz2 = sv.elapsed_2hz(hz2_frame())
    assert len(hz2) == 9                                            # None and "garbage" rows dropped
    assert hz2["elapsed_seconds"].iloc[0] == 0.0                    # the row with ScalcO2 NaN is t0
    assert hz2["elapsed_seconds"].iloc[1] == 0.5


def test_draw_time_is_median_of_marker_rows():
    markers = sv.marker_rows(sv.elapsed_2hz(hz2_frame()))
    assert markers["Sample"].dtype == np.int64
    t, n = sv.draw_time(markers, 3)
    assert n == 2 and t == pytest.approx(1.25)                      # median of 1.0, 1.5 (even count)
    t, n = sv.draw_time(markers, 4)
    assert n == 4 and t == pytest.approx((11.0 + 13.0) / 2)        # 10, 11, 13, 20
    t, n = sv.draw_time(markers, 5)                                 # its only row had an invalid Timestamp
    assert n == 0 and np.isnan(t)


def test_non_integer_marker_raises():
    frame = hz2_frame()
    frame.loc[3, "Sample"] = 3.5
    with pytest.raises(AssertionError):
        sv.marker_rows(sv.elapsed_2hz(frame))


def test_no_marker_status():
    assert sv.match_window(np.nan, np.array([0.0]), np.array([10.0]), 0.0) == ("no marker", None)


# ----------------------------------------------------------------------------- window matching

STARTS = np.array([0.0, 5.0, 10.0, 30.0])
ENDS = STARTS + 10.0


def test_one_containing_window():
    assert sv.match_window(2.0, STARTS, ENDS, 0.0) == ("matched", 0)
    assert sv.match_window(35.0, STARTS, ENDS, 0.0) == ("matched", 3)


def test_span_start_included_end_excluded():
    assert sv.match_window(30.0, STARTS, ENDS, 0.0) == ("matched", 3)      # 30 = start of [30, 40)
    assert sv.match_window(40.0, STARTS, ENDS, 0.0) == ("no window", None)  # 40 = end of [30, 40)


def test_two_windows_nearest_centre():
    # 6.0 sits in [0, 10) (centre 5) and [5, 15) (centre 10): nearest is the first
    assert sv.match_window(6.0, STARTS, ENDS, 0.0) == ("matched", 0)
    # 8.0: centres 5 (|3|) and 10 (|2|) -> the second
    assert sv.match_window(8.0, STARTS, ENDS, 0.0) == ("matched", 1)


def test_exact_tie_goes_to_earlier_start():
    # 7.5: centres 5 and 10, both 2.5 away
    assert sv.match_window(7.5, STARTS, ENDS, 0.0) == ("matched", 0)
    # order of the inputs must not matter
    assert sv.match_window(7.5, STARTS[::-1].copy(), ENDS[::-1].copy(), 0.0) == ("matched", 3)


def test_no_window():
    assert sv.match_window(25.0, STARTS, ENDS, 0.0) == ("no window", None)


def test_delta_shifts_the_span():
    assert sv.match_window(42.0, STARTS, ENDS, 5.0) == ("matched", 3)      # [35, 45)
    assert sv.match_window(34.0, STARTS, ENDS, 5.0) == ("no window", None)
    assert sv.match_window(2.0, STARTS, ENDS, -3.0) == ("matched", 0)       # in [-3, 7) and [2, 12): centres 2 and 7
    assert sv.match_window(4.5, STARTS, ENDS, -3.0) == ("matched", 0)      # centres 2 and 7, tie -> earlier


def test_three_windows_raise():
    starts = np.array([0.0, 2.0, 4.0])
    with pytest.raises(AssertionError):
        sv.match_window(5.0, starts, starts + 10.0, 0.0)


# ----------------------------------------------------------------------------- out-of-fold join

def oof_tables():
    windows = pd.DataFrame({"recording_id": ["a", "a", "b", "b"], "patient_id": ["x", "x", "y", "y"],
                            "start_time_seconds": [0.0, 5.0, 0.0, 5.0], "label": [1, 0, 0, 1]})
    oof = pd.DataFrame({"row_index": [0, 1, 2, 3], "recording_id": ["a", "a", "b", "b"], "patient_id": ["x", "x", "y", "y"],
                        "start_time_seconds": [0.0, 5.0, 0.0, 5.0], "fold": [1, 1, 2, 2], "y_true": [1, 0, 0, 1],
                        "y_prob": [0.9, 0.2, 0.4, 0.6]})
    return windows, oof


def test_oof_row_index_join_equals_key_join():
    windows, oof = oof_tables()
    positions = np.array([3, 0, 3])
    got = sv.join_oof(windows, oof, positions)
    keyed = windows.iloc[positions][["recording_id", "start_time_seconds"]].merge(
        oof, on=["recording_id", "start_time_seconds"], how="left", validate="many_to_one")
    assert got["y_prob"].tolist() == keyed["y_prob"].tolist() == [0.6, 0.9, 0.6]
    assert got["row_index"].tolist() == [3, 0, 3]


def test_oof_join_raises_on_key_mismatch():
    windows, oof = oof_tables()
    oof.loc[2, "start_time_seconds"] = 10.0
    with pytest.raises(AssertionError):
        sv.join_oof(windows, oof, np.array([2]))


# ----------------------------------------------------------------------------- CLI guards

def test_missing_prereg_commit_raises():
    with pytest.raises(SystemExit):
        sv.main(["--smoke", "--out", str(ROOT / "build" / "scalco2-vs-sao2-v1" / "never")])


def test_existing_output_dir_raises_before_writing(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        sv.main(["--prereg-commit", "456b006", "--out", str(out)])
    assert list(out.iterdir()) == []


def test_unknown_prereg_commit_refused(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(SystemExit):
        sv.main(["--prereg-commit", "0" * 40, "--out", str(out)])
    assert not out.exists()


def test_smoke_refuses_official_outputs_and_outside_build(tmp_path):
    with pytest.raises(SystemExit):
        sv.main(["--smoke", "--prereg-commit", "456b006", "--out", str(ROOT / "outputs" / "scalco2-vs-sao2-v1")])
    with pytest.raises(SystemExit):
        sv.main(["--smoke", "--prereg-commit", "456b006", "--out", str(tmp_path / "x")])
    assert not (ROOT / "outputs" / "scalco2-vs-sao2-v1").exists()
