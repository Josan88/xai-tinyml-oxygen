"""Fig. 1 (on-device pipeline) for the edge-KAN draft: scripts/make_ieee_pipeline_figure_draft.py with the model
box, the explanation box and the timing label changed; nothing else differs. Writes
outputs/ieee-paper-figures-edge-kan/fig1_pipeline.png."""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "make_ieee_pipeline_figure_draft.py"
code = SRC.read_text(encoding="utf-8")
for old, new in (
    (r'"TinyKAN 16 → 12 → 1 (321 parameters)\nP ≥ 0.50 Candidate; 2 in a row Confirmed"',
     r'"KAN 16 → 12 → 1, 1632 parameters\nP ≥ 0.50 Candidate; 2 in a row Confirmed"'),
    (r'"Mean perturbation: 1 + 16 forwards\ntop three |Δi| named"',
     r'"Mean perturbation: 1 + 16 replacements\ntop three |Δi| named"'),
    (r"median\n444.3 ms", r"median\n433.9 ms"),
    ('/ "ieee-paper-figures-proposal" / "fig_pipeline_draft.png"', '/ "ieee-paper-figures-edge-kan" / "fig1_pipeline.png"'),
):
    if code.count(old) != 1:
        raise SystemExit(f"not found once: {old}")
    code = code.replace(old, new)
exec(compile(code, str(SRC), "exec"), {"__name__": "__main__", "__file__": str(SRC)})
