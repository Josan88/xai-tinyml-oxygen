# TinyKAN web dashboard

Chrome/Edge page for the EAT40006 prototype. Assumed field list (16 Sep 2026): disclaimer, signal, TinyKAN score, candidate, confirmed, glossed explanation, top-3 features, recent-window log. Toy SpO₂ is not shown.

Android is out. This is not a medical device.

## Use without the board

Open `index.html` in any modern browser (or serve the folder). Replay is the 20 OpenOximetry windows in `replay_windows.js` (built from `outputs/openox_tinykan_explanation_rubric_sample.csv` by `node build_replay.js`). Click a row. Score is `firmware_probability` against 0.50. The window id is the rubric file name, not this page's decision. Replay is not live and has no infrared trace, HR, or Confirmed flag.

## Explanation withheld (operator page)

The operator page hides the explanation when it would contradict the result or say nothing (`explanationGate(state, mode)` in `dashboard_core.js`). The lead is the first of the three mean-perturbation deltas, which is the largest |Δ|. The flag side is the one the verdict uses: the Candidate flag in live mode, and score ≥ 0.50 in replay. There are two reasons to withhold. **zero**: no lead, or a lead |Δ| below 0.0005, which prints as 0.000. **opposes**: a flagged window whose lead lowered the score, or an unflagged window whose lead raised it. A withheld window shows a muted sentence in place of the bars, and the same sentence appears in the "This window / Right now" clue, the text summary and the Recent windows "Main clue" cell. A shown window keeps the bars as before, and its clue names the lead feature, its direction and its |Δ| at three decimals. 9 of the 20 replay windows are withheld, all for `opposes`. The survey stimulus is unchanged: `survey.js` still calls `renderDrivers` without opts and `buildExplanation`, and neither of those applies the rule.

Shown bars on the operator page carry two lines above them and drop 0.000 slots. The definition line (`driverNote`) says each bar is how much that feature's value raised or lowered the score compared with the explainer reference (training average, or the typical non-low-oxygen value on label-0 images). The reading line (`driverSummary`) is counted from the signs of the bars whose |Δ| is at least 0.0005, the same cut as the withheld rule: "All three raised the score.", "All three lowered the score.", "Pulled both ways: two raised the score, one lowered it." (or one/two), "Only two features moved this score; …", or "Only one feature moved this score." Because the firmware sends the three largest |Δ|, a 0.000 slot means no other feature moved the score; its name is filled by feature order, so the bar is dropped (`activeDrivers`). The reading adds no threshold of its own, names no expected feature, and says nothing about oxygen, cause or trust. Of the 11 replay windows whose bars are shown, 4 read all raised, 2 all lowered, 5 pulled both ways. The survey stimulus does not get these lines.

Each shown bar also has a one-line caption (`measuresFor`) saying what that feature measures. It does not say oxygen, saturation, or cause, and it is not the bar number. The lead bar gets a static icon when that feature is the red/infrared ratio (`ratio_of_ratios`, `r_beat`) or waveform peakedness (`kurtosis`). The icon is not drawn on the infrared trace. The survey render does not get the caption or the icon.

## Live Bluetooth

The current sensor firmware advertises `OxygenRiskDemo`. Do not flash a parity or timing image for a demo — those builds have no BLE.

1. Chrome or Edge (desktop or Android). Safari/Firefox have no Web Bluetooth.
2. Windows Bluetooth on. Allow the browser prompt.
3. Serve this folder and open **http://localhost:8000** (Web Bluetooth will not run from a `file://` URL):

```
python -m http.server 8000 --bind 127.0.0.1
```

4. **Connect to Arduino** → select **OxygenRiskDemo**.

Packet (firmware): `spo2|status|explanation|red|ir|signal|xai1|xai2|xai3|bpm`. `bpm` is optional; `-` or a missing field shows a dash. Live mode shows field 0 as **SpO2 est** (the Red/IR estimate, one decimal). It is not TinyKAN output and not a calibrated reading. Replay has no SpO₂. The explanation sentence does not use the estimate. BLE updates about twice a second, on the same cadence as the serial status line. On the `nano33ble_v2_f1a_spo2` image (ADR-0010), field 0 comes from the Maxim reference algorithm: an integer shown with one decimal (e.g. `97.0`), `---` when invalid or outside 70–100, uncalibrated, and not TinyKAN; other images keep the demo formula.

Newer images (v2 label-0, P5) append field 10, the model saturation estimate (one decimal, or `-` when unavailable), and field 11, the explainer reference (`label0`; absent = training mean). The **Model sat. est** tile appears only once a packet carries field 10, filled only for a completed window. It estimates OpenOximetry ScalcO₂, is uncalibrated for this sensor and is **not SpO₂**; it is kept out of every sentence, the replay, Recent windows and the survey (wording: `SAT_EST_LABEL`, `SAT_EST_NOTE`).
With `label0` the "Why this score" heading adds "vs a typical non-low-oxygen window" and the zero-withheld sentence names that reference; the withheld rule is unchanged.

## Survey

`/survey` is a ~10-minute user evaluation of the replay view. Each respondent gets 4 of the 20 windows, one per firmware quadrant (TP, FP, TN, FN, judged on `firmware_probability` against 0.50, not on the rubric id). The order is random and the window id is never shown. Questions: what the score means (once), then per window: flagged?, top driver, its direction, four 1–5 ratings (compact, matches, helps, trust), and next action. Then SUS (10 items), four Explanation Satisfaction Scale items (Hoffman et al., 2018: 1, 3, 6, 8), and three optional open questions. Question text lives in `survey_core.js`; the information sheet is in `survey.html`.

Run the server (Node 18+, no packages):

```
node dashboard/server.js
```

It serves the dashboard at `/` and the survey at `/survey` on `127.0.0.1:8000`. Live Bluetooth still works from `http://localhost:8000`.

- **Pilot (default).** Responses go to `data/survey/pilot.jsonl` and the page shows a Pilot banner. Use this for test runs.
- **Live.** `SURVEY_MODE=live` writes `data/survey/responses.jsonl`. It refuses to start without an ethics basis, and exactly one must be set:
  - `SURVEY_ETHICS_REF`: an approval or exemption reference, shown on the information sheet.
  - `SURVEY_ETHICS_BASIS=supervisor`: no reference. The sheet shows the fixed `SUPERVISOR_ETHICS_NOTE` from `server.js`, which gives Sheena's advice of 18 Sep 2026 and says no ethics committee reviewed the survey. This is the basis Jason chose on 2 Nov 2026 (ADR-0009).

  Each record stores `ethics_basis`. Progress Report §3.6.2 commits to ethics approval before human testing, so the final report has to say which basis the survey used.

```powershell
$env:SURVEY_MODE = "live"; $env:SURVEY_ETHICS_BASIS = "supervisor"; node dashboard/server.js
```

Other env: `PORT` (8000), `HOST` (127.0.0.1), `SURVEY_DATA_DIR` (`data/survey`, git-ignored).

Stored per response: a random id, server time, answers, per-window seconds, total duration, and server-side scoring. No IP address, user agent, or cookie. The browser keeps a localStorage flag only to warn about a second submission. Limits: 32 KB body, 20 POSTs per visitor per hour (`CF-Connecting-IP`), a hidden honeypot field, and a whitelist of served files.

Summary for the report:

```
node dashboard/summarize_survey.js          # live responses
node dashboard/summarize_survey.js --pilot  # pilot responses
```

### Cloudflare Tunnel

Add a public hostname to the existing tunnel (Zero Trust → Networks → Tunnels → your tunnel → Public Hostname): for example `survey.<domain>`, service type HTTP, URL `localhost:8000` if cloudflared runs on the same machine as `server.js`, otherwise `<that machine's LAN IP>:8000` with `HOST=0.0.0.0`. Send respondents `https://survey.<domain>/survey`. Do not put Cloudflare Access in front of it. The machine must stay awake while the survey is open.

### Current deployment (13 Oct 2026)

- Host: a separate always-on Windows machine on the author's LAN, managed over SSH with key authentication.
- Public URL: `https://survey.<domain>/survey` (Cloudflared Windows service, remote-managed ingress to `http://localhost:8000`).
- `C:\tinykan-survey\`: `node\` (portable Node 24 LTS, SHA-256 checked), `app\dashboard\` (copy of this folder), `data\survey\` (JSONL), `logs\server.log`, `run.cmd` (env and restart loop).
- Scheduled task `TinyKAN Survey`: at startup, as SYSTEM, no time limit. Sleep on AC is off.
- Mode: **pilot**. `pilot.jsonl` holds one setup check (open answer "setup check via tunnel (Hermes)"). On 2 Nov 2026 Jason decided to go live on the supervisor basis (ADR-0009). The switch has not been made: the host was offline that day (no SSH or ping; public URL 530).
- Go live:
  1. Copy the current `dashboard/` served files and `server.js` into `C:\tinykan-survey\app\dashboard\`. The host copy predates the supervisor basis.
  2. In `C:\tinykan-survey\run.cmd`, set `SURVEY_MODE=live` and `SURVEY_ETHICS_BASIS=supervisor`, and do not set `SURVEY_ETHICS_REF`.
  3. Run `C:\tinykan-survey\restart.cmd` from an elevated prompt (or over SSH). Killing only `node.exe` keeps the old settings, because `run.cmd` reads its `set` lines once.
  4. Check that `https://survey.<domain>/api/survey/config` returns `mode: live` and the ethics note.
- Update code: copy changed files into `C:\tinykan-survey\app\dashboard\` and kill `C:\tinykan-survey\node\node.exe`; the loop restarts it within 10 s.

Caveats for the write-up: remote respondents see replay only (no board), and the rubric's 6-point scores were given to the host CSV sentence, not to the dashboard sentence shown here, so compare survey ratings with the rubric by quadrant, not window by window.

## Files

| File | Role |
|---|---|
| `index.html` | Page |
| `styles.css` | Layout and theme |
| `dashboard_core.js` | Parse packet + explanation sentence |
| `app.js` | BLE, replay, DOM |
| `dashboard_view.js` | Score track and driver bars, shared by dashboard and survey |
| `replay_windows.js` | 20 OpenOximetry windows (generated) |
| `build_replay.js` | Regenerates `replay_windows.js` from the CSV |
| `test_dashboard_core.js` | `node test_dashboard_core.js` |
| `test_dashboard_view.js` | `node test_dashboard_view.js` |
| `survey.html`, `survey.css`, `survey.js` | Survey page |
| `survey_core.js` | Questions, window assignment, validation, scoring |
| `server.js` | Static server plus `POST /api/survey` |
| `summarize_survey.js` | Markdown summary of responses |
| `test_survey_core.js`, `test_server.js` | `node test_survey_core.js`, `node test_server.js` |

## License

Academic project — EAT40006, Swinburne University of Technology Sarawak.
