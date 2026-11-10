# 0009 - The survey goes live on the supervisor's advice, without an ethics reference

**Status:** Decided by Jason, 2 Nov 2026. Code changed and tested. Host deployment pending: on 2 Nov the survey host did not answer SSH or ping, and the public URL returned Cloudflare 530.

## Context

- Since 13 Oct the survey server has refused live mode without `SURVEY_ETHICS_REF`. The reason was EAT40005 Progress Report §3.6.2: "Future human testing will secure ethics approval, consent, and data controls first."
- No ethics application or exemption form has been lodged. At Swinburne an exemption is still a form lodged in ERM by a staff Chief Investigator, so there is no reference number to enter.
- Supervisor advice. Week 3 meeting (Teams, 18 Sep 2026, vault `Supervisor meeting - Week 3`): Sheena said no ethics approval is required, and that "real users" means his peers. Her earlier comment (11 Sep) was that if he is not surveying critically ill patients, he most likely will not need ethics approval.
- What the survey is. An anonymous web survey of 4 replay windows from the public OpenOximetry dataset, with no sensor and no body data. It stores no name, email, IP address or user agent. Participants tick consent boxes, including one confirming they are 18 or older.

## Decision

1. **New live-mode basis.** Live mode accepts `SURVEY_ETHICS_BASIS=supervisor` in place of `SURVEY_ETHICS_REF`. Exactly one of the two must be set. Any other basis value fails.
2. **Wording on the information sheet.** The sheet shows one fixed sentence, `SUPERVISOR_ETHICS_NOTE` in `dashboard/server.js`. It names the supervisor and the date of her advice, and says that no ethics committee reviewed the survey. It claims no approval or exemption.
3. **Records.** Each saved response carries `ethics_basis` (`supervisor`, `reference`, or `null` in pilot).
4. **Scope.** This covers the survey as built: peers, anonymous, replay windows only. It does not cover live sensor sessions on other people, walking, or a certified-oximeter comparison. Each of those needs its own decision.

## Consequences

- **The final report must reconcile this with Progress Report §3.6.2.** It should say that the survey ran on the supervisor's advice (18 Sep 2026) and that no Swinburne ethics committee reviewed it. It must not call the survey approved or exempt.
- **The survey measures the unfiltered explanation.** The stimulus does not apply the operator-page withhold rule (cd9d087). Ratings therefore describe the explanation as shown before that rule.
- **Changing the basis later.** If a committee reference arrives, set `SURVEY_ETHICS_REF` instead. Responses collected before then keep `ethics_basis: supervisor`.
