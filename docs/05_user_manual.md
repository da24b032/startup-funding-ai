# 5. User Manual (for non-technical users)

## What this tool does
You describe a startup (where it is, what it does, how much funding it has raised, who invested, and what it has achieved). The tool estimates the chance that the startup will be **acquired** by another company rather than **close down**, and tells you which details pushed the estimate up or down.

> **Please remember:** this is a statistical estimate learned from about 900 past startups. It is **not** a guarantee and must not be your only basis for an investment, lending or hiring decision.

## Opening the tool
Your administrator will give you a web address (for a local install: `http://localhost:8501`). Open it in any modern browser. The menu on the left lists four pages.

---

## Page 1 — Predict a startup's outcome
1. *(Optional)* Under **Start from an example**, choose a sample startup. The form fills itself so you can see how it works.
2. Fill in the form. It is split into three tabs — **Basics**, **Funding**, **Investors**, **Traction**. Hover over the **?** next to a field for an explanation.
   - Drop-downs (State, Industry): pick from the list.
   - On/off switches (e.g. *Venture capital backing*): switch **on** for "yes".
   - Numbers: type a value or use **+ / −**. The form will not accept impossible values.
   - *Years from founding to first milestone* is optional: tick **provide** only if the startup has milestones.
3. Press **Predict outcome**.

**Reading the result**
- A green card means **Likely to be acquired**; a red card means **Likely to close**.
- The percentage is the estimated chance. Around 50 % means the tool is unsure.
- **What influenced this estimate** lists the details that moved the result most, e.g. *"Venture capital backing raises the acquisition chance by about 2 percentage points"*. These describe how the model behaves; they are not proof of cause.

**If you see a red message**, read it — it names the field to fix (for example "last funding cannot be earlier than first funding").

## Page 2 — Batch prediction (many startups at once)
1. Click **Download a blank CSV template** and open it in Excel or Google Sheets.
2. Fill one row per startup (use the *What columns are needed?* box for the meaning of each column). Use 0/1 for yes/no columns. Leave the milestone-age cells empty if there are no milestones.
3. Save as CSV and upload it with **Choose a CSV file** (up to 500 startups).
4. Press **Run predictions**. You will see totals and a table; click **Download results** to save it.

## Page 3 — Pipeline console *(for the person who maintains the model)*
Shows how the model was built, in five tabs:
- **Pipeline flow** — the sequence of steps (prepare data → train → test).
- **Data engineering** — how many rows were processed, how long each step took, data-quality issues found.
- **Model training** — the three candidate models compared, and which inputs matter most.
- **Run console** — every pipeline run with ✅ success or ❌ failed; switch on **Show only failures** to see problems.
- **Traceability** — exactly which code, data and experiment produced the model in use.
Use **Refresh** to update. Links open the MLflow, Grafana and Prometheus tools.

## Page 4 — Monitoring
Shows whether the startups being entered still look like the ones the model learned from.
- "No drift detected" — everything normal.
- "Drift detected in N features" — the incoming data has changed; ask the model maintainer to review or retrain. This does **not** mean predictions are wrong.
- You need at least 30 predictions before this can be assessed. **Reset window** starts counting again.

---

## Troubleshooting
| What you see | What to do |
|---|---|
| "The prediction service is not reachable" | The back-end is not running. Contact your administrator, or start it (see README), then reload the page. |
| "The model is not loaded yet" | The model has not been trained. Run the training step (README → *Quick start*). |
| "Please fix the following: …" | One of your inputs is not allowed; the message says which. |
| Batch upload: "missing these columns" | Use the downloadable template and keep its column names. |
| Results look surprising | Check the *What influenced this estimate* list; remember it is an estimate. |

## Glossary
**Acquired** — bought by another company. **Closed** — shut down. **Funding round** — a time the startup raised money. **Series A–D** — successive, larger funding rounds. **Milestone** — a recorded achievement (product launch, partnership…). **Drift** — the incoming data has changed compared with the training data.

## Responsible use
The data covers past, mostly US startups and may reflect their biases. Use the tool for learning and as one input among many.
