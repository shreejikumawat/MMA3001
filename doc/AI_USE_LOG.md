# AI use log

> Running record of material AI use, kept for the report's AI-reflection section
> (MMA3001 brief, §7). **Shreeji: review every entry, correct anything that is
> not accurate, and add your own entries as you work.** Entries are claims you are
> responsible for.

Tool: Claude (Anthropic), via the Claude app with browser and computer access.

| Date | Task | What AI produced | How I checked it | Errors / unhelpful output found | Decision that stayed mine |
|---|---|---|---|---|---|
| 7 Oct 2026 | Gather project requirements | Read Moodle, edStem threads, Unit Overview Spreadsheet and slides; summarised dates, deliverables, rubric | _(to fill: compare against the brief PDF yourself)_ | Its summary briefly mixed "10-min interview" (spreadsheet) with "5 + 3 min" (brief); both are in the sources | Chose Dataset 1 (apple trees) over a self-proposed project |
| 13 Oct 2026 | Dataset download and audit | Audit script; found 562 trees, 42 renamed row-4 winter files, 6 groups of duplicated row-4 winter depth maps, depth direction (far = larger) | _(to fill: open 5 random trees and confirm by eye)_ | Its first filename regex silently skipped the 42 row-4 files; it caught this from the "unparsed" count and fixed it. It first assumed near = bright depth; the audit showed the opposite | _(to fill)_ |
| 13 Oct 2026 | Label registration | Proposed FFT cross-correlation of GT against an image-derived bark map | Visual overlay of 12 trees | Ridge-texture reference was too noisy (peak ratio ≈ 1.09); colour reference fixed ~9/12 trees but was pulled onto grey posts in others, so automatic correction was **not** adopted. Replaced by a shift-compensated metric reported alongside the strict one | _(to fill: do you agree with not auto-correcting?)_ |
| 13 Oct 2026 | Code: metrics, model, training, tests | `src/treeid/*`, `tests/*` (58 tests) | pytest; manufactured-solution tests (known shifts, analytical V-angle) | A metric test failed because several shifts tied within tolerance; the function was changed to return the plateau centre | _(to fill)_ |
| 13 to 14 Oct 2026 | Experiments | 33 training runs (v1, v2, v2r; 64/128/256 px; RGB vs RGB-D; 3 seeds), analysis, profiling | Re-run `scripts/analyse.py`; compare the report tables with `results/summary_table.csv` | v1 masks ~3x too wide (seen in figure, not in mean F1); v2 change did not help; fork rule biased by mask width (found with a thickening test, replaced) | _(to fill)_ |
| 14 Oct 2026 | Model selection | Noticed the 64 px model was best on test rows | `scripts/validation_selection.py`: validation rows alone rank it first (0.290 vs 0.250) | Selecting on test numbers would have been test-set reuse; checked on validation before adopting | _(to fill: do you accept 64 px RGB-D as the final model?)_ |
| 14 Oct 2026 | Report draft | Draft of all sections except your own reflection | _(to fill)_ | Unverifiable journal citation removed; a claim that the tool flags missing depth was false and was corrected | _(to fill)_ |
| 14 Oct 2026 | Manual trunk check | Annotator page and comparison script | **You do the clicking**: this ground truth must come from you, not the AI | | Your annotations |

## Approximate contribution

_(Fill in honestly before submission, e.g. "AI wrote the first draft of most code
and the report outline; I reviewed, re-ran, edited X, Y, Z and wrote the final text
of sections A, B.")_
