# Initial Pilot Protocol

This protocol validates the no-training contour baseline.  It does not make a
claim that every historical map type is supported.

## Corpus

Use exactly three legally accessible scans, one for each scenario in
`examples/korea_pilot_manifest.template.json`:

1. `mountain_clear`: clear contour ink and predominantly mountainous terrain.
2. `label_dense`: dense place names that interrupt contours.
3. `degraded_complex`: faded ink, scan noise, or dense symbols.

Keep full scans in `data/raw/` outside Git.  Copy the template manifest to a
local, ignored path, replace its sheet IDs and paths, then create one
registration JSON per scan with **Register Map**.

## Repeatable baseline pass

For each sheet, choose or tune a `MapProfile` for the contour ink.  Run
**Assess Contour Baseline** with the scan, profile, and sheet ID.  The default
maximum analysis dimension is 2,048 pixels so results are fast and comparable.
Save its JSON report in `data/derived/`.

Then run **Extract Contours** at the intended resolution.  Store visible
segments, proposed links, and approved completion segments in a GeoPackage.
Do not approve a proposed link merely to reduce endpoint counts: a false join
is worse than a retained gap.

## Measurements to record

For each whole sheet record visible-contour precision and recall against a
small manually traced reference set, endpoint count, false intersections,
candidate approval rate, registration RMSE, and review time.  The automatic
baseline report supplies descriptive counts only; manual references are needed
before reporting accuracy.

## Decision gate for ML

Do not train a model for this first pass.  Introduce segmentation training only
if the three reports and manual review show that profile tuning and conservative
link review cannot meet the agreed accuracy and review-time targets.  Any later
training set should contain source-rights metadata and examples beyond the
initial Korean pilot before claiming international support.
