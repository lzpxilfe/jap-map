# Portable QGIS Review Workflow

The source-controlled review bundle lets the same annotation session continue
on another computer without transferring the four full-resolution raw scans.
The QGIS project uses relative paths and includes the twelve pilot tiles,
raster and vector proposals, and the mutable review GeoPackage.

## First setup on another computer

1. Install Git and QGIS 3.40 or later.
2. Clone this repository, or pull the latest `main` branch.
3. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**
   and choose `dist/historical-map-tools-0.2.0.zip` from the clone.
4. Enable **Historical Map Tools**.
5. Double-click `Open Review Project.cmd` in the repository root.

The active classification layer is **Annotation layers → Quick review queue —
development only**. Select one or more proposals and classify them with:

- `Ctrl+1`: contour
- `Ctrl+2`: text
- `Ctrl+3`: road or river
- `Ctrl+0`: unsure

The shortcut saves each classification directly to
`data/derived/annotation_package/contour_annotations.gpkg`.

For visual A/B comparison, enable the green **Ink v2 vector proposals — A/B
review only** group and alternate it with the cyan **Automatic vector
proposals — review only** group. Ink proposals are tracked separately and do
not enter or modify the active classification queue.

## Moving between computers safely

The GeoPackage is a binary database and Git cannot merge two edited copies.
Never review on two computers at the same time.

1. Close QGIS on the other computer and push its work.
2. On the computer you are about to use, run `Update Review Workspace.cmd`.
3. Open the project and classify proposals.
4. Completely close QGIS so every database handle is released.
5. Run `Save Review Work.cmd` to commit and push the label database.

If either helper reports an error, stop before opening or editing the project.
Resolve the Git state first. Full raw scans remain ignored because they are
needed only when regenerating or expanding the pilot tiles, not for review.
