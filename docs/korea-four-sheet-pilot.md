# Korea Connected Four-Sheet Pilot

The local pilot contains four adjacent monochrome 1:50,000 scans. Full scans
remain in ignored `data/raw/`; the manifest, provenance, and registration GCPs
are source controlled.

```text
174 Cheongyang | 178 Gongju
---------------+-----------
173 Buyeo      | 177 Nonsan
```

## Canonical source coordinates

The registration CRS is provisionally `EPSG:5132` (Tokyo 1892). Use the
nominal minute boundaries below. The small `+10.4″` printed beside longitude
is the 1918 Tokyo-origin correction note; adding it to EPSG:5132 coordinates
would apply that correction twice.

| Sheet | West | East | South | North |
|---|---:|---:|---:|---:|
| 174 Cheongyang | 126°45′00″ | 127°00′00″ | 36°20′00″ | 36°30′00″ |
| 178 Gongju | 127°00′00″ | 127°15′00″ | 36°20′00″ | 36°30′00″ |
| 173 Buyeo | 126°45′00″ | 127°00′00″ | 36°10′00″ | 36°20′00″ |
| 177 Nonsan | 127°00′00″ | 127°15′00″ | 36°10′00″ | 36°20′00″ |

Equivalent `EPSG:4301` longitudes add the official `10.405″` offset. Preserve
the printed form in provenance but do not mix coordinate epochs in one file.

## Printed-map pixel GCPs

Pixel origin is the top-left of each full JPEG. These points refer to the inner
printed map neatline, not the outer black frame or the image corners.

| Sheet | NW | NE | SE | SW | Estimated corner uncertainty |
|---|---:|---:|---:|---:|---:|
| 173 Buyeo | 785,430 | 6086,423 | 6116,4762 | 796,4764 | ±3 px |
| 174 Cheongyang | 711,455 | 6035,435 | 6040,4792 | 732,4800 | ±5 px |
| 177 Nonsan | 770,443 | 6072,442 | 6087,4792 | 765,4791 | ±3 px |
| 178 Gongju | 728,457 | 6033,459 | 6048,4792 | 732,4798 | ±3 px |

Four-corner projective RMSE is numerically zero because four points determine
the transform exactly. It is a fit residual, not an accuracy estimate. Shared
edge continuity and later internal control points must provide validation.

## Extraction consequence

All four scans are 8-bit grayscale JPEGs, not faded color scans. Color-profile
extraction cannot work. The next baseline uses full-resolution overlapping
tiles, local background normalization, adaptive dark-line/ridge detection, and
topology-aware tracing. Gongju has the highest measured ink density and is held
out from any initial parameter fitting or model training.

`annotation_tiles.json` defines twelve 1024×1024 full-resolution scenes: three
scene types from each sheet. Buyeo, Cheongyang, and Nonsan are development
sources; all Gongju tiles are `holdout_test`. Generate ignored georeferenced
tiles and a contact sheet with:

```bash
python scripts/create_annotation_package.py \
  examples/korea_four_sheet_pilot/manifest.json \
  examples/korea_four_sheet_pilot/annotation_tiles.json
```

These first twelve tiles are for baseline inspection and annotation design.
They are not yet a sufficient training set and must not be called one.

With a QGIS Python environment active, create the annotation project and its
empty `contour_gt`, `hard_negative`, and `ignore_area` GeoPackage layers:

```bash
python scripts/create_annotation_qgis_project.py \
  data/derived/annotation_package/index.json
```

Generate red review-only candidate overlays before opening or rebuilding the
project. The project detects `candidates/candidate_index.json` automatically:

```bash
python scripts/generate_grayscale_candidates.py \
  data/derived/annotation_package/index.json
python scripts/create_annotation_qgis_project.py \
  data/derived/annotation_package/index.json
```

Rebuilding the project does not overwrite an existing `contour_annotations.gpkg`.
The red overlay includes roads, rivers, and some text; use it to trace or
confirm visible contours, not as ground truth.

Never digitize or tune parameters against the Gongju holdout layers. They may
be opened for final evaluation only after a baseline or model is frozen.

CRS references: [GSI origin history](https://www.gsi.go.jp/sokuchikijun/sankaku-genten.html),
[GSI old-map FAQ](https://service.gsi.go.jp/map-photos/app/help), and
[EPSG transformation 5133](https://epsg.io/5133).
