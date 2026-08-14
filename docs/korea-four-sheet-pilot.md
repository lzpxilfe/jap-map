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

CRS references: [GSI origin history](https://www.gsi.go.jp/sokuchikijun/sankaku-genten.html),
[GSI old-map FAQ](https://service.gsi.go.jp/map-photos/app/help), and
[EPSG transformation 5133](https://epsg.io/5133).
