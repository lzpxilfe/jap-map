# 등고선 추출 개선 — 실제 도엽 비교

2026-09-12. 여백 OCR와 분리해 **등고선 누락·비등고선 혼입·끊김**을 우선합니다.
부여·청양·논산의 원해상도 개발 타일 9개에서 수정 전후를 같은 입력으로 비교했습니다.
공주 홀드아웃과 기존 검수 GeoPackage는 수정·학습·튜닝에 사용하지 않았습니다.

후속 [주변 방향·간격 분류 실험](contour-neighborhood-experiment.md)에서는 고정 선분 표본을
162개로 넓히고 도엽 단위 학습/검증을 수행했습니다. 도로·하천 혼입 감소와 추가 누락을
함께 확인해 기본값 승격 없이 선별·보수·불확실 레이어를 별도로 제공합니다.

## 이번에 고친 것

1. **디지털 모서리에서 생기는 가짜 분기.** 8방향 연결이 직각 경유 경로와
   대각선 지름길을 동시에 만들면서 정상 곡선을 짧은 가지로 잘랐습니다.
   연결된 100픽셀 계단형 선이 길이 필터 후 0개가 되는 사례를 재현하고 수정했습니다.
   대각선 지름길만 제거하며, 진짜 T/X 분기·닫힌 고리·별개의 평행선은 보존합니다.
   기존 픽셀 사이에 없는 공백 연결을 만들지 않습니다.
2. **미리보기와 실제 파일의 불일치.** 이전 PNG는 필터 전 중심선 전체를 보여줬지만
   GeoJSON에는 일부만 들어갔습니다. 이제 PNG도 실제 내보낸 벡터만 그립니다.
3. **문맥 점수에 따른 별도 검수 후보.** 전체 점수 파일을 보존하면서 낮은 점수의
   선을 제외한 GeoJSON을 별도로 내보냅니다. 도형과 ID는 원래 선의 부분집합이며
   사람 승인이나 `contour_gt` 생성은 하지 않습니다.
4. **QGIS와 실행부.** Processing에 선택적 `MINIMUM_SCORE`를 추가했습니다.
   기본 0은 기존처럼 모든 후보를 유지하며, 0보다 큰 값은 모델 파일이 있어야 합니다.
   ONNX 점수 실행에서 정의되지 않은 이미지 변수를 참조하던 오류도 수정했습니다.

기본 벡터 그래프는 `corner_safe`, 접선 묶기는 기본 꺼짐(`split`)입니다.
adapter는 `jap-map-ink-adapter/3`으로 올려 이전 결과와 실행 ID를 구분합니다.
이전 검수 라벨을 새 선에 자동으로 옮기지 않습니다.

## 실제 점검 결과

원본을 먼저 보고 선정한 **작은 12개 영역**에서, 진하기 ≤176인 원본 잉크가
출력선의 2픽셀 안에 포함되는 비율을 측정했습니다. AI가 고른 영역이며 사람의
독립 정답선이 아닙니다. 따라서 아래 수치는 **전체 도엽 precision/recall이 아닙니다**.
모델·현재 결과를 보고 영역을 바꾸지 않았고, 모든 영역·원본 해시를
`examples/contour_spotchecks.v1.json`에 고정했습니다.

| 점검 영역 | 수정 전 | 모서리 수정만 | 모서리 수정 + 문맥 필터 0.1 |
| --- | ---: | ---: | ---: |
| 등고선 6곳 — 높을수록 좋음 | 91.17% | 91.17% | 91.11% |
| 문자 3곳 — 낮을수록 좋음 | 28.72% | 29.06% | 18.19% |
| 도로·하천 3곳 — 낮을수록 좋음 | 36.41% | 36.57% | 36.57% |

문맥 필터는 점검한 등고선의 피복을 거의 유지하면서 문자 혼입을 줄였습니다.
**도로·하천과 등고선을 구분하는 문제는 아직 해결되지 않았습니다.** 희미한 선,
지명에 가려진 선과 전체 도엽의 잘못된 연결 수도 추가 평가가 필요합니다.
단순히 선 개수가 많아졌다는 이유로 등고선 성능이 개선됐다고 판단하지 않습니다.

9개 타일의 전체 출력은 수정 전 8,347선, 모서리 수정 후 8,736선이며,
문맥 필터 후보는 그중 7,940선입니다. 내보낸 벡터를 다시 그린 픽셀 수는
모서리 수정으로 294,863→306,424가 됐지만, 이는 **선화 보존 진단**이지
정답 등고선 회복량이 아닙니다.

필터는 이번 지도에 맞춰 새로 학습한 모델이 아닙니다. 기존 합성 실험의 고정
로지스틱 모델과 그 실험에서 정한 0.1 임계값을 사용했습니다.
`examples/ink_synthetic_legacy_model.json`에 계수·원래 검증 기록을 보존했습니다.
그 파일의 `promotion_passed`는 과거 **합성** 평가에만 해당하고,
`historical_map_promotion_passed`는 `false`입니다. 0.5·0.9도 개발 진단으로
확인했지만 이번 후보에는 적용하지 않았습니다.

## 채택하지 않은 설정

- 최소 길이를 48픽셀로 올리면 점검 문자 획은 사라졌지만 등고선 피복도
  69.94%로 떨어졌습니다. 기본 길이 18픽셀을 유지했습니다.
- 세 갈래 분기에서 접선이 명확한 두 팔만 묶는 실험도 했습니다. 문자 피복이
  43.82%로 증가하고 등고선 피복은 늘지 않아 기본 경로에 채택하지 않았습니다.
  `--junction-policy tangent_pairs`는 비기본 연구 옵션이며 성공한 개선안이 아닙니다.
- 공백을 자동 보간하거나 가까운 옆 등고선으로 연결하는 동작은 추가하지 않았습니다.

## 바로 확인할 로컬 산출물

- `data/derived/contour-extraction-2026-09-12/contour-comparison.qgz`:
  새 QGIS 비교 프로젝트. 9개 타일, 총 36개 레이어를 저장 후 다시 읽어 검증했습니다.
  각 타일의 **추출 비교 — 하나씩 켜기**에서 수정 전·변환 수정·문맥 필터를
  번갈아 봅니다. 원본은 아래에 유지되며, 결과 레이어는 읽기 전용입니다.
- `data/derived/contour-extraction-2026-09-12/comparison-final/report.html`:
  설치 없이 원본 확대·수정 전후를 나란히 보는 로컬 화면.
- `data/derived/contour-extraction-2026-09-12/context-filter-final/`:
  전체 점수 GeoJSON, 별도 `*-ink-contour-candidates.geojson`, 모델 사본과 해시.
- `docs/evidence/contour-extraction-2026-09-12.json`:
  입력 해시·타일별 진단·12개 점검 영역 수치. 정식 승격은 `false`입니다.

QGIS 3.44.8에서 통합 테스트 6개(실제 Processing 필터 실행 포함), 일반 Python에서
회귀 테스트 133개가 통과했습니다. 비교 프로젝트의 36개 레이어도 모두 유효했습니다.
테스트 개수와 위 점검 수치는 전체 도엽의 정답 등고선 정확도를 보증하지 않습니다.

## 재실행

NumPy·Pillow·SciPy·scikit-image가 있는 환경에서 실행합니다. OCR·PyTorch 설치는
필요 없습니다. 출력은 새 폴더를 사용하고 기존 검수 파일은 그대로 둡니다.

```sh
python scripts/generate_ink_centerline_candidates.py \
  data/derived/annotation_package/index.json \
  --output-dir data/derived/contour-next/ink --workers 2

python scripts/score_ink_segments.py \
  data/derived/annotation_package/index.json examples/ink_synthetic_legacy_model.json \
  --ink-index data/derived/contour-next/ink/ink_candidate_vector_index.json \
  --output-dir data/derived/contour-next/scores --threshold 0.1

python scripts/compare_contour_vector_outputs.py \
  data/derived/annotation_package/index.json --probes examples/contour_spotchecks.v1.json \
  --before data/derived/annotation_package/ink_candidate_vectors_evidence_v2/ink_candidate_vector_index.json \
  --after data/derived/contour-next/ink/ink_candidate_vector_index.json \
  --filtered-index data/derived/contour-next/scores/ink_contour_score_index.json \
  --output data/derived/contour-next/comparison
```

QGIS Python 환경에서는 `scripts/build_contour_comparison_project.py`에 같은
`index`, `--before`, `--after`, `--scores`와 새 `--output ...qgz`를 주면 비교
프로젝트를 만듭니다. 기존 `annotation_project.qgz`나 검수 GeoPackage를 바꾸지 않습니다.

다음 우선순위는 실제 도엽의 문자·도로·하천을 구분하는 문맥 판정과 희미한 선
누락입니다. 이번 작은 영역의 개선을 전체 도엽 정확도로 확대 해석하지 않습니다.
