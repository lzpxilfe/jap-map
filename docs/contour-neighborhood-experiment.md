# 주변 방향·간격으로 등고선 분류 강화

2026-09-12. [이전 중심선 변환 개선](contour-extraction-progress.md) 다음 단계입니다.
부여·청양·논산의 실제 개발 타일 9개에서 주변 선 배열을 이용하는 분류기를 구현하고,
도엽을 하나씩 제외한 학습·출력까지 실행했습니다. 공주 홀드아웃, 기존 사람 검수
GeoPackage, 원본 지도, QGIS 기본 추출 설정은 변경하지 않았습니다.

후속 [접촉 선 묶음·새 표본 평가와 사람 검수 인계](contour-forward-human-handoff.md)를
완료했습니다. 아래는 이전 실험의 기록이며, 새 표본 결과로 덮어쓰지 않습니다.

## 결과부터

주변 문맥은 도로·하천 구분에 도움이 됐지만, 누락과의 상충 관계가 남았습니다.
아래는 **AI가 원본에서 임시 판독한 고정 선분 표본**의 결과입니다. 사람 정답이나
전체 도엽 정확도가 아닙니다. 선 길이 구간별 같은 수를 뽑았으므로 모집단 비율도 아닙니다.

| 방법 | 등고선 유지 / 106 | 비등고선 제외 / 42 | 도로·하천 제외 / 22 |
| --- | ---: | ---: | ---: |
| 기존 합성 모델, 고정 0.1 | 104 | 13 | 0 |
| 새 주변 문맥, 선별 기준 | 101 | 25 | 12 |
| 새 주변 문맥, 보수 기준 | 104 | 14 | 7 |

선별 기준은 비등고선 12개를 더 제외하는 대신 등고선 3개를 추가로 놓쳤습니다.
보수 기준은 등고선 유지 수가 기존과 같지만, **동일한 선을 유지한 것은 아닙니다**.
도로 제외는 늘었으나 기호 혼입이 증가해 전체 비등고선 제외의 순증은 1개입니다.
따라서 새 분류기를 기본값으로 승격하지 않았습니다. 기존과 새 후보를 단순 교집합으로
만드는 것도 등고선 추가 누락이 있어 기본 적용하지 않았습니다.

- 선별 기준이 놓친 AI 등고선: `S055`, `S058`, `S077`, `S086`, `S090`.
- 보수 기준이 놓친 AI 등고선: `S058`, `S090`.
- 기존 기준이 놓친 AI 등고선: `S090`, `S117`.
- `S084`의 도로 획은 새 후보에서 제외됐지만, `S157`의 긴 도로는 여전히 남습니다.
  `S117`의 짧은 등고선은 회복됐고, `S058`은 추가 누락 사례입니다.

`context-tradeoff-examples.png`의 S084·S058은 **실험 후 고른 설명용 사례**입니다.
그 두 사례를 별도의 성능 평가 집합으로 취급하지 않습니다. 비교 화면에는 성공·실패·
모호한 표본을 포함한 162개 모두가 있습니다.

## 무엇을 추가했나

선 하나의 길이·직선성·획 진하기 등 기존 11개 특징에 **23개 주변 특징**을 더했습니다.

- 원해상도 반경 12·36·84픽셀에서 주변 획의 방향 일관성, 대상 선의 법선과의 정렬,
  정렬 변화, 잉크 밀도를 계산합니다.
- 선을 따라 5곳에서 수직 방향 ±48픽셀을 살펴 이웃 획의 수, 양쪽 존재 여부,
  간격 변동, 중심 획 폭을 측정합니다. 평행한 도로도 있으므로 방향만으로 판정하지 않습니다.
- 8픽셀 간격으로 완화한 회전량을 이용해 한 픽셀짜리 계단 모서리와 큰 굴곡을 구분합니다.
- NumPy·SciPy 기반의 작은 로지스틱 모델입니다. 영상 미분은 타일마다 한 번 캐시합니다.
  새 모델 다운로드, PyTorch 설치, OCR, 대규모 위성영상 모델은 사용하지 않았습니다.

모델은 기존 선을 **분류만** 합니다. 새 선·공백 연결·표고·DEM·사람 승인 정보를 만들지 않습니다.
34개 입력 특징에는 도엽명, 타일 ID, 좌표, 장면 이름이 들어가지 않습니다.
점수는 보정된 등고선 확률이 아니라 검수용 순위 점수입니다.

## 평가 설계와 한계

1. 같은 corner-safe 실행의 8,736개 원본 후보에서 `SHA256(seed:segment_uid)` 순으로
   길이 `[18,28)`, `[28,60)`, `[60,∞)` 구간마다 타일별 6개씩, 총 162개를 선정했습니다.
   seed는 `20260912`이고 선택 화면에는 모델 점수를 표시하지 않았습니다.
2. 원본과 대상 선만 보고 AI 임시 라벨을 확정했습니다. 등고선 106, 문자 9, 도로·하천 22,
   기호 11, 혼합 2, 불명확 12개입니다. 혼합·불명확 14개는 학습·수치 평가에서 제외했습니다.
   기존 실험을 이미 알고 있었으므로 사람의 완전 맹검 연구라고 주장하지 않습니다.
3. 외부 반복에서는 한 도엽을 평가용으로 빼고 다른 두 도엽만으로 학습했습니다.
   평균·표준편차도 학습 자료에서만 계산합니다. 그 두 도엽끼리 다시 한 장씩 제외한
   내부 검증 점수로 임계값을 정합니다. 외부 평가 도엽의 라벨·점수로 임계값을 고르지 않습니다.
4. L2=0.1을 두 특징 집합 모두에 고정했습니다. 첫 선별 정책은 내부 등고선 유지 목표 95%입니다.
   그 결과의 추가 누락을 보고 100% 목표의 보수 정책을 추가했습니다. 이 두 번째 비교는
   **같은 개발 자료를 다시 본 탐색 실험**이며, 독립된 새 시험으로 표현하지 않습니다.

특징 제거 비교도 실행했습니다. 같은 AI 표본으로 **기존 11개 특징만 재학습**하면
선별 정책에서 등고선 92/106 유지, 비등고선 15/42 제외였습니다. 주변 특징을 추가한
34개 모델은 101/106, 25/42였습니다. 보수 정책의 11개 모델은 106/106, 5/42입니다.
표본이 작고 라벨에 오류가 있을 수 있어, 이 비교만으로 각 특징의 인과 효과를 단정하지 않습니다.

실제 9개 타일 출력에도 각 타일이 속한 도엽을 뺀 모델을 사용했습니다.
모든 자료로 학습한 최종 모델·보정 임계값도 실험 JSON에 보존했지만, 그 모델의
독립 평가가 끝난 것은 아니므로 이번 비교 프로젝트에는 사용하지 않았습니다.

이전의 작은 12개 영역도 그대로 다시 확인했습니다. 원본 잉크가 출력선 2픽셀 이내에
포함되는 비율이며, **정식 precision/recall이 아닙니다**.

| 고정 영역 | 기존 합성 필터 | 새 선별 / 보수 |
| --- | ---: | ---: |
| 등고선 6곳: 높을수록 좋음 | 91.11% | 91.17% / 91.17% |
| 문자 3곳: 낮을수록 좋음 | 18.19% | 0% / 0% |
| 도로·하천 3곳: 낮을수록 좋음 | 36.57% | 0.49% / 0.49% |

작은 영역만 보면 결과가 매우 좋아 보이지만, 넓힌 고정 표본과 전체 이미지에서는
잔존 도로·기호와 추가 누락이 드러났습니다. 이 때문에 작은 영역의 개선만으로 승격하지 않습니다.
현재 평가는 **이미 생성된 후보의 분류**입니다. 처음부터 검출되지 않은 희미한 등고선,
문자에 가린 부분, 기존의 잘못된 합류, 전체 지도 연결성은 독립 정답선이 더 필요합니다.

## 산출물과 검수 방법

공통 경로는 `data/derived/contour-context-2026-09-12/`입니다.

- `candidates-final/report.html`: 원본·기존·선별·보수 출력, 162개 개별 표본과 9개 전체 타일.
- `contour-context-final.qgz`: 새 QGIS 비교 프로젝트. 54개 레이어를 저장 후 다시 읽어 검증했습니다.
  선별 비교 그룹에서는 기존/선별/보수를 하나씩 켭니다. **주황색 불확실 선**은 재검수 대상이며
  등고선 승인 표시가 아닙니다. 회색 전체 점수 레이어는 기본 꺼짐 상태로 원래 후보를 보존합니다.
- `candidates-final/*-all-scores.geojson`: 원래 8,736개 선을 모두 보존한 점수 레이어.
  모든 도형이 원래 것과 동일함을 실제 파일을 다시 읽어 확인했습니다.
- 별도 부분집합: 기존 7,940개, 선별 6,785개, 보수 7,804개, 불확실 1,687개.
  불확실은 `(기존 유지 또는 보수 유지) 그리고 선별 제외`입니다. 방법별 레이어는 서로 중복됩니다.
- `experiment-v1/experiment.json`, `experiment-conservative-v2/experiment.json`:
  모델 계수, 내부·외부 학습/평가 ID, 임계값, 표본별 외부 평가 점수.
- `semantic-samples/manifest.json`, `examples/contour_semantic_labels.ai-v1.json`:
  고정 표본과 별도 AI 라벨. 표본을 재생성해 manifest SHA256이 동일함을 확인했습니다.
- [저장된 증거 JSON](evidence/contour-neighborhood-2026-09-12.json): 입력 해시, 방법별 점수,
  모델·임계값, 원본 보존 검증과 한계. 원본 래스터·대용량 파생물은 Git에 넣지 않습니다.

비-QGIS 회귀 테스트 146개, QGIS 3.44.8 통합 테스트 7개가 통과했습니다.
54개 레이어 중 벡터 45개는 읽기 전용입니다. 소스 표본·전체 타일·QGIS 렌더를 눈으로 확인했습니다.
현재 머신에서 9개 타일 점수화와 비교 이미지 생성은 약 9.1초였으며, 별도 성능 벤치마크는 아닙니다.
실험 자체는 플러그인 자동 설치·기존 프로젝트 변경 없이 수행했습니다.
이후 사용자 요청에 따라 코드·기록과 소스에 맞춘 설치용 ZIP을 커밋·푸시 대상으로 정리했습니다.
새 문맥 분류기는 여전히 연구 CLI이며 플러그인의 기본 추출기를 대체하지 않습니다.

## 재실행

기존 개발 타일과 아래 고정 raw vector index가 로컬에 있어야 합니다.
Python 3.12, NumPy, SciPy, Pillow로 실행하며 모든 출력에는 **새 폴더**를 지정합니다.
이번 환경은 Python 3.12.14 / NumPy 2.5.1 / SciPy 1.18.0 / Pillow 12.3.0이었습니다.

```sh
python scripts/prepare_contour_semantic_samples.py \
  data/derived/annotation_package/index.json \
  --vectors data/derived/contour-extraction-2026-09-12/corner-safe-final/ink_candidate_vector_index.json \
  --output data/derived/contour-context-next/samples

python scripts/run_contour_context_experiment.py \
  data/derived/annotation_package/index.json \
  --samples data/derived/contour-context-next/samples/manifest.json \
  --labels examples/contour_semantic_labels.ai-v1.json \
  --legacy-model examples/ink_synthetic_legacy_model.json \
  --target-recall 0.95 --output data/derived/contour-context-next/balanced
```

같은 학습 명령을 `--target-recall 1 --output data/derived/contour-context-next/conservative`로
한 번 더 실행한 뒤, 실제 출력과 비교 화면을 생성합니다.

```sh
python scripts/score_contour_context_candidates.py \
  data/derived/annotation_package/index.json \
  --vectors data/derived/contour-extraction-2026-09-12/corner-safe-final/ink_candidate_vector_index.json \
  --legacy-model examples/ink_synthetic_legacy_model.json \
  --balanced-experiment data/derived/contour-context-next/balanced/experiment.json \
  --conservative-experiment data/derived/contour-context-next/conservative/experiment.json \
  --samples data/derived/contour-context-next/samples/manifest.json \
  --labels examples/contour_semantic_labels.ai-v1.json \
  --probes examples/contour_spotchecks.v1.json \
  --illustration-ids S084 S058 --output data/derived/contour-context-next/candidates

# QGIS의 Python 환경에서 실행
python scripts/build_contour_comparison_project.py \
  data/derived/annotation_package/index.json \
  --after data/derived/contour-extraction-2026-09-12/corner-safe-final/ink_candidate_vector_index.json \
  --context-index data/derived/contour-context-next/candidates/context_candidate_index.json \
  --output data/derived/contour-context-next/comparison.qgz \
  --preview data/derived/contour-context-next/qgis-preview.png
```

다른 그래프 실행·다른 표본·다른 원본에는 이 AI 라벨을 자동 이전하지 않습니다.
다음 승격 조건은 사람의 독립 판독, 저밀도 지형의 도로·둑·하천을 포함한 추가 음성 표본,
도엽 외 검증에서의 누락·잘못된 연결 점검입니다. 공주 최종 홀드아웃은 그 전까지 닫아 둡니다.
