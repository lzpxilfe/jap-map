# Ink v2 중심선 A/B 워크플로

## 적용 범위

이 저장소는 [AI Vectorizer for Archaeology](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology)의 Ink v2 중심선 검출을 별도 A/B 백엔드로 사용합니다. 현재 기준 구현은 커밋 [`f55d45da6228bd0c60e02618a2bb5031a55c54b4`](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/tree/f55d45da6228bd0c60e02618a2bb5031a55c54b4)에 고정되어 있으며, 중심선 외에 source-grid 방향·coherence evidence도 기록합니다.

가져온 부분은 9/15/31 px 다중 스케일 잉크 반응, 타일 정규화, 중심선화와 짧은 가지 제거입니다. QGIS 상호작용용 Live-Wire는 아직 포함하지 않습니다. 두 저장소 모두 GPL-2.0이며, 구현 출처와 고정 커밋은 각 GeoJSON feature와 결과 index에 기록됩니다.

Ink v2는 **등고선 분류기**가 아닙니다. 등고선 연결은 좋아지지만 문자, 도로, 하천, 도곽, 기호도 함께 검출합니다. 따라서 산출물은 `review_only`이며 기존 grayscale ridge 후보와 `contour_annotations.gpkg`의 `proposal_review` 큐를 수정하지 않습니다.

## 현재 계획과 승격 조건

1. 개발 타일에서 Ink 중심선과 기존 ridge 후보를 나란히 비교합니다.
2. 검수자가 대표 타일에서 `contour`, `text`, `road_river`, `unsure` 오류 유형을 확인합니다.
3. 수동 정답선으로 연결성 향상과 오검출 증가를 함께 계량합니다.
4. 이득이 확인된 경우에만 Ink를 제한적 자동 보정 또는 Live-Wire 수동 추적에 연결합니다.
5. 공주 홀드아웃은 설정·임계값을 고정한 뒤 최종 평가에만 사용합니다.

Ink 전체 선화를 등고선으로 승격하지 않고 끝점 사이에서만 사용하는 후속 실험은 [등고선 끊김 복원 연구](contour-completion-research.md)에 정리했습니다.

현재 9개 개발 타일의 정답선 없는 진단에서는 Ink 중심선이 기존 ridge 후보의 약 99.8%를 3 px 이내에서 포함했고, 중심선 픽셀은 평균 약 2.12배였습니다. 이는 연결성 가능성을 보여줄 뿐 precision/recall 또는 정확도 주장이 아닙니다.

## 생성 방법

Python 환경에 `numpy`와 `Pillow`가 필요합니다. `scipy`와 `scikit-image`가 있으면 사용하고, 없으면 NumPy fallback으로 동작합니다. 저장소에 포함된 A/B 산출물은 의존성 차이로 재생성 결과가 달라지지 않도록 함께 버전 관리합니다.

```bash
python -m pip install numpy Pillow
python scripts/generate_ink_centerline_candidates.py \
  data/derived/annotation_package/index.json
```

기본값은 개발 타일 9개만 처리합니다. 결과는 다음 위치에 따로 생성됩니다.

- `data/derived/annotation_package/ink_candidate_vectors_evidence_v2/*-ink-proposals.geojson`
- `data/derived/annotation_package/ink_candidate_vectors_evidence_v2/*-ink-preview.png`
- `data/derived/annotation_package/ink_candidate_vectors_evidence_v2/ink_candidate_vector_index.json`
- `data/derived/annotation_package/ink_candidate_vectors_evidence_v2/ink_candidate_contact_sheet.png`

홀드아웃까지 생성해야 하는 최종 평가 시점에만 명시적으로 실행합니다.

```bash
python scripts/generate_ink_centerline_candidates.py \
  data/derived/annotation_package/index.json \
  --include-holdout
```

## QGIS에서 비교하기

기존 프로젝트의 `Ink v2 vector proposals — A/B review only`은 이전 vector output을 보존한다. v2 evidence output은 Processing의 **Extract Ink Proposals / Ink 선분 추출·검토** 또는 별도 GeoJSON 레이어로 검토하며, 기존 검수 GeoPackage를 자동으로 교체하지 않는다.

```bash
python scripts/create_annotation_qgis_project.py \
  data/derived/annotation_package/index.json
```

기존 cyan `Automatic vector proposals — review only` 그룹과 번갈아 켜서 비교합니다. Ink 레이어는 별도 GeoJSON이므로 분류 상태를 저장하는 `Quick review queue — development only`에는 들어가지 않습니다.

프로젝트 재생성 전에 QGIS를 닫아야 합니다. 스크립트는 기존 GeoPackage가 있으면 새로 만들지 않으므로 이미 저장한 사람의 검수 상태를 유지합니다.
