# Ink 선분 학습 워크플로

## 왜 이 순서인가

Ink v2는 끊어진 가는 선까지 잘 보존하지만, 검은색 글자·도로·하천·기호도
함께 중심선화한다. 따라서 Ink 자체를 약화시키거나 끝점 연결부터 시도하지
않고 다음 순서를 사용한다.

`Ink 중심선 보존 → 문맥을 본 선분 분류 → 등고선 후보만 결합 → 마지막 끊김 복원`

현재 단계는 두 번째 단계의 정답 자료를 만드는 과정이다. 자동으로 등고선
라벨을 붙이지 않으며, 정답이 쌓이기 전에는 정확도 수치를 출력하지 않는다.

## 준비된 검토 큐

개발 도엽 3장(부여·청양·논산)의 9개 타일에서 타일마다 40개씩, 총 360개
Ink 선분을 골랐다. 길이·직선성·회전량·주변 잉크 밀도가 서로 다른 선분을
다양성 표본 추출하여 산지 등고선만 과대표집되지 않게 했다.

공주 3개 타일은 생성, 표본 추출, 모델 튜닝에서 모두 제외되어 있다. 모델
평가는 타일 무작위 분할이 아니라 부여·청양·논산 중 도엽 하나 전체를 번갈아
빼는 leave-one-sheet-out 방식으로만 수행한다.

## QGIS에서 라벨 붙이기

1. `annotation_project.qgz`를 연다.
2. **Annotation layers → Ink segment learning queue — label this**를 선택한다.
3. 선분을 선택하고 아래 단축키를 누른다.

- `Ctrl+1`: 등고선 (`contour`)
- `Ctrl+2`: 글자 (`text`)
- `Ctrl+3`: 도로 또는 하천 (`road_river`)
- `Ctrl+4`: 지도 기호 (`symbol`)
- `Ctrl+0`: 판단 보류 (`unsure`)

한 번에 명확히 같은 종류인 선분 여러 개를 선택해도 된다. 애매하면 억지로
이진 라벨을 붙이지 말고 `unsure`를 사용한다. 각 타일의 전체 표본은
`data/derived/annotation_package/ink_segment_review/*-ink-review-contact.png`에서
빨간색으로 미리 볼 수 있다.

## 학습 준비 상태 확인

QGIS를 닫은 뒤 저장소 루트에서 실행한다.

```bash
python3 scripts/train_ink_segment_classifier.py \
  data/derived/annotation_package/contour_annotations.gpkg
```

현재는 360개가 모두 `unreviewed`이므로 결과가 `waiting_for_labels`로 나오는
것이 정상이다. 최소한 등고선 20개와 비등고선 20개가 세 개발 도엽에 걸쳐
모여야 기준 모델을 학습한다. `text`, `road_river`, `symbol`은 비등고선으로
학습하지만 오류 분석에서는 원래 범주를 보존한다.

학습 결과는 `ink_segment_baseline.json`에 저장된다. 이 모델은 NumPy나
scikit-learn 없이 실행되는 작은 로지스틱 기준선이며, 결과를 자동으로
`contour_gt`에 넣지 않는다. 검증된 뒤 다음 단계에서 영상 문맥 패치 기반
분할 모델과 비교할 기준으로 사용한다.

## 재생성

Ink 결과가 바뀌었을 때만 아래 명령으로 표본 원본을 다시 만든다. 이미
GeoPackage에 라벨을 붙이기 시작했다면 먼저 백업하고, 기존
`ink_segment_review` 레이어를 덮어쓰지 않는다.

```bash
python3 scripts/prepare_ink_segment_review.py \
  data/derived/annotation_package/index.json
```
