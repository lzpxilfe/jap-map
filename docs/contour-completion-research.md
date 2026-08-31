# 등고선 끊김 복원 연구

## 결론

검은 픽셀이나 Ink 중심선만으로는 등고선, 글자, 도로, 하천, 기호를 구분할 수 없습니다. 따라서 자동 연결은 다음 두 단계를 분리합니다.

1. **의미 판정:** 사람이 `contour`로 검수했거나 향후 contour segmentation 모델이 판정한 선만 anchor로 사용합니다.
2. **기하 복원:** 선택된 두 끝점의 거리·접선·곡률·Ink 지지도를 이용해 끊어진 구간만 제안합니다.

기본 명령은 1단계의 사람 검수를 요구합니다. `--anchor-status all`은 분류되지 않은 ridge 선 전체를 사용하는 연구 모드이며 결과를 자동 적용하면 안 됩니다.

## 실패 유형 매트릭스

| 상황 | 관측 문제 | 현재 처리 | 안전장치 |
|---|---|---|---|
| 진한 선이 옅어짐 | ridge는 끊기지만 약한 Ink 반응은 남음 | 좁은 Hermite corridor 안에서 Ink 최단경로 탐색 | 40 px 이하, 양 끝 접선 오차 35° 이하, 경로 신장률 1.65 이하 |
| 글자가 선과 겹침 | 글자 획을 따라가거나 분기 발생 | Ink 분기 경로를 거부하고 접선 기반 amodal 보간 | 긴 anchor 60 px 이상, 접선 오차 18° 이하 |
| 긴 글자가 선을 가림 | 중간 Ink가 사라져 최단경로가 없음 | 최대 72 px까지 낮은 점수의 cubic Hermite 후보 | 자동 적용 금지, 한 끝점당 하나의 비모호 후보만 허용 |
| 기호가 선과 겹침 | 원·사각형·절벽기호 외곽으로 우회 | 기호 perimeter를 따라가지 않고 원래 접선으로 통과 | 교차·분기 주변은 `hermite_occlusion`으로 별도 표시 |
| 도로·하천과 교차 | 가까운 검은 선으로 잘못 연결 | 서로 마주보는 접선이 아니면 즉시 거부 | 수직/역방향 접선, 과도한 곡률과 우회 경로 거부 |
| 인접 평행 등고선 | 옆 등고선 끝점이 더 가까울 수 있음 | 거리뿐 아니라 양 끝의 방향 polarity를 함께 사용 | 점수가 비슷한 목적지가 둘이면 둘 다 `ambiguous_endpoint`로 거부 |
| 이미 이어진 선 | skeleton 분해 때문에 가짜 끝점 발생 | 기존 ridge vector를 rasterize해 실제 결손 run 측정 | 3 px 이상의 새 구간과 20% 이상의 미피복 구간 요구 |
| 짧은 글자 획 | 글자 획 두 개가 마주보는 것처럼 보임 | 40 px 미만 source line은 anchor가 될 수 없음 | 보간은 더 엄격하게 60 px 이상 요구 |
| 타일 경계 | 다음 타일에 이어질 정상 등고선일 수 있음 | 현재 타일 안에서 연결하지 않음 | 12 px 경계 buffer의 끝점 제외 |

이 행들과 닫힌 기호, 자기연결, 최대거리 초과, 완만한 곡선, 과도한 우회를 포함한 총 15개 사례가 `tests/test_contour_completion.py`에 양성·음성 테스트로 고정되어 있습니다.

## 알고리즘

짧은 가시 연결에는 Live-Wire와 같은 weighted graph 관점을 사용하되, 전체 이미지가 아니라 두 끝점 사이의 좁은 곡선 corridor만 탐색합니다. [Intelligent Scissors 연구](https://doi.org/10.1006/gmip.1998.0480)는 경계 추적을 weighted graph의 최적경로 문제로 구성합니다.

글자나 기호가 실제 선을 가린 경우에는 보이는 잉크를 따라가는 대신 두 끝점의 위치와 접선을 이용한 cubic Hermite curve를 만듭니다. 이는 끝점 거리와 gradient direction으로 짝을 찾고 Hermite spline으로 등고선을 복원하는 [Subedi 등의 접근](https://arxiv.org/abs/2412.15515), 가려진 구간에서 방향·곡률 연속성과 불필요한 inflection 최소화를 요구하는 [Takeichi 등의 amodal completion 연구](https://doi.org/10.1068/p240373)와 같은 원칙입니다.

반면 의미 분류는 별도 문제입니다. 고전적인 흑백 지형도 연구에서도 connected component를 text·line art·icon으로 나누더라도 겹침이 남았고, black layer 전체 vectorization이 핵심 난제로 보고되었습니다([Li et al., 1999](https://digitalcommons.unl.edu/cseconfwork/32/)). 최근 contour vectorization 연구 역시 단순 threshold는 넓은 문맥을 보지 못해 false positive/negative가 많으며, U-Net 같은 segmentation이 annotation과 도엽 차이에 더 강하다고 보고합니다([Vynikal & Pacina, 2025](https://doi.org/10.3390/ijgi14050201)).

## 실제 개발 타일 진단

분류되지 않은 ridge 선을 명시적으로 사용한 `--anchor-status all` 연구 결과는 다음과 같습니다.

- 개발 타일 9개만 처리, 공주 holdout 제외
- Ink 전체 제안 3,271개 대신 끝점 연결 후보 41개
- `ink_path` 21개, `hermite_occlusion` 20개
- 지명·도로 장면에서 29개, 산지 장면에서 5개, 수계·평야 장면에서 7개

연결 수는 크게 줄었지만 지명·도로 장면의 비율이 여전히 높습니다. 즉 거리·접선·곡률만으로는 contour semantic precision을 보장하지 못합니다. 이 산출물은 실패 유형을 찾기 위한 연구 레이어이며 accuracy/precision/recall 주장이 아닙니다.

## 실행

안전한 기본 실행은 사람이 `contour`로 검수한 기존 제안만 anchor로 사용합니다.

```bash
python scripts/generate_contour_completion_candidates.py \
  data/derived/annotation_package/index.json
```

분류 전 linework 전체에 대한 실패 유형 연구는 다음처럼 명시적으로 실행합니다.

```bash
python scripts/generate_contour_completion_candidates.py \
  data/derived/annotation_package/index.json \
  --anchor-status all
```

공주 holdout은 두 명령 모두 기본 제외됩니다. 최종 설정을 고정한 평가 시점에만 `--include-holdout`을 추가합니다.

QGIS의 `Contour completion proposals — research only` 그룹은 기본으로 꺼져 있습니다.

- 초록: `ink_path`
- 주황: `hermite_occlusion`
- 보라: `hermite_gap`

어떤 mode도 자동 등고선이 아니며 기존 `proposal_review` 큐나 GeoPackage를 수정하지 않습니다.

## 다음 학습 단계

1. 기존 ridge 검수 큐에서 `contour`, `text`, `road_river`, `unsure`를 축적합니다.
2. 타일 단위가 아니라 도엽 단위로 train/validation/test를 분리합니다.
3. 밝기·대비·blur·압축과 함께 이 문서의 가림 유형을 합성 augmentation으로 추가합니다.
4. contour probability와 topology-aware loss를 비교하되 dangling end 감소와 잘못된 intersection 증가를 함께 측정합니다.
5. 고정된 모델 결과에만 이 completion backend를 적용하고 마지막으로 공주 holdout을 한 번 평가합니다.
