# 사람 검수 결과를 별도 GIS 추가선으로 내보내기

이 단계는 모델을 다시 학습하거나 기존 선망을 덮어쓰는 작업이 아닙니다.
원래 444개 AI 제안에 대한 대화 판단을 검증하고, 의미와 연결이 모두 승인된 추가선만 분리합니다.

2026-09-13 현재 15개 원래 사례에 응답을 받았습니다.

| 구분 | 개수 | 출력 파일 |
| --- | ---: | --- |
| 연결·등고선 의미 승인 | 4 | `approved-contour.geojson` |
| 연결은 승인, 의미는 추정·불확실·별도 언급 없음 | 4 | `semantics-pending.geojson` |
| 추적은 맞지만 등고선 아님 | 1 | `non-contour.geojson` |
| 기존 연결 거절 | 3 | `rejected-route.geojson` |
| 개별 승인·거절 미정 | 3 | `unresolved-route.geojson` |

승인된 등고선 추가선은 A0179, A0048, A0041, A0013-R1입니다.
A0013-R1은 “굽힘은 선호보다 크지만 수용 가능”이라는 평가를 함께 보존합니다.
기존 A0013의 도형 대신 승인된 R1을 사용하며, 원래 파일은 변경하지 않습니다.
나머지 429개는 미검수입니다. 위 개수는 정확도나 전체 지도 완성률이 아닙니다.

## 공개 기록과 로컬 자료

Git에는 다음 소규모 파일만 들어갑니다.

- [판정 기록](../examples/assisted-human-review-2026-09-13/review-ledger.json): 현재 판단, 발언 근거, A0512 정정 이력, 보류 상태.
- [승인된 수정선](../examples/assisted-human-review-2026-09-13/approved-revisions.geojson): A0013-R1의 실제 좌표.

스크린샷, 원본 지도, 모델 가중치, 개인 파일 경로, 실제 판독자의 확인되지 않은 신원은 공개하지 않습니다.
원본·주석 그림은 로컬 `user_drawing/` 및 `data/derived/`에 보존합니다.
공개 변환기는 허용된 필드만 선택하며, 개인 경로가 발언에 섞여 있으면 공개를 중단합니다.
발언 기록을 검증하는 것이지 실제 판독자의 신원을 증명하는 기능은 아닙니다.

## 재현

기존 최종 패킷의 `drawing-report.json`과 `ai-proposals.geojson`이 필요합니다.
내보내기 자체는 원본 래스터·모델 다운로드나 QGIS 실행이 필요하지 않습니다.
파일 해시는 공개 기록과 일치해야 하며 모든 출력 폴더는 새 폴더여야 합니다.

```sh
python scripts/export_assisted_human_review.py \
  data/derived/contour-assisted-2026-09-13/human-review-final \
  examples/assisted-human-review-2026-09-13/review-ledger.json \
  --revisions examples/assisted-human-review-2026-09-13/approved-revisions.geojson \
  --next-id A0059 \
  --output data/derived/assisted-review-export-replay
```

같은 명령으로 검증한 로컬 최종 결과는
`data/derived/contour-assisted-2026-09-13/reviewed-export-v1/`에 있습니다.
`approved-contour.geojson`을 QGIS에 별도 추가 레이어로 열 수 있습니다.
`reviewed-proposals.geojson`은 15개 사례와 판정 구분을 함께 담고,
`remaining-review-queue.json`은 이미 응답한 사례와 보류된 주변 사례를 구분합니다.
초기 화면으로 준비한 다음 한 건은 A0059이며 자동 승인하지 않습니다.

로컬 대화 기록에서 공개 기록을 다시 만들 때는 다음을 사용합니다.

```sh
python scripts/publish_assisted_review_ledger.py \
  data/derived/contour-assisted-2026-09-13/human-review-final \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-008/feedback.json \
  --session data/derived/contour-assisted-2026-09-13/human-feedback/chat-008/session.json \
  --output data/derived/assisted-public-ledger-replay
```

## 검증 계약

- 연결 승인만으로 등고선 승인을 만들지 않습니다. A0137은 선 추적이 맞아도 등고선에서 제외합니다.
- `?`는 선이나 비등고선·배경 정답이 아니라 불확실성 주석입니다. 표시 없는 획도 자동 승인하지 않습니다.
- A0512의 철회된 임시 승인은 현재 승인 근거로 사용할 수 없습니다.
- 수정선에는 해당 revision ID에 대한 명시적 승인, 정확한 도형 해시, 동일 끝점·CRS가 필요합니다.
- 승인된 곡선도 공백을 추론한 선이면 관측 잉크가 아닙니다. 표고를 임의 부여하지 않습니다.
- 사례 판정을 해당 끝점이 속한 전체 원본 선으로 전파하지 않습니다.
- 기존 검수 GPKG, 동결 분류기, S/E 자료 및 공주 홀드아웃은 변경하지 않습니다.
- 자동 선별 단계는 사람이 승인한 패킷이나 홀드아웃을 입력받으면 중단합니다.
- 생성·기하 선별·AI 시각 선별의 모든 GeoJSON에 원본 CRS를 명시합니다.

## 더 작은 곡률 수정부터 제안하기

A0013-R1을 임의로 다시 펴지 않습니다. 앞으로 별도로 요청받는 곡률 수정에는
`--subtle-side positive` 또는 `--subtle-side negative`를 사용할 수 있습니다.
방향은 주변 흐름을 보고 명시적으로 고르며, 초기 크기는 0.5 원본 픽셀과 공백 길이의 5% 중 작은 값입니다.
이 값은 사용자 피드백 뒤 선택한 보수적인 초안 정책이지 학습된 수치나 사용자 지정 수치가 아닙니다.
정확한 수치가 필요하면 기존 `--offset-pixels`도 사용할 수 있습니다. 두 방식은 동시에 지정할 수 없습니다.

좌표계·승인/의미 분리·철회·불확실성·수정선 변조·비공개 경로·파일 보존을 회귀 검사합니다.
최신 검증 기록은 [후속 증거](evidence/contour-assisted-human-review-2026-09-13.json)에 있습니다.
