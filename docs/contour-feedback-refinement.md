# 피드백 기반 연결 형상 강화 — 2026-09-14

사례별 수동 수정에서 반복된 문제를 공통 생성 코드로 옮겼습니다. 대상은 **짧은 연결의
형상·접속점·검수 안전성**입니다. 등고선/하천/문자 분류기나 전체 지도 추출기를 새로 학습한 것은 아닙니다.
원래 8,105개 선, 444개 제안, 사람 승인 도형과 S/E 자료는 보존했습니다.

## 반영한 규칙

- **잔굴곡 정리:** 16픽셀 이하의 전진하는 연결에서 작은 S자 굴곡만 단일 곡률의 곡선으로
  정리합니다. 양끝 좌표는 정확히 고정하며 저장 폴리라인의 최대 이동을
  `min(0.35px, 공백 길이 × 0.05)`로 제한합니다. 실제 큰 굴곡과 긴 숫자 공백은 펴지 않습니다.
- **접속점 재검토:** 두 원선 안쪽의 2–12픽셀 구간에서 획에 수직인 밝기 단면을 읽고,
  국소 대비로 옅은 잉크도 평가합니다. 공백 안의 숫자 획을 지지 표본으로 읽지 않습니다.
  두 강한 평행 획, 지지 부족, 불안정한 직선/이차곡선 적합, 관찰 범위에 따른 방향 불일치는 보류합니다.
- **국소 원선 교체:** 안정된 방향이 기존 끝점과 어긋날 때 원선 끝부분을 최대 8픽셀씩
  재검토하고, 실제 원선 위의 잘라낼 지점에 접선이 연속인 새 경로를 붙입니다.
  원선 UID·도형 해시·잘라낼 길이·정확한 접속점을 계약으로 저장합니다. **미승인 수정안이며
  단순 추가용 선이 아닙니다.** 승인 뒤에도 별도 작업 사본에서 두 끝부분을 교체해야 합니다.
- **곡부와 연결 상대 보존:** 기존 곡부의 반대쪽으로 선을 미는 국소 적합이나 경쟁하는
  끝점 쌍은 문맥 검수로 넘깁니다. 손떨림 좌표를 정답처럼 추종하지 않습니다.
- **보수적 교차 검사:** 끝점을 포함한 전체 수정 경로의 1픽셀 주변에서 제3의 원선,
  다른 기존 제안, 승인 도형, 다른 새 수정안과의 접촉을 검사합니다. 같은 원선을 이미
  검수한 경우 국소 교체는 합쳐진 계약을 다시 검수해야 합니다.
- **표시 품질:** 원본 지도는 픽셀값·가로세로 비율·픽셀 중심 좌표를 유지하고,
  벡터 표시만 4배 supersampling으로 매끄럽게 그립니다. 예전 스크린샷 정합용 렌더러는 그대로 둡니다.

`histcontour_core/gap_refinement.py`가 공통 계산을 맡습니다. QGIS **Ink Trace → G**와
새 `generate_assisted_contour_drawing.py` 실행에는 끝점을 고정하는 정리만 적용됩니다.
QGIS에서는 Enter 전까지 미리보기이며, 실제 접속점 이동은 별도 검수 CLI에서만 제안합니다.
QGIS 설치 환경은 자동 교체하지 않았고 플러그인 ZIP을 다시 빌드했습니다.

## 개발 타일 실행 결과

최신 입력은 `human-feedback/chat-034/portable-ledger/`의 검수 기록과 승인 수정선입니다.
응답 35건과 주변 문맥 보류 11건을 잠그고, 나머지 398건을 검사했습니다.

| 처리 | 개수 |
| --- | ---: |
| 새 미승인 수정안 | 115 |
| └ 끝점 고정 잔굴곡 정리 | 57 |
| └ 국소 원선 끝부분 교체가 필요한 수정 | 58 |
| 원래 형태 유지 | 42 |
| 문맥 검수 필요 | 241 |

115개 중 28개는 잔굴곡 정리만 제안할 수 있고 접속점 추정의 근거는 불충분합니다.
그 경고를 숨기지 않으며 검수 순서에서 뒤에 둡니다. 241개 중 193개는 경쟁 끝점 쌍이고,
23개는 기존 곡부 방향과 국소 적합이 충돌했습니다. 이 수치는 정확도나 승인율이 아닙니다.
검수 큐는 근거 수준·수정 종류 안에서 9개 타일을 번갈아 보여 한 타일에 편중되지 않게 했습니다.

로컬 결과:

- `data/derived/contour-feedback-refinement-2026-09-14/v1-verified/START-HERE.md`
- 같은 폴더의 `proposed-revisions.geojson`, `refinement-report.json`, `review-queue.json`, 비교판 9장.
- `data/derived/contour-feedback-refinement-2026-09-14/v1-verified-evaluation/development-evaluation.json`

### 이미 받은 피드백으로 하는 개발 회귀 점검

명확히 승인된 수정선 4건과 새 알고리즘 출력을 **원래 공백의 공통 구간**에서 비교했습니다.
원선을 더 길게 다룬 수정안에 끝점 길이 차이로 벌점을 주지 않고, 원래 연결축에 수직인 경로 차이를 측정합니다.

| 사례 | 수정 전 평균 차이(px) | 새 계산 평균 차이(px) |
| --- | ---: | ---: |
| A0011 | 0.423 | 0.169 |
| A0038 | 0.584 | 0.076 |
| A0006 | 0.057 | 0.003 |
| A0034 | 0.529 | 0.229 |
| 4건 평균 | 0.398 | 0.119 |

“좀 과하지만 수용 가능”했던 A0013-R1은 이상적 곡선으로 간주하지 않고 별도 기록했습니다.
첫 개발안은 이 사례를 반대쪽으로 밀어 기준선과 더 멀어졌습니다. 일반적인 곡부 반전·경쟁 쌍
보류 규칙을 추가했고, 최종 계산에서는 자동 수정을 보류합니다. **이미 승인된 A0013-R1을
원래 기계 선으로 되돌렸다는 뜻이 아닙니다.** 실제 사용 도형은 승인본 그대로 유지합니다.

같은 피드백이 구현에 영향을 주었으므로 위 4건은 독립 시험이나 일반화 정확도의 근거가 아닙니다.
수정된 새 미검수 115건에는 아직 사람 판단이 없습니다. 기존 35건 재질문·승인 도형 변경은 0건입니다.

## 재현

NumPy, SciPy, Pillow가 필요합니다. 실행 시 출력 폴더는 아직 없어야 합니다.
원본/기준선/검수 기록/승인 수정선/타일의 해시와 원본 CRS를 확인하며, 홀드아웃 입력을 거절합니다.

```sh
python scripts/refine_assisted_drawing_from_feedback.py \
  data/derived/contour-assisted-2026-09-13/human-review-final \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/review-ledger.json \
  --revisions data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/approved-revisions.geojson \
  --output data/derived/feedback-refinement-replay --previews 9

python scripts/evaluate_feedback_refinement.py \
  data/derived/contour-assisted-2026-09-13/human-review-final \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/review-ledger.json \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/approved-revisions.geojson \
  data/derived/feedback-refinement-replay \
  --output data/derived/feedback-refinement-evaluation-replay
```

검수 기록은 **잠금과 개발 회귀 비교**에만 사용하며, 새 분류 학습이나 임계값 자동 최적화에 넣지 않습니다.
문자 공백에 사라진 여러 선의 연결 순서, 희미한 선의 전체 복원, 하천·도로와 등고선 구별은
여전히 사람이 판단할 부분입니다. 흰 공백·물음표·추정 의견을 배경 정답이나 확정 라벨로 바꾸지 않습니다.

검증 범위와 파일 해시는 [실행 증거](evidence/contour-feedback-refinement-2026-09-14.json)에 기록합니다.

로컬 비-QGIS 회귀 **232개**, QGIS 3.44.8 통합 **11개**를 통과했습니다. 실제 새 GeoJSON의
115개 도형을 QGIS에서 열어 원본 CRS·도형 유효성·미승인 상태를 확인했고, 국소 교체 58건의
계약과 끝점 고정 57건을 다시 검증했습니다. 9개 타일의 비교판과 빌드한 ZIP의 소스 일치도 확인했습니다.
