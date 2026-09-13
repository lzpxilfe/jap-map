# 강화된 연결을 적용한 벡터 선망 — 2026-09-14

앞서 사용한 **9개 개발 타일**에 피드백 기반 연결을 실제 적용하고, 맞닿은 선분을 하나의
LineString으로 합쳤습니다. 전체 도엽을 새로 추출한 것은 아니며, 기존 잉크 벡터와 동결된
분류기의 보수 후보를 사용했습니다. 원본 패킷·사람 판단·승인 도형·공주 홀드아웃은 그대로입니다.

로컬 전달 폴더는 `data/derived/contour-vectorized-2026-09-14/v1-portable/`입니다.

## 결과

| 항목 | 개수 |
| --- | ---: |
| 원래 선분 | 8,105 |
| 기존의 같은 끝점만 합친 선 | 7,364 |
| 승인 연결 23개만 적용한 선망 | 7,341 |
| 강화된 자동 연결까지 적용한 벡터 후보 | 7,212 |
| 사람 승인 연결 | 23 |
| 미승인 자동 연결 | 129 |
| └ 강화 수정안 / 원래 형태가 지지된 연결 | 87 / 42 |
| 원선 끝부분 교체 | 122개 원선 (승인 6 + 자동 시나리오 116) |
| 합치지 않은 미검수 연결 | 280 |
| 열린 끝점: 전 → 후 | 14,220 → 13,916 |

선 개수의 감소에는 **기존 접점 741개를 정리한 효과**와 새 연결 152개의 효과가 함께 들어갑니다.
이를 정확도나 새로 발견한 등고선 수로 표현하지 않습니다. 새 수정안 115개 중 근거가 약한
28개는 합치지 않았고, 문맥 검수 241건·주변 스케치 보류 11건도 미적용으로 남겼습니다.
연결 승인과 등고선 의미가 모두 확인되지 않은 사람 판단은 자동으로 연결선에 편입하지 않았습니다.

## 파일과 QGIS 표시

- `contour-vectorization.qgz`: 상대 경로를 쓰는 QGIS 프로젝트. 레이어 16개와 타일 북마크 9개.
- `contour-vectorization.gpkg`: 실제 벡터 레이어 6개. 후보 선망, 승인 연결만 적용한 후보 선망,
  원래 선분, 승인 연결, 자동 연결, 합치지 않은 검수 연결을 분리합니다.
- `contour-candidates.geojson`: 원본 CRS **EPSG:5132**를 명시한 7,212개 벡터 후보.
  일반 웹 GeoJSON처럼 WGS84로 해석하면 안 됩니다.
- `approved-connections.geojson`: 승인된 정확한 연결 23개. 기존 승인본과 같은 도형입니다.
- `assembled-parts.geojson`, `source-tail-replacements.geojson`, `network-membership.json`:
  어떤 원선 끝부분을 교체하고 어떤 연결을 합쳤는지 추적하는 구성 요소와 대응표.
  이미 완성된 후보 선망에 이 구성 요소를 다시 덧붙이지 마세요.
- `needs-review.geojson`: 합치지 않은 280개 원래 연결 제안과 보류 이유.
- `previews/`: 타일별 원본 / 지도 위 벡터 / 벡터만 비교판 9장.

QGIS에서 파랑은 전체 벡터 **후보**, 초록은 승인된 연결 구간만, 주황은 미승인 자동 연결입니다.
‘선망 비교’ 그룹에서 적용 전·승인 연결만·강화 후를 전환합니다. ‘원본 지도’ 그룹을 끄면
벡터만 보이며, 프로젝트 북마크로 타일을 바꿀 수 있습니다. 읽기 전용 전달본이므로 후속 편집은 별도 사본에서 합니다.

## 합치기 계약

1. 최신 `chat-034` 기록과 입력 파일·도형 해시를 검증합니다.
2. 승인된 국소 수정은 지정된 원선 끝부분을 교체한 뒤 연결합니다. 자동 수정은 별도의 미승인
   시나리오에만 같은 계약을 적용하며, 승인 연결만 적용한 대안도 함께 저장합니다.
3. 자동 연결은 끝점이 아직 실제 원선에 있는지, 전체 경로가 제3의 선·승인선·다른 새 연결에
   닿는지 다시 확인합니다. 약한 근거·경쟁 연결·기존 판단·보류 상태를 승인으로 바꾸지 않습니다.
4. **차수 2인 기존 끝점에서만** 합칩니다. 분기점에서 어느 선으로 갈지 임의 선택하지 않으며
   서로 다른 타일도 합치지 않습니다. 허용 오차는 좌표 직렬화 오차 수준의 0.00001 원본 픽셀입니다.
   승인선을 포함한 각 구성 요소의 모든 좌표를 보존하고 방향이 바뀐 경우 대응표에 기록합니다.
5. 합쳐진 전체 선에는 `human_approved=false`, `whole_line_semantics_approved=false`,
   `training_eligible=false`를 유지합니다. 한 연결의 승인이 원선 전체에 전파되지 않습니다.

## 검증과 남은 한계

비-QGIS 테스트 **242개**, QGIS 통합 테스트 **12개**를 통과했습니다. GeoPackage 6개 레이어의
**186,531개 좌표**를 저장 전 GeoJSON과 다시 대조하고 도형 유효성을 확인했습니다.
QGIS 프로젝트 재열기, 상대 경로, 별도 폴더로 옮기는 합성 통합 검사도 수행했습니다.

희미한 선의 누락, 숫자 공백의 큰 생략, 하천·도로·문자의 혼입은 여전히 보입니다. 이번 출력은
등고선 정답이나 지도 완성본이 아니며 표고를 부여하지 않았습니다. 미승인 연결을 별도 작업
사본에 적용한 것이지 사용자가 그 도형을 승인했다고 기록한 것은 아닙니다.

## 재현

첫 단계는 NumPy·SciPy·Pillow가 있는 Python, 두 번째는 QGIS Python에서 실행합니다.
출력 폴더는 새 폴더여야 합니다. QGIS의 현재 열린 프로젝트나 설치 환경은 건드리지 않습니다.

```sh
python scripts/vectorize_assisted_contours.py \
  data/derived/contour-assisted-2026-09-13/human-review-final \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/review-ledger.json \
  data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/portable-ledger/approved-revisions.geojson \
  data/derived/contour-feedback-refinement-2026-09-14/v1-verified \
  --output data/derived/contour-vectorization-replay

# QGIS Python에서:
python scripts/build_feedback_vectorization_project.py data/derived/contour-vectorization-replay
```

검증 기록: [실행 증거](evidence/contour-vectorization-feedback-2026-09-14.json).
