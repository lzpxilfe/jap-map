# 등고선 강화 검수 파이프라인 실행

`scripts/run_reconstruction_pipeline.py`는 이미 저장된 원본 지도·OCR 근거·승인 기록을
입력으로 관측선 추출 → 숫자 분리 → 짧은 복원 → 숫자 주변 연결 후보 → QGIS 검수 →
연속선 조립 → 원본 근거 평활화 → 의미 참조 대조 → 교차 검증 → 통합 프로젝트를 실행한다.

이 명령은 모델을 내려받거나 학습하지 않는다. 자동 결과를 사람 승인/등고선 정답으로
승격하지 않으며, 홀드아웃 지도도 사용하지 않는다. OCR 자체를 새로 실행하는 명령은 아니다.

## 필요한 입력

- 고정 `drawing-report.json`과 원본 TIFF가 있는 지도 패킷
- 해당 패킷의 고정24개 개발 구역 manifest
- 원본에 대응하는 문자 검출 및 성분별 OCR 보고서
- 기존 승인 연결 GeoJSON과 국소 피드백 적용 패킷
- 명시적 국소 비등고선 판단 GeoJSON
- NumPy/SciPy/Pillow/scikit-image 실행 환경과 QGIS Python 실행기

지도/OCR 근거는 로컬 자료이므로 Git 저장소만 복제하면 자동으로 생기지 않는다.
QGIS 실행기의 심볼릭 링크 경로는 유지해야 한다. macOS 예시는 아래와 같다.

```sh
python scripts/run_reconstruction_pipeline.py \
  --packet data/derived/contour-assisted-2026-09-13/human-review-final \
  --manifest examples/contour_reconstruction_regions.v1.json \
  --detection data/derived/contour-reconstruction-2026-09-14/text-detection-v1/text-detection.json \
  --recognition data/derived/contour-reconstruction-2026-09-14/component-text-recognition-all24-v1/component-text-recognition.json \
  --approved data/derived/contour-vectorized-2026-09-14/v1-portable/approved-connections.geojson \
  --feedback data/derived/contour-reconstruction-2026-09-14/connection-feedback-001 \
  --non-contour-references data/derived/contour-assisted-2026-09-13/human-feedback/chat-034/reviewed-export/non-contour.geojson \
  --qgis-python /Applications/QGIS.app/Contents/MacOS/python \
  --output data/derived/my-new-reconstruction-run
```

`python`에는 위 수치 라이브러리가 설치된 실행기를 사용한다. `--plan-only`를 붙이면
실행 명령만 확인하며 출력 폴더를 만들지 않는다. 출력 경로는 매번 새 폴더여야 한다.

## 성공과 실패

각 단계의 로그와 `단계명-result.json`을 독립적으로 보존한다. 종료 코드가0이어도
필수 산출물이 없으면 중단한다. 실패 뒤 후속 단계는 실행하지 않는다.
성공 시 `run-status.json`의 `completed_review_pipeline`은 실행 완료를 뜻할 뿐
등고선 정확도나 전체 목표 달성을 뜻하지 않는다. 입력/코드가 실행 중 바뀌어도 실패한다.

최종 프로젝트는 `integrated/contour-network-review.qgz`다. 원본·연속선·평활화선을
전환하고 승인선/미승인 복원/의미 검수 선을 구분해서 볼 수 있다. 최종 실행 폴더를
옮길 때는 그 안의 프로젝트 폴더 전체를 함께 이동한다. 원본 입력은 수정하지 않는다.
