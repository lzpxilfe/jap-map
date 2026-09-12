# 여백 OCR 보조 파일럿

PaddleOCR를 **읽기 후보 생성기**로만 시험하는 독립 CLI입니다. QGIS 메뉴에
추가하거나 모델을 기본 설치하지 않습니다. 전체 지도 분할·등고선 추출과도
별도 실험입니다. 좌표·CRS·GCP·등록 JSON·GeoPackage를 변경하는 기능은 없습니다.

2026-09-12에는 Stanford의 공개 원본 한국 20도엽을 직접 확보하고, 격리 환경에서
실제 PP-OCRv5 mobile 모델을 실행했습니다. [원본 목록·실행 기록·한계](margin-ocr-stanford20.md)를
참조하세요. 이것은 **사람의 독립 정답에 따른 20도엽 정확도 검증 완료**를 뜻하지
않습니다. 별도 합성 raster + 가짜 엔진 테스트는 입출력 계약만 검증합니다.
기존 지도 내부 타일의 선분 라벨도 OCR 정답으로 재사용하지 않습니다.

## 1. 원본과 crop 준비

`examples/margin_ocr_pilot.template.json`을 Git에서 제외된
`data/raw/margin-pilot.json`으로 복사해 채웁니다. 원본 경로는 manifest 파일
위치 기준 상대 경로 또는 절대 경로입니다. `scan_source`와 `rights`에는 실제
출처·이용 조건을 기록합니다. 연구자가 직접 도엽 ID와 여백 위치를 정해야 하므로
템플릿의 경로·영역은 비워 두었습니다.

각 `regions` 항목은 다음 구조입니다. 아래 숫자는 **형식 설명용**이며 실제
도엽 좌표가 아닙니다.

```json
{"region_id": "nw-coordinate", "kind": "corner_nw", "pixel_box": [120, 80, 620, 280]}
```

- 종류: `corner_nw`, `corner_ne`, `corner_se`, `corner_sw`, `title`, `scale`, `legend`.
- `corpus_kind`: 실제 스캔은 `historical_scan`, 합성 실행 점검은 `synthetic_smoke`.
  합성 데이터는 20개를 채워도 `synthetic_smoke_only`이며 실측 상태가 되지 않습니다.
- 영역: 원본 저장 raster의 `[left, top, right, bottom]` 정수 픽셀 좌표.
  왼쪽·위는 포함하고 오른쪽·아래는 제외합니다. 경위도나 GCP가 아닙니다.
- 한 crop은 최대 4백만 픽셀, 원본 면적의 25% 이하여야 합니다. 전체 지도와
  범위 밖 영역은 거부합니다. 범례가 크면 의미 단위로 나눕니다.
- 원본은 최대 2억 5천만 픽셀입니다. 대형 JP2를 열 때만 Pillow의 메타데이터
  검사 한도를 잠시 조정하고 복원합니다. 원본 전체를 OCR 입력으로 쓰지는 않습니다.
- 코드가 여백의 의미를 자동 검증하지는 않습니다. 연구자가 내부 지도 영역이
  아닌지 확인해야 합니다. EXIF 자동 회전·확대·축소·선명화는 하지 않으며
  원래 위치의 픽셀을 RGB PNG로 저장합니다. 원본 파일은 그대로 유지합니다.
- `split`은 명시적으로 `development`여야 합니다. 기존 `178-gongju`와 모든
  holdout split은 제외합니다. 같은 파일을 다른 ID로 중복 집계하는 것도 거부합니다.
  재인코딩한 같은 도엽이나 다른 이름으로 바꾼 holdout까지 식별하는 장치는
  아니므로 연구자가 도엽 단위 중복·누출을 확인해야 합니다.

```sh
python scripts/margin_ocr_pilot.py prepare data/raw/margin-pilot.json \
  --output data/derived/margin-ocr/crops-v1
```

이 단계에는 Pillow만 필요하며 PaddleOCR를 불러오지 않습니다.
`crops.html`에서 원본 crop을 보고 **OCR를 보기 전에** 독립 정답을 전사합니다.
읽을 수 없는 글자는 추측으로 채우지 말고 판독 불가로 남깁니다. 원본·crop의
SHA256, 픽셀 위치, 원본 크기, 출처·권리는 `bundle.json`에 기록됩니다.
전체 지도 파일은 OCR 엔진에 전달되지 않습니다.

## 2. 로컬 모델 출처 고정

이번 어댑터의 기준 API는
[PaddleOCR v3.7.0 고정 소스](https://github.com/PaddlePaddle/PaddleOCR/blob/b03f46425e8ff4442b268ce449e3eef758146cd4/paddleocr/_pipelines/ocr.py)입니다.
동작·출력 형식은
[같은 버전의 OCR 문서](https://github.com/PaddlePaddle/PaddleOCR/blob/b03f46425e8ff4442b268ce449e3eef758146cd4/docs/version3.x/pipeline_usage/OCR.en.md)를 기준으로 합니다.
이 API 확인은 옛 활자에 대한 정확도 검증이나 특정 OS의 설치 검증이 아닙니다.

QGIS와 분리된 Python 환경에 `requirements-margin-ocr.txt`의 PaddleOCR·PaddleX·PaddlePaddle을
준비합니다. 저장소의 일반 테스트나 QGIS 설치는 이 의존성을 설치하지 않습니다.
검출·인식용 **Paddle inference 디렉터리**를 연구자가 별도로 확보합니다.
일본어·옛 문자에 적합한 인식 모델인지 검토해야 하며, 이 파일럿은 자동으로
모델을 고르거나 다운로드하지 않습니다. ONNX 변환본 실행은 아직 지원하지 않습니다.
실행한 공식 mobile 모델 두 개의 출처·revision·파일별 사전 SHA256은
`examples/margin_ocr_models.mobile-v5.json`에 보관했습니다. 이는 선택 가능한
실험 명세이며 QGIS의 기본 모델이나 권장 정확도 보증이 아닙니다.

`examples/margin_ocr_models.template.json`을 로컬 `data/raw/margin-models.json`으로
복사해 다음을 채웁니다.

- `directory`: 기존 모델 디렉터리. 상대 경로는 spec 파일 기준입니다.
- `model_name`: 모델 파일에 대응하는 PaddleOCR/PaddleX 모델명.
- `source_url`, `source_revision`: 실제 모델 배포 출처와 고정 버전·커밋 등.
- `weight_license`: 해당 weight의 확인된 이용 조건. 코드 저장소 라이선스만으로
  개별 weight나 변환본의 조건을 대신하지 않습니다.
- `expected_files`(선택): 미리 확인한 파일명→SHA256. 지정하면 pin 단계부터
  파일 목록과 지문이 모두 일치해야 합니다.

```sh
python scripts/margin_ocr_pilot.py pin-models data/raw/margin-models.json \
  --output data/derived/margin-ocr/model-lock-v1.json
python scripts/margin_ocr_pilot.py check-models \
  data/derived/margin-ocr/model-lock-v1.json
```

lock은 로컬 파일 전부의 크기·SHA256과 설치된 PaddleOCR·PaddleX·PaddlePaddle
버전을 기록합니다. 추가·삭제·변경된 파일, symlink, 버전 불일치가 있으면 실행을
거부합니다. 나머지 전이 의존성까지 고정하는 환경 lock은 아니므로 재현 실험에는
별도 환경 명세도 보관하세요. `pin-models`는 **연구자가 제출한 출처 + 로컬 파일
지문**이지 배포자의 진위 확인이나 사용 허가 발급 절차가 아닙니다. 새 파일을
임의로 재고정하기 전에 출처·이용 조건을 다시 확인해야 합니다.

## 3. 읽기 후보 생성과 검수

```sh
python scripts/margin_ocr_pilot.py run \
  data/derived/margin-ocr/crops-v1/bundle.json \
  --model-lock data/derived/margin-ocr/model-lock-v1.json \
  --output data/derived/margin-ocr/run-v1
```

원본과 crop 해시를 다시 확인한 뒤, 실행 폴더에 복사한 crop만 CPU 엔진으로
읽습니다. 문서 회전·펴기·글줄 방향 모델은 끄며, 인식 점수 0 이상을 유지합니다.
Python 소켓 연결은 독립 CLI 프로세스 안에서 차단합니다. 이것은 네이티브 코드
전체를 통제하는 OS 네트워크 sandbox가 아니므로 엄격한 오프라인 환경은 외부에서
별도로 구성해야 합니다. 이 실행부를 QGIS 프로세스 안에서 호출하지 마세요.
PaddleX의 모델 출처 접속 확인을 끄고 캐시를 해당 실행 폴더로 분리합니다.
`--det-max-side 640`처럼 검출기의 내부 최대 변 길이를 고정할 수 있습니다
(128–4096). 원본 PNG는 바뀌지 않으며 정확한 override를 `result.json`에
기록합니다. 생략하면 고정 패키지의 기본값을 사용합니다. 이 설정 하나가 모든
크기의 문자에 최적이라는 보장은 없으며, 시험 후 설정 변경 시 새 실행으로 남깁니다.

실행 결과는 다음과 같습니다.

- `report.html` + `crops/`: 원문 crop, OCR 원문, 글줄별 엔진 신뢰도. 외부 서버나
  스크립트 없이 로컬에서 보며 OCR 문자는 HTML로 실행되지 않도록 이스케이프합니다.
- `result.json`: 원문·폴리곤·점수·소요 시간·오류·입력/모델 해시. 점수는
  보정된 정답 확률이 아니며, 높은 점수도 승인이 되지 않습니다.
- `review.template.json`: 모든 항목이 `unreviewed`인 별도 검수 양식.
- `bundle.json`, `model-lock.json`: 이 실행에 사용한 입력과 모델 명세 사본.

검수 양식은 **새 이름** `review.json`으로 복사한 후 편집합니다. OCR 원문은
수정하지 않습니다. `review_status`는 `unreviewed`, `approved`, `corrected`,
`rejected`, `unreadable` 중 하나입니다. `approved`는 원문과 같은
`reviewed_text`를 명시해야 하고, 변경하면 `corrected`로 기록합니다. 검수자는
`reviewer`, 시간은 선택적으로 `review_seconds`에 남깁니다. 실행 오류는
승인할 수 없으며 별도 `error`로 집계됩니다. 아무 글자도 검출되지 않은
`no_text`와 실행 실패를 혼동하지 않습니다.

독립 전사문을 `reference_text`에 옮기고, OCR를 보고 수정한 정답이 아닌 경우에만
`reference_independent: true`를 표시합니다. 이 표시는 연구자의 확인 기록이며
코드가 독립 전사 여부를 증명하지는 않습니다. 검수 파일은 결과 전체의 해시에
묶여 있으므로 다른 실행 결과와 섞으면 거부합니다.
AI의 잠정 판독이나 도서관 목록 제목은 사람의 독립 전사를 대신하지 않습니다.
이번 실험의 AI 제목 비교는 별도 요약에서만 표시하고 정식 검수 양식은 전부
`unreviewed`, 정식 평가 참조는 비어 있는 상태로 보존했습니다.

어떤 검수 상태도 좌표·CRS·GCP 자동 적용으로 이어지지 않습니다. 승인된 판독을
GIS에 활용하는 단계는 기존 등록 도구에서 연구자가 별도로 수행합니다.

## 4. 한국 도엽 약 20장 평가

도엽 단위로 `clear` 7장, `old_type` 7장, `degraded` 6장을 우선 구성하는
파일럿 계획입니다. 한 도엽이 여러 특성을 가지면 주 시나리오 하나를 고정하고
겹치는 조건은 별도 연구 메모에 남깁니다. 현 공주 holdout은 계속 봉인합니다.
각 도엽에서 네 모서리·도엽명·축척·범례를 같은 선택 원칙으로 crop하고,
인쇄되지 않은 항목과 판독 불가 항목을 구분해 기록합니다. 특정 항목의 결과가
좋다는 이유로 어려운 crop을 평가에서 빼지 않습니다.

```sh
python scripts/margin_ocr_pilot.py evaluate \
  data/derived/margin-ocr/run-v1/result.json \
  --reviews data/derived/margin-ocr/run-v1/review.json \
  --output data/derived/margin-ocr/evaluation-v1.json
```

평가는 수정문이 아닌 **원래 OCR 출력**을 독립 정답과 비교합니다.
원문 완전 일치율, NFC·공백만 정규화한 완전 일치율, 문자 오류율(CER)을
영역 종류·시나리오별로 보고합니다. 도·분·초 기호, 부호, 구두점, 옛 문자는
삭제하거나 숫자로 추정하지 않습니다. CER은 정규화한 참조 문자 수를 분모로
하는 편집 거리이며, 빈 참조만 있으면 `null`입니다. 빈 참조에 대한 실행 오류도
완전 일치에서는 반드시 실패로 집계합니다. CER 하나만으로 성공을 판단하지 마세요.

독립 정답으로 평가한 20개 이상 도엽, 세 시나리오 포함, 전 crop의 참조/판독 불가
처리가 충족될 때만 `pilot_measured`가 됩니다. 시나리오별 7/7/6 구성, 영역별
충분한 표본, 이중 전사의 일치도까지 자동 보증하는 상태는 아닙니다. 미완료는
`pilot_incomplete`로 남기며 판독 불가 수·실행 오류·시간 측정 범위를 별도로
보고합니다. 두 상태 모두 `promotion_passed: false`입니다. 수치 목표를 사전에
정하고 연구자가 영역별 오독·좌표 기호 오류·검수 시간까지 검토한 후 도입을
판정해야 합니다. 20장은 초기 파일럿이지 일반화 성능의 확정 표본이 아닙니다.

모든 출력은 새 파일·새 폴더만 허용합니다. 준비 중 중단되어 `bundle.json`이
없거나 실행 중 중단되어 `result.json`이 없으면 그 폴더는 미완성입니다. 기존
검수 파일을 덮어쓰지 말고 원인을 해결한 후 새 출력 위치로 재실행하세요.

## 범위와 후속 우선순위

여백 판독 파일럿은 [초기 3종 도엽 기준선](pilot-protocol.md)의 등고선 segmentation
훈련 도입 기준을 완화하지 않습니다. 기존 Ink·합성 분류 실험도 실제 도엽의
수동 평가를 대체하지 않습니다.

AI-Vectorizer-for-Archaeology의 EfficientSAM은 유지하며, EfficientViT-SAM은
같은 crop·클릭 prompt에서 실제 자료 30–50개를 비교하는 challenger adapter
설계 대상으로만 둡니다. 이번 jap-map 변경은 그 저장소나 모델을 수정·설치하지
않습니다. Prithvi-EO-2.0·TerraMind·OlmoEarth·GraphCast 등 대형 지구관측 모델은
이 파일럿에 추가하지 않습니다.

회귀 검증: `python -m unittest -v tests.test_margin_ocr`.
테스트의 가짜 OCR 출력·모델 파일·20도엽 레코드는 계약 검증용이며 실제 자료나
실제 모델 실행 결과로 보고해서는 안 됩니다.
