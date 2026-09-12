# 한국 20도엽 여백 OCR 실제 실행 기록

실행일: 2026-09-12. 이 기록은 **실제 원본 확보·로컬 추론의 검증**이며,
사람의 독립 전사에 따른 정확도 검증이나 QGIS 기본 기능 승인이 아닙니다.
모든 OCR 결과는 검수 전 후보이고 좌표·CRS·GCP 적용은 0건입니다.

최종 로컬 실행: **20도엽·140 crop·실행 오류 0건**, CPU crop 추론 합계 72.35초.
원문 이미지·OCR 원문·글줄 점수·모델 지문을 보존했습니다. 회귀 테스트 119개와
격리 환경의 67개 패키지 호환성 점검도 통과했습니다. QGIS 통합 테스트나 배포는
이번 독립 CLI 작업에서 새로 실행하지 않았습니다.

## 원본과 권리

[Stanford의 일본제 한국 1:50,000 지형도 색인](https://earthworks.stanford.edu/catalog/stanford-ng525ny5879)에서
20개 지역군의 원본 JP2를 내려받았습니다. 이 코퍼스에는 일본어·한자 옛 활자와
우횡서 제목, 회전된 경위도, 흐린 인쇄·누런 종이·접힌 자국이 포함됩니다.
일부는 일본어 도식을 보존한 재인쇄본입니다. 현대 한국어 OCR 표본이 아닙니다.

- 원본: 20개, 합계 **476,759,502 bytes**. 미리보기와 모델·환경 용량은 제외합니다.
- 각 원본의 IIIF manifest에 표시된 이용 조건:
  [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/).
  PDM 표시는 소장기관의 공공영역 식별이며 새 저작권 허가서가 아닙니다.
- 원본 URL, 소장기관 페이지, 원본·manifest SHA256과 크기는
  [최종 실행 증거 JSON](evidence/margin-ocr-stanford20-2026-09-12-v2.json)에 보관했습니다.
- Gongju 지역군 D11 전체를 제외했습니다. 색인 지오메트리는 출처 메타데이터일 뿐
  CRS·GCP나 정답 좌표로 적용하지 않았습니다.
- 원본과 모델 파일은 Git에서 제외된 로컬 `data/raw/`에 있습니다.
  사용자 자료를 외부 OCR 서버로 업로드하지 않았습니다.

선택은 지역군당 1장, R 접미 변형 선호를 번갈아 적용한 뒤 record ID 순으로
고정했습니다. E9만 사전 실행 점검용 舞鶴里를 사용했습니다. OCR 점수로 표본을
선별하지 않았지만 무작위 표본도 아닙니다. 이들 20장은 모두 개발용입니다.
현재 시나리오는 `old_type`으로 통일하고, 육안 품질 메모를 별도로 남겼습니다.
계획한 `clear`/`old_type`/`degraded` 7/7/6 층화 코퍼스를 충족한다고 주장하지 않습니다.

| 도엽 목록 제목 | Stanford ID | 도엽 목록 제목 | Stanford ID |
| --- | --- | --- | --- |
| 訓戎 | cj987zm6276 | 平昌 | bv010zs1865 |
| 會寧東部 | bk177wq3097 | 義林吉 | bv496gt9686 |
| 四芝洞 | dd086rf1647 | 靈山 | cx546js8403 |
| 院徳場 | br996kj2193 | 廣州 | bb826jm1914 |
| 羅興里 | db935yc4212 | 利川 | ds857mg7684 |
| 南興洞 | bf834yf7024 | 全州 | bn462py5224 |
| 甑山 | fp192cq1681 | 辰橋 | cq534zf2874 |
| 玉溪 | dg017hn3043 | 舞鶴里 | pr492sf6025 |
| 石浦 | gk886vb5506 | 綾州 | cv625mc5401 |
| 梁山 | bg467zc7844 | 飛揚島 | cc013td3269 |

## 고정한 실행 조건

- macOS arm64, 별도 Python 3.12.14 환경. 기존 QGIS·다른 저장소 환경은 변경하지 않았습니다.
- PaddleOCR 3.7.0 / PaddleX 3.7.2 / PaddlePaddle 3.3.1, CPU.
- 공식 [PP-OCRv5 mobile 검출 모델](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det/tree/0d63e78e2b680928f6b1747d76a08db6e645efb7)과
  [mobile 인식 모델](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/tree/682f20538d8c086cb2128e5cfac775e6c4904e85).
  고정 revision의 공식 모델 카드 두 개 모두 Apache-2.0을 표시합니다.
- 모델별 5개 파일의 SHA256을 사전 고정했습니다. weight는 upstream LFS SHA256,
  나머지는 upstream Git blob ID와 대조한 파일입니다.
  명세: `examples/margin_ocr_models.mobile-v5.json`.
- 검출 설정: `text_det_limit_type=max`, `text_det_limit_side_len=640`.
  문서 방향·펴기·글줄 방향 모델은 끄고 인식 점수 임계값은 0입니다.
- 원본 픽셀 그대로 자른 RGB PNG만 OCR에 전달했습니다. 원본은 회전·리사이즈하지
  않았습니다. 검출기의 내부 리사이즈와 원문 보존은 구별해야 합니다.
- Python 소켓 접속을 막고 PaddleX 출처 접속 확인을 끈 독립 CLI입니다.
  네이티브 코드까지 차단하는 OS 네트워크 격리의 증명은 아닙니다.

원본당 네 모서리 80개, 제목 20개, 축척 20개, 범례가 보이는 10도엽에서 2개씩
범례 패치 20개: **총 140개 crop**입니다. 범례 전체를 전사한 시험은 아닙니다.
1600 px 미리보기에서 선형 도곽을 찾아 원본 픽셀 영역을 제안하고 AI가 20장
미리보기를 확인했습니다. 사람의 crop 승인과 전사는 아직 없습니다.

## 첫 실행 결과와 발견한 문제

첫 20도엽 실행 `stanford-20-run-max640-v1`은 140개 crop에서 실행 오류 0건,
검출 있음 137개, 검출 없음 3개였습니다. 검출 있음은 **정답 판독 성공**을
뜻하지 않습니다. 빈 문자열 인식 결과도 원래 엔진 점수 0과 함께 유지합니다.

초기화 1.64초, crop 추론 합계 59.20초, crop당 중앙값 0.178초였습니다.
한 번의 CPU 실행 시간이며 다운로드·JP2 디코딩·crop 생성·해시 확인·사람 검수는
제외했습니다. 다른 기기의 속도나 반복 측정 분포로 일반화하지 않습니다.

원본 확대 점검에서 일부 축척의 아래쪽과 범례 오른쪽 글자가 잘린 것을
확인했습니다. 첫 결과와 입력은 보존하고, 다음 버전에서 축척 아래 여유와
범례 오른쪽 여유만 넓혔습니다. 이 변경은 모델 정확도 개선으로 계산하면 안 됩니다.

제목의 잠정 참고문은 20도엽 OCR 전에 원본 미리보기를 AI가 보고 작성했습니다.
소장기관의 목록 제목은 보이는 상태였으므로 블라인드 사람 정답이 아닙니다.
사전 OCR를 본 舞鶴里와 미리보기에서 불확실했던 義林吉는 비교에서 미리 제외했습니다.
나머지 **18개 중 13개**가 화면상 왼쪽→오른쪽 문자 순서로 원문 완전 일치했습니다.
이는 **AI 참고문에 대한 제목 한정 탐색 비교**이지 72%의 공인 OCR 정확도가 아닙니다.
공백만 제거한 비교도 13개로 같았고, 옛 글자·번체·간체는 임의 정규화하지 않았습니다.

관찰된 실패 예시는 다음과 같습니다. 인용한 원문은 교정하지 않은 엔진 출력입니다.

- 石浦 제목은 화면상 `浦石`인데 `浦后`로 읽었습니다(엔진 점수 0.835).
- 飛揚島의 화면상 `島揚飛`를 `島扬飛`로 읽어 문자 형태가 달라졌습니다(0.812).
- 院徳場은 `場徳院` 대신 `場德院`을 출력했습니다(0.978).
  이런 변형을 허용할지는 사람이 전사 규칙을 정해야 합니다.
- 회전된 경위도와 도·분·초 기호는 누락·순서 뒤바뀜·숫자 오독이 나타났습니다.
  특히 큰 엔진 점수를 정답 좌표의 보증으로 사용할 수 없습니다.
- 사전 점검의 좁은 舞鶴里 제목 crop은 기본 검출 설정에서 획 조각으로 읽혔고,
  같은 crop에 max-side 640을 적용하면 `里鶴舞`가 나왔습니다. 하지만 본 코퍼스의
  더 넓은 제목 crop에서는 같은 설정으로 `里鷗舞`가 나왔습니다. crop 범위까지
  고정하지 않은 설정 비교는 재현 가능한 성능 비교가 아닙니다.

정식 평가 파일은 `pilot_incomplete`, 독립 사람 참조 0도엽, 정확도·CER `null`,
모든 항목 `unreviewed`, `promotion_passed=false`입니다. 모델 설치와 실행 검증은
완료했지만 **독립 사람 판독률 검증은 남아 있습니다**.

## 글자 잘림을 보완한 최종 재실행

`stanford-20-run-max640-v2`에서도 140개 중 검출 있음 137개, 검출 없음 3개,
실행 오류 0건이었습니다. 초기화 1.63초, crop 추론 합계 72.35초, 중앙값
0.243초입니다. 범례와 축척 40개만 넓혔고, 제목·네 모서리 100개의 PNG는 첫
실행과 SHA256이 같습니다. 모델·검출 설정·제목 참고문은 바꾸지 않았습니다.
제목 탐색 비교도 동일하게 18개 중 13개 일치였습니다.

예를 들어 辰橋의 잘린 축척 crop은 첫 실행에서 `R`, `小`만 출력했으나,
넓힌 crop에서는 `尺之一`, `分万`, `五`와 눈금 숫자가 나왔습니다. 羅興里의
범례 오른쪽에서 잘렸던 글자도 원본 crop에 포함된 것을 확인했습니다.
이는 입력 범위의 보완이지 좌표 자동 판독의 검증이 아닙니다. 일부 범례 행의
패치 경계와 축척 막대·단위까지 완전하게 포함했는지는 여전히 수동 점검 대상입니다.

첫 실행은 [v1 증거](evidence/margin-ocr-stanford20-2026-09-12.json), 최종 재실행은
[v2 증거](evidence/margin-ocr-stanford20-2026-09-12-v2.json)로 따로 남겼습니다.
최종 로컬 검수 화면은
`data/derived/margin-ocr/stanford-20-run-max640-v2/report.html`입니다.
정식 검수 양식은 여전히 전부 `unreviewed`이고 정식 평가는 `pilot_incomplete`입니다.

## 재실행과 로컬 산출물

이미 확보한 원본·모델·환경을 그대로 이용하는 명령입니다. 매 실행은 새 출력
폴더를 사용해야 합니다. 원본 다운로드와 OCR 실행은 서로 다른 명령입니다.

```sh
OCR_PY=data/derived/margin-ocr-runtime/bin/python

# 선택한 공개 원본을 새 폴더에 다시 확보할 때만 실행
$OCR_PY scripts/fetch_margin_ocr_corpus.py \
  data/raw/stanford-korea50k/index/index_map.geojson \
  --output data/raw/stanford-korea50k/reacquired-20

# 미리보기 기반 후보를 만든 뒤 원본 crop의 글자 잘림을 확인
$OCR_PY scripts/propose_margin_crops.py \
  data/raw/stanford-korea50k/pilot-20-v1/sources.json \
  --visual-notes examples/margin_ocr_stanford20.visual-notes.json \
  --output data/derived/margin-ocr/proposals-next
$OCR_PY scripts/margin_ocr_pilot.py prepare \
  data/derived/margin-ocr/proposals-next/manifest.proposed.json \
  --output data/derived/margin-ocr/crops-next

$OCR_PY scripts/margin_ocr_pilot.py check-models \
  data/derived/margin-ocr/model-lock-mobile-v5.json
$OCR_PY scripts/margin_ocr_pilot.py run \
  data/derived/margin-ocr/crops-next/bundle.json \
  --model-lock data/derived/margin-ocr/model-lock-mobile-v5.json \
  --det-max-side 640 --output data/derived/margin-ocr/run-next
$OCR_PY scripts/summarize_margin_ocr_pilot.py \
  data/derived/margin-ocr/run-next \
  --visual-notes examples/margin_ocr_stanford20.visual-notes.json \
  --sources data/raw/stanford-korea50k/pilot-20-v1/sources.json \
  --output data/derived/margin-ocr/summary-next.json
```

새 환경에는 `requirements-margin-ocr.txt`가 상위 패키지 명세입니다. 이번 설치의
전이 패키지 hash lock은 로컬 `data/derived/margin-ocr-runtime-lock.txt`이며 SHA256은
`4ca8b2235967783da75d7d26b92902e4ee1adc643d09aefe15873fd7d22940df`입니다.
일반 테스트·QGIS 배포에는 이 OCR 패키지를 추가하지 않았습니다. 고정 모델
revision에서 명세의 파일 5개씩만 받아 같은 상대 디렉터리에 둔 뒤 `pin-models`로
다시 검증해야 합니다. 추가 캐시 파일도 모델 지문 불일치로 처리합니다.

[Stanford 색인 archive](https://stacks.stanford.edu/object/ng525ny5879)의
현재 받아 둔 archive SHA256은
`6cb369e89c2df4a5c7406b7ee31e1ab3d3bdb39f1df130e63782f3458aaeab3f`입니다.
그 안의 `index/index_map.geojson` SHA256은
`0af31910d9654b021692a40cd5b3f41a2c8c5344b6653d50e5fa2553934e97dd`입니다.
온라인 색인이 바뀔 수 있으므로 재확보 시 선택 목록·해시 차이를 확인해야 합니다.

로컬 각 실행 폴더에는 `report.html`, 원문 `crops/`, `result.json`, 검수 양식,
입력 bundle·모델 lock 사본이 있습니다. 원본 획과 OCR 점수를 함께 검토할 수
있습니다. 자세한 검수·정식 평가 방식은 [파일럿 가이드](margin-ocr-pilot.md)를 따릅니다.

## 남은 판정 조건

1. 연구자가 네 모서리·축척·범례 crop의 글자 잘림과 읽기 순서를 확인합니다.
2. OCR를 가린 상태에서 독립 전사하고, 판독 불가·옛 글자 변형 규칙을 명시합니다.
3. 세 품질 시나리오와 영역별 표본을 갖춘 뒤 원문 정확도·CER·기호 오류·검수 시간을
   평가합니다. 현재 AI 제목 비교를 그 수치로 대체하지 않습니다.
4. 별도 사전 수용 기준을 충족한 경우에만 보조 기능 도입을 검토합니다. 이 실험은
   등고선 segmentation 도입 기준을 완화하거나 EfficientSAM을 교체하지 않습니다.
