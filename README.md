# Historical Map Tools — QGIS 플러그인

> **v0.2 — Global core, Korean-first validation:** 한국 도엽을 첫 검증 코퍼스로 삼되, CRS·지도 시리즈 규칙·OCR 설정·출처 메타데이터를 국가에 고정하지 않는 역사 지도 복원 도구입니다. [한국어 안내](README.ko.md)

## v0.2 workflow: frame → registration → contours

1. **Create Sheet Frame / 도곽 만들기**: 지리 DMS/DD 또는 투영 X/Y를 사용해 도곽과 구조화된 출처 정보를 저장합니다.
2. **Register Map / 지도 맞추기**: 원본 스캔의 4모서리·선택적 내부 GCP·프로젝티브 변환계수·RMSE를 JSON으로 보존합니다.
3. **Extract Contours / 등고 추출하기**: 프로파일 기반 가시 등고선과 검수용 연결 후보를 출력합니다. 후보 `status`를 `approved`로 바꾼 후 **Create Approved Contour Completions**을 실행하면 승인된 보완 구간만 출력합니다.
4. **Assess Contour Baseline / 등고선 기준선 측정**: 원본 스캔과 `MapProfile`을 입력하면, 학습 없이 색상·형태학 처리 기반으로 가시선 수·길이·끝점·검수 후보 수를 JSON으로 기록합니다. 기본 분석 크기는 긴 변 2,048 px이며, 이 보고서는 정확도 주장이 아니라 프로파일 조정과 수동 정답선 준비를 위한 진단입니다.

한국용 Tokyo CRS는 설정 가능한 예시 프로파일일 뿐 코드의 국가 분기가 아닙니다. 원본 고해상도 스캔은 Git에서 제외하고, 검수에 필요한 작은 타일·후보·GeoPackage만 휴대 가능한 묶음으로 관리합니다. 모든 도엽에는 제작기관·목적·시기·언어/문자·CRS/수직기준·출처·권리·역사적 맥락을 보존합니다.

첫 검증을 위한 세 도엽 구성, 반복 실행 절차, ML 도입 기준은 [Initial Pilot Protocol](docs/pilot-protocol.md)에 정리했습니다.

현재 우선 작업은 [실제 도엽의 등고선 추출 개선](docs/contour-extraction-progress.md)입니다.
9개 개발 타일에서 벡터 변환의 가짜 분기와 미리보기 불일치를 수정하고,
문맥 필터 후보·원본 겹침 비교·새 QGIS 비교 프로젝트를 만들었습니다.
도로·하천 혼입은 아직 해결되지 않았으며 모든 출력은 검수 후보입니다.
후속 [주변 방향·간격 분류 실험](docs/contour-neighborhood-experiment.md)은 162개 고정 표본과
도엽 단위 검증으로 혼입 감소·추가 누락을 함께 확인했습니다. 선별·보수·불확실 후보와
54레이어 QGIS 비교 프로젝트를 제공하며, 새 모델은 기본 추출기를 대체하지 않습니다.

별도의 [PaddleOCR 여백 보조 파일럿](docs/margin-ocr-pilot.md)은 명시적으로 선택한
crop의 원문 이미지·OCR 원문·신뢰도를 기록하는 독립 CLI입니다. 모든 결과는
검수 전 후보이며 좌표·CRS·GCP를 자동 적용하지 않습니다. Stanford 공개 원본
20도엽을 확보해 실제 로컬 모델을 실행했습니다. [실행 기록과 한계](docs/margin-ocr-stanford20.md)를
참조하세요. 사람의 독립 전사에 따른 판독률은 아직 검증하지 않았습니다.

실제 스캔에 여백·범례·외곽 프레임이 있는 경우 **Register Map**에서 이미지 모서리가 아니라 인쇄된 지도 내부 도곽의 `NW → NE → SE → SW`를 클릭합니다. Processing의 `PIXEL_CORNERS`에도 같은 순서의 픽셀 좌표를 입력할 수 있습니다.

현재 로컬 파일럿은 청양–공주 / 부여–논산의 인접 흑백 도엽 4장입니다. 좌표 범위, `+10.4″` 보정 표기의 해석, 내부 도곽 픽셀 GCP와 검증 방법은 [Korea Connected Four-Sheet Pilot](docs/korea-four-sheet-pilot.md)에 기록했습니다. 원본은 Git에 포함되지 않습니다.

다른 컴퓨터에서 같은 QGIS 검수 작업을 이어가는 방법은 [Portable QGIS Review Workflow](docs/portable-review-workflow.md)를 따르세요. 저장소의 `Open Review Project.cmd`로 프로젝트를 열고, 작업 전 `Update Review Workspace.cmd`, QGIS 종료 후 `Save Review Work.cmd`를 사용합니다.

흑백 선화 후보를 비교할 때는 [Ink v2 Centerline A/B Workflow](docs/ink-centerline-ab-workflow.md)를 사용합니다. 이 백엔드는 ArchaeoTrace Ink v2를 고정 커밋에서 이식한 **검수 전용 선화 중심선**이며, 기존 등고 후보나 `proposal_review` 큐를 덮어쓰지 않습니다. 공주 홀드아웃 타일은 기본 실행에서 제외됩니다.

끊어진 등고선의 옅은 구간·글자 가림·기호 겹침은 [Contour Completion Research](docs/contour-completion-research.md)에서 별도로 다룹니다. 기본값은 사람이 `contour`로 검수한 선만 연결하며, 분류 전 전체 linework 실험은 명시적인 `--anchor-status all`에서만 허용합니다.

> **역사 지형도의 도·분·초 좌표를 그대로 입력해 도곽(外圍線) 폴리곤을 QGIS에 생성합니다.**

[![버전](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/lzpxilfe/jap-map/releases)
[![QGIS](https://img.shields.io/badge/QGIS-%E2%89%A53.40-green)](https://qgis.org)
[![라이선스](https://img.shields.io/badge/license-GPL--2.0-orange)](LICENSE)

---

## 왜 이 플러그인인가?

역사 지형도(조선 지형도, 일제강점기 측량 도엽 등)에는 귀퉁이마다 **도°분′초″** 형식의 위도·경도가 인쇄되어 있습니다. 그런데 일반적인 GIS 도구는 **십진도(Decimal Degrees)** 방식의 단일 숫자 입력만 지원합니다.

이때 `36°30′`을 그냥 `36.30`으로 입력하면 실제 위치와 약 28 km 차이가 나지만, **프로그램은 아무 오류도 내지 않고 전혀 다른 곳에 도곽을 그립니다.** 사용자는 왜 좌표가 틀렸는지 거꾸로 추적해야 하는 상황에 놓입니다.

**Historical Map Tools**는 이 문제를 뿌리부터 해결합니다.

- **도°분′초″ 세 칸 입력**: 지도에 인쇄된 값을 칸별로 그대로 옮겨 씁니다.
- **실시간 십진도 미리보기**: 입력하는 즉시 해석 결과(`→ 36.500000°`)를 보여줍니다.
- **즉각적인 오류 알림**: 분·초 값이 0–59를 벗어나거나 범위가 잘못되면 붉은 경고가 바로 표시됩니다.

---

## 설치

### ZIP으로 설치 (권장)

1. [Releases](https://github.com/lzpxilfe/jap-map/releases) 페이지에서 최신 `jap_map.zip` 을 내려받습니다.
2. QGIS 메뉴에서 **플러그인 → 플러그인 관리 및 설치** 를 엽니다.
3. **"ZIP에서 설치"** 탭을 클릭합니다.
4. 내려받은 `jap_map.zip` 파일을 선택합니다.
5. **"플러그인 설치"** 버튼을 누릅니다.
6. 설치 후 플러그인 목록에서 **Historical Map Tools** 체크박스를 활성화합니다.

> **요구 환경**: QGIS ≥ 3.40, Python 3.12

---

## 사용 방법

### 1단계 — 다이얼로그 열기

QGIS 메뉴에서 **벡터 → 도곽 만들기…** 또는 툴바의 **역사지형도** 아이콘을 클릭합니다.

### 2단계 — 입력 좌표 CRS 선택

지도 귀퉁이에 적힌 좌표가 어떤 측지계인지 선택합니다.

| CRS | EPSG | 언제 쓰나 |
|---|---|---|
| **Tokyo 1892** | `EPSG:5132` | 조선 지형도 1:50,000 초기 판본 |
| **Tokyo / Tokyo 1918** | `EPSG:4301` | 조선 지형도 개정판, 지적도 |
| **WGS 84** | `EPSG:4326` | 이미 현대 좌표로 변환된 자료 |
| 기타 | `기타…` 버튼 | QGIS에 등록된 모든 CRS 검색 |

> **팁**: 판본이 불확실하면 Tokyo 1892와 Tokyo 1918 두 결과를 모두 만들어 현재 배경지도와 비교하세요. 두 결과는 약 300–400 m 차이가 납니다.

### 3단계 — 모서리 좌표 입력

도곽의 네 귀퉁이(좌상·우상·좌하·우하)에 각각 위도·경도를 입력합니다.

```
예) 지도에 "N 37° 30′ 00″ / E 127° 00′ 00″" 라고 적혀 있다면:

  위도 칸: 37 ° | 30 ′ | 00 ″ | N
  경도 칸: 127 ° | 00 ′ | 00 ″ | E
```

입력하는 즉시 아래에 **→ 37.500000°** 처럼 십진도 해석 결과가 초록색으로 표시됩니다. 이 값이 예상한 위치와 다르면 지금 바로 확인하고 수정할 수 있습니다.

### 4단계 — 도곽 만들기

**도곽 만들기** 버튼을 누르면 현재 프로젝트에 임시 폴리곤 레이어가 추가됩니다.

- 레이어 이름은 도엽명으로 자동 설정됩니다.
- 레이어를 영구 저장하려면 레이어 패널에서 오른쪽 클릭 → **다른 이름으로 저장** 을 사용하세요.

---

## 파일 구성

```
jap_map/
├── __init__.py            # QGIS 진입점 (classFactory)
├── plugin.py              # 플러그인 클래스, 메뉴/툴바 등록
├── dialog.py              # UI 다이얼로그 (도·분·초 입력 위젯 포함)
├── icon.svg               # 툴바 아이콘
├── metadata.txt           # QGIS 플러그인 메타데이터
└── core/
    ├── __init__.py        # 공개 API 노출
    ├── coordinates.py     # 도·분·초 → 십진도 변환 함수
    ├── frame.py           # 도곽 폴리곤 생성 로직
    └── layer_manager.py   # QGIS 레이어 추가/관리
```

---

## 측지계 참고 사항

조선 지형도는 일본 육지측량부가 제작한 도엽으로, **Tokyo Datum(동경원점)** 을 기준으로 합니다. 이 측지계는 WGS 84와 다음과 같은 차이가 있습니다.

- **경도**: 약 +10.4″ 동쪽 오프셋 (약 ~285 m)
- **위도**: 약 +0.1″ 북쪽 오프셋 (약 ~3 m)

따라서 도엽의 좌표를 WGS 84로 잘못 입력하면 도곽이 약 285 m 동쪽으로 어긋납니다.

---

## 개발 / 기여

```bash
# 저장소 복제
git clone https://github.com/lzpxilfe/jap-map.git
cd jap-map

# 단위 테스트 실행 (QGIS 없이 실행 가능)
python -m unittest tests.test_coordinates tests.test_frame tests.test_histcontour_core tests.test_ink tests.test_ink_candidate_script tests.test_contour_completion tests.test_contour_completion_script tests.test_segment_review tests.test_ink_segment_review_script

# Ink v2 개발 타일 A/B 후보 생성 (NumPy와 Pillow 필요)
python scripts/generate_ink_centerline_candidates.py data/derived/annotation_package/index.json

# 사람이 contour로 판정한 선 사이의 끊김 후보 생성
python scripts/generate_contour_completion_candidates.py data/derived/annotation_package/index.json

# Ink 선분 학습용 360개 다양성 검토 큐 생성 (공주 홀드아웃 제외)
python scripts/prepare_ink_segment_review.py data/derived/annotation_package/index.json

# 라벨 수 확인 및 준비되면 도엽 단위 교차검증 기준 모델 학습
python scripts/train_ink_segment_classifier.py data/derived/annotation_package/contour_annotations.gpkg

# 합성 지도 120개 × 8개 변형에서 Ink 등고선 점수 기준선 측정
python scripts/run_synthetic_ink_experiment.py --write-cases

# QGIS 통합 테스트
# QGIS Python Console에서:
#   import unittest
#   unittest.main(module='tests.test_layer_manager', exit=False)

# 플러그인 ZIP 빌드
python scripts/build_plugin_zip.py
```

### 기여 방법

1. 이 저장소를 Fork합니다.
2. 새 브랜치를 만듭니다 (`git checkout -b feat/my-feature`)
3. 변경 사항을 커밋합니다 (`git commit -m 'feat: my feature'`)
4. Push 후 Pull Request를 보냅니다.

---

## 변경 이력

### v0.1.0 (2026-08-06)
- **도·분·초 3칸 분리 입력 UI** 도입 (`QSpinBox` × 2 + `QDoubleSpinBox`)
- **실시간 십진도 미리보기**: 입력 즉시 해석 결과 표시
- **즉각적인 범위 오류 경고**: 분·초 0–59 범위 초과 시 붉은 알림
- **CRS 즉시 반영 버그 수정**: `QgsProjectionSelectionWidget` 내부 상태 지연 우회
- `dms_to_decimal()` 변환 함수 추가 (`jap_map.core.coordinates`)

---

## 라이선스

이 프로젝트는 [GNU General Public License v2.0](LICENSE) 하에 배포됩니다.
