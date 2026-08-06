# Historical Map Tools — QGIS 플러그인

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
python -m unittest tests.test_coordinates tests.test_frame

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
