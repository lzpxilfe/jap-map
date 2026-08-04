# Historical Map Tools

QGIS 플러그인 `Historical Map Tools`는 역사 지형도의 네 귀퉁이 위도·경도를 입력해 도곽 Polygon을 만듭니다.

## 사용

1. `jap_map` 플러그인 ZIP을 QGIS에서 **플러그인 > ZIP에서 설치**로 설치합니다.
2. Vector 메뉴 또는 툴바에서 **도곽 만들기…**를 엽니다.
3. 원도에 맞는 CRS를 고릅니다.
4. 좌상·우상·우하·좌하에 위도와 경도를 입력합니다.
5. **도곽 만들기**를 누릅니다.

십진도(`37.5`), 도분(`37°30′`), 도분초(`37°30′00″N`)를 사용할 수 있습니다. 결과는 현재 프로젝트에 임시 레이어로 추가됩니다. 장기 보관이 필요하면 QGIS의 레이어 메뉴에서 영구 저장을 사용하세요.

### 측지계 참고

조선 지형도는 `Tokyo 1892 (EPSG:5132)`를 먼저 시도할 수 있지만, 판본에 따라 `Tokyo / Tokyo 1918 (EPSG:4301)` 경위도가 사용되었을 가능성이 있습니다. 두 체계의 결과를 도엽 정보와 현대 배경지도에 비교해 선택해야 합니다.

## 개발

```text
python -m unittest tests.test_coordinates tests.test_frame
python scripts/build_plugin_zip.py
```

QGIS 통합 테스트는 QGIS가 제공하는 `qgis_testrunner.sh`에서 실행합니다.

## 범위

현재 버전은 도곽 생성만 제공합니다. 래스터 지오리퍼런싱, 자동 도곽선 검출, OCR, 등고선 벡터화는 후속 단계입니다.
