# TRD: Activation Function 시각화 GUI

> **Technical Requirements Document**
> Version: 1.0.0
> Last Updated: 2026-03-25
> Status: Draft
> 참조 PRD: PRD_activation_function_gui.md v1.0.0

---

## 1. 기술 스택 (Tech Stack)

| 항목 | 선택 | 버전 | 비고 |
|------|------|------|------|
| **언어** | Python | 3.10+ | |
| **GUI 프레임워크** | PySide6 | 6.6+ | Qt 6 기반 공식 Python 바인딩 |
| **그래프 라이브러리** | Matplotlib | 3.8+ | PySide6 백엔드 사용 |
| **수치 연산** | NumPy | 1.26+ | 함수 계산 및 미분 |
| **의존성 관리** | pip / requirements.txt | - | |

### 1.1 Matplotlib ↔ PySide6 통합

```python
# Matplotlib 백엔드를 Qt6Agg로 설정
import matplotlib
matplotlib.use("Qt6Agg")
from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
```

---

## 2. 프로젝트 구조 (Project Structure)

```
activation_gui/
│
├── main.py                     # 진입점 (QApplication 실행)
│
├── ui/
│   ├── main_window.py          # QMainWindow: 전체 레이아웃 구성
│   ├── sidebar.py              # QListWidget 기반 함수 목록 사이드바
│   ├── graph_canvas.py         # Matplotlib FigureCanvas 래퍼
│   ├── parameter_panel.py      # 파라미터 슬라이더 / 스핀박스 패널
│   └── description_panel.py    # 수식 + 설명 텍스트 패널
│
├── core/
│   ├── functions.py            # Activation Function 수치 구현
│   ├── gradients.py            # Gradient(미분) 수치 구현
│   └── function_registry.py   # 함수 메타데이터 등록/조회
│
├── i18n/
│   ├── ko.json                 # 한국어 문자열
│   └── en.json                 # 영어 문자열
│
├── assets/
│   └── icons/                  # 툴바 아이콘 등
│
├── tests/
│   ├── test_functions.py
│   └── test_gradients.py
│
└── requirements.txt
```

---

## 3. 모듈 상세 설계

### 3.1 `core/functions.py` — Activation Function 구현

모든 함수는 `np.ndarray` 입력을 받아 `np.ndarray`를 반환한다.

```python
import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    # 시각화용: 단일 입력 벡터 처리 (x를 배열로 취급)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x >= 0, x, alpha * x)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def gelu(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return x * sigmoid(beta * x)

def prelu(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    return np.where(x >= 0, x, alpha * x)

def selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
```

### 3.2 `core/gradients.py` — Gradient(미분) 구현

```python
def grad_relu(x):
    return np.where(x >= 0, 1.0, 0.0)

def grad_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def grad_tanh(x):
    return 1 - np.tanh(x)**2

def grad_leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, 1.0, alpha)

def grad_elu(x, alpha=1.0):
    return np.where(x >= 0, 1.0, alpha * np.exp(x))

def grad_gelu(x):
    # 수치 미분 사용 (근사)
    h = 1e-5
    return (gelu(x + h) - gelu(x - h)) / (2 * h)

def grad_swish(x, beta=1.0):
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)

def grad_prelu(x, alpha=0.25):
    return np.where(x >= 0, 1.0, alpha)

def grad_selu(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * np.where(x >= 0, 1.0, alpha * np.exp(x))
```

### 3.3 `core/function_registry.py` — 함수 메타데이터

각 함수의 메타데이터를 딕셔너리로 관리한다.

```python
FUNCTION_REGISTRY = {
    "ReLU": {
        "fn": relu,
        "grad": grad_relu,
        "params": {},  # 파라미터 없음
        "formula_ko": r"$f(x) = \max(0, x)$",
        "formula_en": r"$f(x) = \max(0, x)$",
        "grad_formula": r"$f'(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$",
        "desc_ko": "가장 널리 쓰이는 활성화 함수...",
        "desc_en": "The most widely used activation function...",
        "category_ko": "기본",
        "category_en": "Basic",
    },
    "Leaky ReLU": {
        "fn": leaky_relu,
        "grad": grad_leaky_relu,
        "params": {
            "alpha": {"min": 0.0, "max": 1.0, "default": 0.01, "step": 0.01}
        },
        ...
    },
    ...
}
```

**파라미터 스펙 테이블**

| 함수 | 파라미터 | 최솟값 | 최댓값 | 기본값 | 스텝 |
|------|----------|--------|--------|--------|------|
| Leaky ReLU | alpha | 0.0 | 1.0 | 0.01 | 0.01 |
| ELU | alpha | 0.1 | 5.0 | 1.0 | 0.1 |
| Swish | beta | 0.1 | 5.0 | 1.0 | 0.1 |
| PReLU | alpha | 0.0 | 1.0 | 0.25 | 0.01 |
| ReLU / Sigmoid / Tanh / GELU / SELU / Softmax | — | — | — | — | — |

---

### 3.4 `ui/main_window.py` — 메인 윈도우

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Activation Function Explorer")
        self.resize(1280, 800)

        # Central Widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 사이드바 (240px 고정 폭)
        self.sidebar = SidebarWidget()
        self.sidebar.setFixedWidth(240)

        # 우측 영역 (그래프 + 파라미터 + 설명)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.graph_canvas = GraphCanvas()
        self.param_panel = ParameterPanel()
        self.desc_panel = DescriptionPanel()

        right_layout.addWidget(self.graph_canvas, stretch=5)
        right_layout.addWidget(self.param_panel, stretch=2)
        right_layout.addWidget(self.desc_panel, stretch=3)

        layout.addWidget(self.sidebar)
        layout.addWidget(right_panel)

        # 시그널 연결
        self.sidebar.functionSelected.connect(self.on_function_selected)
        self.param_panel.paramChanged.connect(self.on_param_changed)
```

### 3.5 `ui/graph_canvas.py` — 그래프 캔버스

```python
class GraphCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        super().__init__(self.fig)
        self.x = np.linspace(-5, 5, 500)

    def plot(self, fn, grad_fn, fn_name: str, params: dict):
        self.ax.clear()
        y = fn(self.x, **params)
        dy = grad_fn(self.x, **params)

        self.ax.plot(self.x, y, label=fn_name, color="#2563EB", linewidth=2)
        self.ax.plot(self.x, dy, label=f"∇ {fn_name}", color="#EA580C",
                     linewidth=1.5, linestyle="--")
        self.ax.axhline(0, color="gray", linewidth=0.5)
        self.ax.axvline(0, color="gray", linewidth=0.5)
        self.ax.legend()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.set_ylim(-2, 2)
        self.draw()
```

### 3.6 `ui/parameter_panel.py` — 파라미터 패널

- 선택된 함수의 `params` 딕셔너리를 읽어 동적으로 위젯을 생성한다
- `QSlider` + `QDoubleSpinBox`를 한 행에 배치
- 파라미터 없는 함수 선택 시 "이 함수는 조절 가능한 파라미터가 없습니다" 텍스트 표시

```python
# 시그널 정의
paramChanged = Signal(dict)  # {"alpha": 0.05} 형태로 emit
```

### 3.7 `ui/description_panel.py` — 수식 & 설명 패널

- 수식 렌더링: `Matplotlib Figure` 내 `fig.text()` 또는 `QLabel`에 Matplotlib `mathtext` 이미지를 렌더링하여 표시
- 설명 텍스트: `QTextBrowser` (HTML 렌더링 지원)
- 현재 언어에 따라 `desc_ko` / `desc_en` 분기

---

## 4. 국제화(i18n) 설계

### 4.1 구조

```json
// i18n/ko.json
{
  "app_title": "활성화 함수 탐색기",
  "sidebar_basic": "기본 함수",
  "sidebar_advanced": "고급 함수",
  "param_label": "파라미터 조절",
  "no_params": "이 함수는 조절 가능한 파라미터가 없습니다.",
  "formula_label": "수식",
  "gradient_label": "미분식",
  "description_label": "설명",
  "lang_toggle": "EN"
}
```

### 4.2 LanguageManager 클래스

```python
class LanguageManager:
    _current_lang = "ko"
    _strings: dict = {}

    @classmethod
    def load(cls, lang: str):
        cls._current_lang = lang
        with open(f"i18n/{lang}.json", encoding="utf-8") as f:
            cls._strings = json.load(f)

    @classmethod
    def t(cls, key: str) -> str:
        return cls._strings.get(key, key)
```

- 언어 전환 시 `QMainWindow`에서 `retranslateUi()` 호출 → 모든 위젯 텍스트 갱신

---

## 5. 시그널/슬롯 흐름 (Signal Flow)

```
SidebarWidget
  └─ Signal: functionSelected(fn_name: str)
       └─ MainWindow.on_function_selected()
            ├─ ParameterPanel.load_params(fn_meta)
            ├─ DescriptionPanel.update(fn_meta)
            └─ GraphCanvas.plot(fn, grad_fn, params)

ParameterPanel
  └─ Signal: paramChanged(params: dict)
       └─ MainWindow.on_param_changed()
            └─ GraphCanvas.plot(fn, grad_fn, params)  ← 실시간 업데이트

ToolbarWidget
  └─ Signal: languageToggled(lang: str)
       └─ MainWindow.on_language_change()
            └─ LanguageManager.load(lang)
            └─ retranslateUi() 호출
```

---

## 6. 성능 요구사항 구현 방법

| 요구사항 | 구현 방법 |
|----------|-----------|
| 슬라이더 조작 후 ≤ 100ms 업데이트 | `QSlider.valueChanged` → `canvas.plot()` 직접 연결, NumPy 벡터 연산으로 빠른 계산 |
| 슬라이더 드래그 중 과도한 redraw 방지 | `QTimer` 디바운싱 (50ms) 적용 |
| 그래프 깜박임 방지 | `fig.canvas.draw_idle()` 사용 |

```python
# 디바운싱 예시
self._update_timer = QTimer()
self._update_timer.setSingleShot(True)
self._update_timer.setInterval(50)
self._update_timer.timeout.connect(self._do_plot)

def on_param_changed(self, params):
    self._pending_params = params
    self._update_timer.start()  # 50ms 내 추가 변경 시 타이머 재시작
```

---

## 7. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| NumPy overflow (e.g. exp 폭발) | `np.clip()` 또는 `np.errstate(over='ignore')` 적용 |
| Softmax 시각화 | 단일 스칼라 입력을 `[x]` 배열로 감싸 처리, 시각화 시 원소 0 값만 표시 |
| 파라미터 범위 위반 | `QDoubleSpinBox` 자체 min/max로 UI 차단 |
| i18n 파일 누락 | 기본값 `"ko"` fallback 처리 |

---

## 8. 테스트 계획

### 8.1 단위 테스트 (`tests/`)

```python
# test_functions.py 예시
def test_relu_positive():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(relu(x), [1.0, 2.0, 3.0])

def test_relu_negative():
    x = np.array([-1.0, -2.0])
    assert np.allclose(relu(x), [0.0, 0.0])

def test_sigmoid_range():
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    assert np.all(y >= 0) and np.all(y <= 1)
```

### 8.2 통합 테스트

| 시나리오 | 검증 내용 |
|----------|-----------|
| 함수 전환 | 사이드바 클릭 시 그래프·수식·파라미터 패널 모두 갱신 |
| 파라미터 슬라이더 조작 | 그래프 실시간 변경, 100ms 이내 반응 |
| 언어 전환 | 모든 UI 텍스트 변경, 수식은 유지 |
| 창 리사이즈 | 레이아웃 비율 유지, 그래프 깨짐 없음 |

---

## 9. 의존성 (requirements.txt)

```
PySide6>=6.6.0
matplotlib>=3.8.0
numpy>=1.26.0
```

---

## 10. 개발 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python main.py

# 테스트
python -m pytest tests/
```

---

## 11. 향후 기술 고려사항 (v2.0)

| 기능 | 기술 방안 |
|------|-----------|
| 함수 간 비교 모드 | `ax` 서브플롯 또는 overlay 방식 |
| 사용자 정의 함수 입력 | `eval()` + 샌드박스 처리 또는 `sympy` 파싱 |
| 그래프 저장 | `fig.savefig()` + `QFileDialog` |
| 다크 모드 | Matplotlib 스타일시트 `plt.style.use('dark_background')` + Qt 팔레트 |

---

*이 문서는 PRD v1.0.0 기반으로 작성되었으며, 구현 착수 전 기술 검토가 필요합니다.*
