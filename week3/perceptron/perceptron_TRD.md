# TRD: 퍼셉트론(Perceptron) 개념 학습 데스크톱 애플리케이션

---

## 1. 개요 (Overview)

| 항목 | 내용 |
|------|------|
| 프로젝트명 | 퍼셉트론 개념 학습 앱 |
| 버전 | v1.0 |
| 작성일 | 2026-03-25 |
| 참조 PRD | PRD v1.0 |
| 진입점 | `python main.py` |
| 상태 | 초안 (Draft) |

---

## 2. 기술 스택 (Tech Stack)

| 카테고리 | 기술 | 버전 | 역할 |
|---------|------|------|------|
| UI 프레임워크 | **PySide6** | ≥ 6.6 | 메인 윈도우, 탭 위젯, QPainter, QTimer |
| 수치 계산 | **NumPy** | ≥ 1.24 | 퍼셉트론 행렬 연산, 게이트 데이터 |
| 그래프 | **Matplotlib** | ≥ 3.7 | 결정 경계, 오류율 그래프 |
| Matplotlib 임베드 | `FigureCanvasQTAgg` | Matplotlib 내장 | PySide6 위젯으로 Matplotlib 삽입 |
| 언어 | **Python** | ≥ 3.11 | 전체 애플리케이션 로직 |

### 의존성 설치

```bash
pip install PySide6 numpy matplotlib
```

**requirements.txt**
```
PySide6>=6.6.0
numpy>=1.24.0
matplotlib>=3.7.0
```

### 기술 선택 근거

| 기술 | 선택 근거 | 대안 및 제외 이유 |
|------|----------|-----------------|
| PySide6 | Qt 공식 Python 바인딩, 크로스플랫폼, LGPL 라이선스 | PyQt6: GPL 라이선스 제약 |
| QPainter | 뉴런 다이어그램 픽셀 단위 커스텀 렌더링 | SVG: 동적 업데이트 복잡 |
| Matplotlib + QTAgg | PySide6 네이티브 임베드, 과학 그래프 특화 | PyQtGraph: 설치 추가, 과학 표현 부족 |
| NumPy | 퍼셉트론 가중치 벡터 연산 효율화 | 순수 Python 리스트: 느리고 오류 위험 |

---

## 3. 프로젝트 구조 (Project Structure)

```
perceptron_app/
├── main.py                       # 진입점: QApplication 생성 및 MainWindow 실행
├── requirements.txt              # 의존성 목록
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py            # MainWindow: QTabWidget + 탭 조립
│   ├── tab_overview.py           # Tab 1: 개요 (스크롤 가능한 설명 페이지)
│   ├── tab_structure.py          # Tab 2: 구조 시각화 + 슬라이더
│   ├── tab_learning.py           # Tab 3: 학습 애니메이션 + 오류 그래프
│   ├── tab_boundary.py           # Tab 4: 결정 경계 시각화
│   └── tab_quiz.py               # Tab 5: 퀴즈
│
├── widgets/
│   ├── __init__.py
│   ├── neuron_diagram.py         # QPainter 퍼셉트론 다이어그램 커스텀 위젯
│   └── mpl_canvas.py             # Matplotlib FigureCanvasQTAgg 래퍼
│
├── core/
│   ├── __init__.py
│   ├── perceptron.py             # Perceptron 모델 클래스 (순전파, 학습 규칙)
│   └── gate_data.py              # AND / OR / XOR 진리표 데이터
│
├── data/
│   ├── __init__.py
│   └── quiz_data.py              # 퀴즈 문제 데이터 (5문제)
│
└── styles/
    ├── __init__.py
    └── theme.py                  # 다크 모드 QSS 스타일시트 문자열 상수
```

---

## 4. 핵심 클래스 설계 (Class Design)

### 4.1 Perceptron (`core/perceptron.py`)

퍼셉트론의 순전파와 학습 로직을 담당하는 순수 Python 클래스 (UI 의존성 없음).

```python
class Perceptron:
    weights: np.ndarray   # shape (2,) — w₁, w₂
    bias: float           # 편향 b
    lr: float             # 학습률 η

    def activation(self, z: float) -> int:
        """계단 함수(Step Function): z >= 0 → 1, else → 0"""

    def predict(self, x: np.ndarray) -> int:
        """순전파: z = w·x + b → activation(z)"""

    def predict_raw(self, x: np.ndarray) -> dict:
        """순전파 상세 반환: {'z': float, 'y_hat': int}"""

    def train_step(self, x: np.ndarray, y: int) -> dict:
        """
        단일 샘플 학습 1스텝 수행
        반환값:
          z       : float  — 가중합
          y_hat   : int    — 예측값
          error   : int    — y - ŷ
          delta_w : ndarray — 가중치 변화량
          delta_b : float  — 편향 변화량
          updated : bool   — 업데이트 발생 여부
          w       : ndarray — 업데이트 후 가중치
          b       : float  — 업데이트 후 편향
        """

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> int:
        """전체 데이터 1에포크 학습 → 오류 수 반환"""

    def reset(self):
        """가중치·편향을 0으로 초기화"""

    def set_weights(self, w1: float, w2: float, b: float):
        """외부에서 가중치 직접 설정 (Tab 2 슬라이더 연동용)"""
```

### 4.2 NeuronDiagramWidget (`widgets/neuron_diagram.py`)

QPainter로 퍼셉트론 구조를 그리는 커스텀 위젯.

```python
class NeuronDiagramWidget(QWidget):

    def set_weights(self, w1: float, w2: float, bias: float):
        """가중치 값 업데이트 후 update() 호출"""

    def set_inputs(self, x1: float, x2: float):
        """입력값 업데이트 후 update() 호출"""

    def paintEvent(self, event: QPaintEvent):
        """전체 다이어그램 렌더링"""

    # 내부 드로잉 메서드
    def _draw_node(self, painter, cx, cy, r, label, bg_color, text_color): ...
    def _draw_weighted_edge(self, painter, x1, y1, x2, y2, weight, label): ...
    def _draw_arrowhead(self, painter, x1, y1, x2, y2, color): ...
    def _draw_section_labels(self, painter, ...): ...
```

**렌더링 요소 명세:**

| 요소 | 구현 방식 | 색상 규칙 |
|------|----------|----------|
| 입력 노드 (x₁, x₂) | `drawEllipse` | `#313244` (다크 그레이) |
| 편향 노드 (+1) | `drawEllipse` | `#f9e2af` (노란색) |
| 가중합 노드 (Σ) | `drawEllipse` | `#1e66f5` (파란색) |
| 활성화 노드 (f) | `drawEllipse` | `#8839ef` (보라색) |
| 출력 노드 (y) | `drawEllipse` | y=1: `#40a02b`, y=0: `#e64553` |
| 엣지 (양수 가중치) | `drawLine` + 화살표 | `#89b4fa` (하늘색), 두께 = \|w\|×2+1 |
| 엣지 (음수 가중치) | `drawLine` + 화살표 | `#f38ba8` (빨간색) |
| 가중치 레이블 | `drawText` | 엣지 중간 위치 |
| 섹션 헤더 | `drawText` | `#a6adc8` (회색) |

### 4.3 MplCanvas (`widgets/mpl_canvas.py`)

Matplotlib Figure를 PySide6 위젯으로 임베드하는 래퍼.

```python
class MplCanvas(FigureCanvasQTAgg):
    fig: Figure
    axes: Axes

    def clear_axes(self):
        """axes 초기화 + 다크 테마 재적용"""

    def draw_decision_boundary(
        self,
        weights: np.ndarray,
        bias: float,
        gate_data: dict,
        gate_name: str
    ):
        """
        결정 경계 그래프 렌더링
        - 데이터 포인트: 출력 1=초록원, 0=빨간사각형
        - 경계선: w₁x + w₂y + b = 0
        - w₂≈0 이면 수직선 처리
        """

    def draw_error_curve(self, errors: list[int]):
        """에포크별 오류 수 꺾은선 그래프"""
```

### 4.4 MainWindow (`ui/main_window.py`)

```python
class MainWindow(QMainWindow):
    def __init__(self):
        # 공유 Perceptron 인스턴스
        self.perceptron = Perceptron()

        # 탭 조립
        self.tab_widget = QTabWidget()
        self.tab_overview   = OverviewTab()
        self.tab_structure  = StructureTab()
        self.tab_learning   = LearningTab()
        self.tab_boundary   = BoundaryTab()
        self.tab_quiz       = QuizTab()

        # Tab 2 슬라이더 → Tab 4 결정 경계 동기화
        self.tab_structure.weights_changed.connect(
            self.tab_boundary.update_weights
        )
```

---

## 5. 퍼셉트론 학습 알고리즘 명세

```
알고리즘: Rosenblatt 퍼셉트론 학습 규칙 (1958)

초기화:
  w = [0.0, 0.0]
  b = 0.0

for epoch in range(MAX_EPOCHS):
  errors = 0
  for (x, y) in dataset:
    z    = w · x + b              ← 가중합
    ŷ   = step(z)                 ← 예측 (step: z≥0 → 1, else → 0)
    error = y - ŷ
    if error ≠ 0:                 ← 예측 오류 시에만 업데이트
      w ← w + η × error × x
      b ← b + η × error
      errors += 1

  if errors == 0:
    break                          ← 수렴: 전체 샘플 완벽 분류

수렴 보장: 데이터가 선형 분리 가능한 경우 유한 스텝 내 보장
          (퍼셉트론 수렴 정리, Rosenblatt 1962)
```

---

## 6. 게이트 데이터 명세 (`core/gate_data.py`)

```python
GATE_DATA = {
    "AND": {
        "X": np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        "y": np.array([0, 0, 0, 1]),
        "linearly_separable": True,
        "description": "두 입력이 모두 1일 때만 출력 1",
    },
    "OR": {
        "X": np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        "y": np.array([0, 1, 1, 1]),
        "linearly_separable": True,
        "description": "하나 이상 입력이 1이면 출력 1",
    },
    "XOR": {
        "X": np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        "y": np.array([0, 1, 1, 0]),
        "linearly_separable": False,
        "description": "입력이 서로 다를 때 출력 1 — 선형 분리 불가",
    },
}
```

---

## 7. 퀴즈 데이터 명세 (`data/quiz_data.py`)

### 문제 타입 정의

```python
@dataclass
class QuizQuestion:
    id: int
    topic: str                    # "bias" | "calc" | "rule" | "limit" | "convergence"
    question: str                 # 문제 텍스트
    options: list[str]            # 4지선다 보기
    answer: int                   # 정답 인덱스 (0-based)
    explanation: str              # 해설
```

### 5문제 목록

| # | 주제 | 문제 요약 |
|---|------|----------|
| 1 | 편향 역할 | 퍼셉트론에서 편향(bias)의 주된 역할은? |
| 2 | 수치 계산 | w=[0.5,0.5], b=-0.7, x=[1,1] 일 때 출력 y는? |
| 3 | 학습 규칙 | 가중치가 업데이트되는 조건은? |
| 4 | XOR 한계 | 단층 퍼셉트론으로 학습할 수 없는 게이트는? |
| 5 | 수렴 정리 | 퍼셉트론 수렴 정리(Convergence Theorem)에 따르면? |

---

## 8. 시그널/슬롯 연결 설계

| 발신 시그널 | 수신 슬롯 | 설명 |
|------------|---------|------|
| `StructureTab.weights_changed(w1, w2, b)` | `BoundaryTab.update_weights` | 구조 탭 슬라이더 → 경계 탭 그래프 동기화 |
| `w1_slider.valueChanged` | `NeuronDiagramWidget.set_weights` | 슬라이더 → 다이어그램 실시간 반영 |
| `input_btn.clicked` | `StructureTab._update_result` | 입력값 → 계산 결과 패널 갱신 |
| `gate_combo.currentTextChanged` | `LearningTab._on_gate_change` | 게이트 변경 → 퍼셉트론 초기화 |
| `train_btn.clicked` | `QTimer.start` | 학습 시작 |
| `stop_btn.clicked` | `QTimer.stop` | 학습 중지 |
| `reset_btn.clicked` | `Perceptron.reset` | 가중치 초기화 |
| `QuizTab.answer_submitted` | `QuizTab._show_feedback` | 답 제출 → 피드백 표시 |

---

## 9. 애니메이션 타이머 설계

```python
# LearningTab 내부

self.timer = QTimer()
self.timer.timeout.connect(self._step)   # 타이머 틱 → 1에포크 학습

def _step(self):
    """에포크 1회 실행 + UI 전체 업데이트"""
    errors = self.perceptron.train_epoch(X, y)
    self.epoch += 1
    self.error_history.append(errors)

    # UI 업데이트
    self._update_status_labels()
    self.canvas.draw_error_curve(self.error_history)
    self._append_log(...)

    # 수렴 또는 최대 에포크 도달 시 정지
    if errors == 0 or self.epoch >= MAX_EPOCHS:
        self.timer.stop()
        self._on_training_done(converged=(errors == 0))
```

---

## 10. 다크 모드 테마 명세 (`styles/theme.py`)

팔레트: **Catppuccin Mocha** 기반

| 토큰 | 색상 코드 | 용도 |
|------|---------|------|
| Base | `#1e1e2e` | 전체 배경 |
| Mantle | `#181825` | 카드/패널 배경 |
| Surface0 | `#313244` | 입력 필드, 슬라이더 트랙 |
| Overlay0 | `#6c7086` | 비활성 텍스트 |
| Text | `#cdd6f4` | 기본 텍스트 |
| Blue | `#89b4fa` | 강조, 버튼, 탭 선택, 양수 가중치 엣지 |
| Red | `#f38ba8` | 오류, 음수 가중치 엣지, 위험 버튼 |
| Green | `#a6e3a1` | 성공, 출력 y=1, 수식 텍스트 |
| Yellow | `#f9e2af` | 편향 노드, 경고 |
| Mauve | `#cba6f7` | 활성화 함수 노드 |

---

## 11. 탭별 레이아웃 설계

### Tab 2 레이아웃 (QHBoxLayout)

```
┌────────────────────────────┬─────────────────────┐
│  NeuronDiagramWidget       │  제어 패널           │
│  (QWidget, QPainter)       │                     │
│  stretch=3                 │  [슬라이더 그룹]     │
│                            │  w₁: ─●──────       │
│  입력 → 가중합 → f → 출력  │  w₂: ──────●─       │
│  (실시간 업데이트)          │  b:  ────●───       │
│                            │                     │
│                            │  [입력값 그룹]       │
│                            │  x₁: [0] [1]        │
│                            │  x₂: [0] [1]        │
│                            │                     │
│                            │  [계산 결과 패널]    │
│                            │  z = ...            │
│                            │  y = 1              │
│                            │                     │
│                            │  [초기화 버튼]       │
└────────────────────────────┴─────────────────────┘
```

### Tab 3 레이아웃 (QHBoxLayout)

```
┌─────────────────┬──────────────────────────────────┐
│  제어 패널       │  Matplotlib 오류율 그래프          │
│  stretch=2      │  (FigureCanvasQTAgg)              │
│                 │  stretch=3                        │
│  [게이트 선택]   │                                   │
│  [학습률]       ├──────────────────────────────────┤
│  [속도]         │  학습 로그 (QTextEdit, read-only)  │
│  [상태 표시]    │  Epoch 1: w=[0.1, 0.1] errors=2   │
│                 │  Epoch 2: w=[0.2, 0.2] errors=1   │
│  [▶시작][⏭1스텝]│  ...                              │
│  [↺초기화]      │                                   │
└─────────────────┴──────────────────────────────────┘
```

### Tab 4 레이아웃 (QHBoxLayout)

```
┌────────────────────────────┬─────────────────────┐
│  Matplotlib 결정 경계 그래프 │  제어 패널           │
│  (FigureCanvasQTAgg)       │                     │
│  stretch=3                 │  [게이트 라디오]     │
│                            │  ◉ AND ○ OR ○ XOR   │
│  ●(1,1) → y=1              │                     │
│  ■(0,0) → y=0              │  [가중치 슬라이더]   │
│  ──── 결정경계선 ────       │  w₁: ─●──────       │
│                            │  w₂: ──────●─       │
│                            │  b:  ────●───       │
│                            │                     │
│                            │  [분리 가능 여부]    │
│                            │  ✅ 선형 분리 가능   │
│                            │  (XOR 시: ⚠️ 불가)   │
└────────────────────────────┴─────────────────────┘
```

---

## 12. 실행 및 설치 방법

```bash
# 1. 저장소 클론 또는 파일 다운로드
cd perceptron_app

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python main.py
```

---

## 13. 미결 사항 (Open Issues)

| # | 항목 | 내용 | 우선순위 |
|---|------|------|---------|
| 1 | 한국어 폰트 | 시스템 폰트 의존 vs 폰트 번들 포함 | 중간 |
| 2 | 앱 아이콘 | QApplication 아이콘 이미지 필요 | 낮음 |
| 3 | 패키징 | PyInstaller로 단일 실행파일 빌드 여부 | 낮음 |
| 4 | Tab 2↔Tab 4 연동 | 구조 탭 슬라이더가 경계 탭에 실시간 반영 여부 | 높음 |

---

## 14. 참고 문서

- PySide6 공식 문서: https://doc.qt.io/qtforpython-6/
- QPainter 가이드: https://doc.qt.io/qt-6/qpainter.html
- Matplotlib Qt 임베드: https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
- NumPy 공식 문서: https://numpy.org/doc/
