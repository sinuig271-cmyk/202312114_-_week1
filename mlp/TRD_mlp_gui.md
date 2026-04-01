# TRD: MLP 개념 및 NumPy 구현 시각화 GUI

> **Technical Requirements Document**
> Version: 1.0.0
> Last Updated: 2026-03-25
> Status: Draft
> 참조 PRD: PRD_mlp_gui.md v1.0.0

---

## 1. 기술 스택 (Tech Stack)

| 항목 | 선택 | 버전 | 비고 |
|------|------|------|------|
| **언어** | Python | 3.10+ | |
| **GUI 프레임워크** | PySide6 | 6.6+ | Qt 6 공식 Python 바인딩 |
| **MLP 연산** | NumPy | 1.26+ | **유일한 연산 라이브러리** |
| **그래프/시각화** | Matplotlib | 3.8+ | Loss 곡선, 가중치 Heatmap, 결정 경계 |
| **네트워크 다이어그램** | PySide6 `QGraphicsScene` | — | 커스텀 노드·엣지 렌더링 |
| **코드 하이라이팅** | 커스텀 `QSyntaxHighlighter` | — | Python/NumPy 구문 강조 |
| **수식 렌더링** | Matplotlib `mathtext` → PNG → `QLabel` | — | LaTeX 스타일 수식 표시 |
| **멀티스레딩** | PySide6 `QThread` + `Signal` | — | 학습 루프 UI 블로킹 방지 |
| **의존성 관리** | pip / requirements.txt | — | |

---

## 2. 프로젝트 구조 (Project Structure)

```
mlp_gui/
│
├── main.py                            # 진입점 (QApplication 실행)
│
├── ui/
│   ├── main_window.py                 # QMainWindow: 탭 구성, 툴바, 언어 전환
│   ├── toolbar.py                     # KO/EN 언어 버튼, 앱 제목
│   │
│   ├── tab_concept/
│   │   ├── concept_tab.py             # 탭 1 루트 위젯 (사이드바 + 콘텐츠 영역)
│   │   ├── section_sidebar.py         # 섹션 목차 QListWidget
│   │   ├── content_area.py            # 섹션별 텍스트/수식/다이어그램 스크롤 영역
│   │   └── concept_diagram.py         # QGraphicsView 미니 다이어그램
│   │
│   ├── tab_code/
│   │   ├── code_tab.py                # 탭 2 루트 위젯
│   │   ├── code_viewer.py             # QTextEdit + 커스텀 SyntaxHighlighter
│   │   ├── formula_viewer.py          # 수식 + 설명 텍스트 우측 패널
│   │   └── step_navigator.py          # ◀ 이전 / ▶ 다음 / Progress Bar
│   │
│   └── tab_train/
│       ├── train_tab.py               # 탭 3 루트 위젯
│       ├── hyperparam_panel.py        # 하이퍼파라미터 설정 패널
│       ├── network_diagram.py         # QGraphicsScene FC 다이어그램 (실시간)
│       ├── loss_chart.py              # Matplotlib Loss 곡선 FigureCanvas
│       ├── weight_heatmap.py          # Matplotlib 가중치 Heatmap FigureCanvas
│       ├── decision_boundary.py       # Matplotlib 결정 경계 FigureCanvas
│       ├── train_control_bar.py       # ▶⏸⏹⏩ 제어 버튼 + 진행 상태
│       └── result_summary.py          # 학습 결과 요약 패널
│
├── core/
│   ├── mlp.py                         # MLP 클래스 (NumPy 전용 구현)
│   ├── activations.py                 # Activation 함수 및 미분
│   ├── losses.py                      # Loss 함수 (MSE, BCE)
│   ├── initializers.py                # 가중치 초기화 (Random, Xavier, He)
│   └── xor_dataset.py                 # XOR 데이터셋 생성
│
├── engine/
│   └── train_worker.py                # QThread 기반 학습 워커
│
├── content/
│   ├── concept_ko.json                # 탭 1 한국어 콘텐츠 (섹션별 텍스트)
│   ├── concept_en.json                # 탭 1 영어 콘텐츠
│   ├── code_steps_ko.json             # 탭 2 한국어 코드+설명
│   └── code_steps_en.json             # 탭 2 영어 코드+설명
│
├── i18n/
│   ├── ko.json                        # UI 레이블 한국어
│   └── en.json                        # UI 레이블 영어
│
├── assets/
│   └── icons/                         # 툴바 아이콘
│
├── tests/
│   ├── test_mlp.py
│   ├── test_activations.py
│   └── test_losses.py
│
└── requirements.txt
```

---

## 3. MLP 핵심 구현 (`core/mlp.py`)

MLP 연산은 **NumPy만** 사용하며, 학습 과정에서 각 단계 결과를 외부에서 접근 가능하도록 저장한다.

### 3.1 MLP 클래스 전체 설계

```python
import numpy as np
from core.activations import apply_activation, apply_activation_derivative
from core.initializers import initialize_weights

class MLP:
    def __init__(
        self,
        layer_sizes: list[int],      # 예: [2, 4, 1] → 입력2, 은닉4, 출력1
        activation: str = "sigmoid", # "sigmoid" | "relu" | "tanh"
        init_method: str = "xavier", # "random" | "xavier" | "he"
        learning_rate: float = 0.1,
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.lr = learning_rate
        self.n_layers = len(layer_sizes) - 1  # 가중치 레이어 수

        # 가중치 및 편향 초기화
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(self.n_layers):
            self.W.append(initialize_weights(
                layer_sizes[i], layer_sizes[i+1], init_method
            ))
            self.b.append(np.zeros((1, layer_sizes[i+1])))

        # Forward 계산 결과 저장 (시각화용)
        self.z_cache: list[np.ndarray] = []   # 가중합
        self.a_cache: list[np.ndarray] = []   # 활성화값 (입력층 포함)

        # Backward 계산 결과 저장 (시각화용)
        self.dW_cache: list[np.ndarray] = []  # 가중치 gradient
        self.db_cache: list[np.ndarray] = []  # 편향 gradient
        self.delta_cache: list[np.ndarray] = [] # 역전파 delta

        # 학습 이력
        self.loss_history: list[float] = []

    # ── Forward Propagation ──────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (n_samples, n_input)
        returns: 출력층 활성화값 shape (n_samples, n_output)
        """
        self.z_cache.clear()
        self.a_cache.clear()

        a = X
        self.a_cache.append(a)  # 입력층

        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]      # (n, in) @ (in, out) + (1, out)
            a = apply_activation(z, self.activation
                                 if i < self.n_layers - 1
                                 else "sigmoid")  # 출력층은 항상 sigmoid (이진 분류)
            self.z_cache.append(z)
            self.a_cache.append(a)

        return a

    # ── Backpropagation ──────────────────────────────────────────
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        y: shape (n_samples, n_output)
        """
        self.dW_cache.clear()
        self.db_cache.clear()
        self.delta_cache.clear()

        n = X.shape[0]
        output = self.a_cache[-1]

        # 출력층 delta: dL/dz = (a - y) * sigmoid'(z)
        delta = (output - y) * apply_activation_derivative(
            self.z_cache[-1], "sigmoid"
        )
        self.delta_cache.append(delta)

        for i in reversed(range(self.n_layers)):
            dW = self.a_cache[i].T @ delta / n   # (in, n) @ (n, out) / n
            db = np.mean(delta, axis=0, keepdims=True)
            self.dW_cache.insert(0, dW)
            self.db_cache.insert(0, db)

            if i > 0:  # 은닉층 delta 역전파
                delta = (delta @ self.W[i].T) * apply_activation_derivative(
                    self.z_cache[i-1], self.activation
                )
                self.delta_cache.insert(0, delta)

    # ── 가중치 업데이트 ──────────────────────────────────────────
    def update_weights(self) -> None:
        for i in range(self.n_layers):
            self.W[i] -= self.lr * self.dW_cache[i]
            self.b[i] -= self.lr * self.db_cache[i]

    # ── 단일 epoch 학습 ──────────────────────────────────────────
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        output = self.forward(X)
        loss = float(np.mean((output - y) ** 2))   # MSE
        self.backward(X, y)
        self.update_weights()
        self.loss_history.append(loss)
        return loss

    # ── 초기화 ───────────────────────────────────────────────────
    def reset(self, init_method: str = "xavier") -> None:
        for i in range(self.n_layers):
            self.W[i] = initialize_weights(
                self.layer_sizes[i], self.layer_sizes[i+1], init_method
            )
            self.b[i] = np.zeros((1, self.layer_sizes[i+1]))
        self.loss_history.clear()
        self.z_cache.clear()
        self.a_cache.clear()
```

---

## 4. 보조 모듈 설계

### 4.1 `core/activations.py`

```python
import numpy as np

def apply_activation(z: np.ndarray, name: str) -> np.ndarray:
    match name:
        case "sigmoid": return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        case "relu":    return np.maximum(0, z)
        case "tanh":    return np.tanh(z)
        case _:         raise ValueError(f"Unknown activation: {name}")

def apply_activation_derivative(z: np.ndarray, name: str) -> np.ndarray:
    match name:
        case "sigmoid":
            s = apply_activation(z, "sigmoid")
            return s * (1 - s)
        case "relu":    return np.where(z > 0, 1.0, 0.0)
        case "tanh":    return 1 - np.tanh(z) ** 2
        case _:         raise ValueError(f"Unknown activation: {name}")
```

### 4.2 `core/initializers.py`

```python
import numpy as np

def initialize_weights(n_in: int, n_out: int, method: str) -> np.ndarray:
    match method:
        case "random":  return np.random.uniform(-1, 1, (n_in, n_out))
        case "xavier":
            std = np.sqrt(2 / (n_in + n_out))
            return np.random.normal(0, std, (n_in, n_out))
        case "he":
            std = np.sqrt(2 / n_in)
            return np.random.normal(0, std, (n_in, n_out))
        case _: raise ValueError(f"Unknown method: {method}")
```

### 4.3 `core/xor_dataset.py`

```python
import numpy as np

def get_xor_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)  # shape (4,2)
    y = np.array([[0],[1],[1],[0]], dtype=float)           # shape (4,1)
    return X, y
```

---

## 5. 학습 워커 (`engine/train_worker.py`)

MLP 학습 루프는 `QThread`에서 실행하여 UI 블로킹을 방지한다.

```python
from PySide6.QtCore import QThread, Signal
import numpy as np

class TrainWorker(QThread):
    # 매 업데이트 주기마다 emit
    epoch_updated = Signal(int, float, list, list)
    # epoch: int, loss: float, a_cache: list, W_list: list

    training_finished = Signal(float, list)
    # final_loss: float, loss_history: list

    def __init__(self, mlp, X, y, epochs: int, update_interval: int = 10):
        super().__init__()
        self.mlp = mlp
        self.X = X
        self.y = y
        self.epochs = epochs
        self.update_interval = update_interval  # N epoch마다 UI 갱신
        self._paused = False
        self._stopped = False

    def run(self):
        for epoch in range(1, self.epochs + 1):
            while self._paused:
                self.msleep(50)
            if self._stopped:
                break

            loss = self.mlp.train_step(self.X, self.y)

            if epoch % self.update_interval == 0:
                self.epoch_updated.emit(
                    epoch,
                    loss,
                    [a.copy() for a in self.mlp.a_cache],
                    [W.copy() for W in self.mlp.W],
                )

        self.training_finished.emit(
            self.mlp.loss_history[-1] if self.mlp.loss_history else 0.0,
            self.mlp.loss_history.copy(),
        )

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._stopped = True
```

---

## 6. 탭별 UI 상세 설계

### 6.1 탭 1 — 개념 설명 (`tab_concept/`)

#### 콘텐츠 데이터 구조 (`content/concept_ko.json`)

```json
{
  "sections": [
    {
      "id": 1,
      "title": "Perceptron이란?",
      "body": "퍼셉트론은 입력값에 가중치를 곱하고...",
      "formula": "z = w_1x_1 + w_2x_2 + b",
      "diagram_type": "perceptron"
    },
    {
      "id": 5,
      "title": "Forward Propagation",
      "body": "각 레이어에서 z = Wx + b를 계산하고...",
      "formula": "z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}",
      "diagram_type": "forward_flow"
    }
  ]
}
```

#### `content_area.py` 렌더링 방식

- `QScrollArea` 내부에 섹션별 `QWidget` 동적 생성
- 수식: `render_formula_to_pixmap(latex_str)` → Matplotlib mathtext → `QLabel`
- 다이어그램: `diagram_type` 값에 따라 `concept_diagram.py`의 팩토리 함수 호출

```python
def render_formula_to_pixmap(latex: str, fontsize: int = 14) -> QPixmap:
    fig, ax = plt.subplots(figsize=(4, 0.8))
    ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.read())
    return pixmap
```

---

### 6.2 탭 2 — 코드 구현 (`tab_code/`)

#### 코드 단계 데이터 구조 (`content/code_steps_ko.json`)

```json
{
  "steps": [
    {
      "step": 1,
      "title": "데이터 준비",
      "code": "import numpy as np\n\nX = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\ny = np.array([[0],[1],[1],[0]], dtype=float)",
      "formula": "X \\in \\mathbb{R}^{4 \\times 2}, \\quad y \\in \\mathbb{R}^{4 \\times 1}",
      "description": "XOR 문제의 입력 행렬 X와 타깃 벡터 y를 NumPy 배열로 정의합니다."
    },
    {
      "step": 4,
      "title": "Forward Propagation",
      "code": "z1 = X @ W1 + b1\na1 = sigmoid(z1)\nz2 = a1 @ W2 + b2\na2 = sigmoid(z2)",
      "formula": "z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \\quad a^{(l)} = \\sigma(z^{(l)})",
      "description": "행렬 곱으로 가중합 z를 계산하고 활성화 함수를 적용합니다."
    }
  ]
}
```

#### `code_viewer.py` — 구문 강조 하이라이터

```python
class NumPySyntaxHighlighter(QSyntaxHighlighter):
    KEYWORDS = ["import", "def", "return", "for", "in", "range", "if", "else"]
    NUMPY_FN = ["np.array", "np.dot", "np.zeros", "np.random", "np.mean",
                "np.exp", "np.maximum", "np.clip", "np.where"]

    def highlightBlock(self, text: str):
        # 키워드: 파랑 bold
        # NumPy 함수: 초록
        # 문자열: 주황
        # 주석: 회색 italic
        # 숫자: 자주
        ...
```

---

### 6.3 탭 3 — 학습 시각화 (`tab_train/`)

#### `network_diagram.py` — 실시간 네트워크 다이어그램

```python
class NetworkDiagram(QGraphicsView):
    def build(self, layer_sizes: list[int]): ...

    def update_activations(self, a_cache: list[np.ndarray]):
        """Forward 완료 후 각 뉴런 활성화값 색상 업데이트"""
        for layer_idx, a in enumerate(a_cache):
            for neuron_idx, val in enumerate(a[0]):   # 배치 첫 번째 샘플 기준
                item = self._neurons[layer_idx][neuron_idx]
                item.set_activation(float(val))

    def highlight_forward(self, layer_id: int): ...
    def highlight_backward(self, layer_id: int): ...
    def update_weights(self, W_list: list[np.ndarray]): ...
```

#### `loss_chart.py` — 실시간 Loss 곡선

```python
class LossChart(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 2.5))
        super().__init__(self.fig)
        self._epochs: list[int] = []
        self._losses: list[float] = []

    def append(self, epoch: int, loss: float):
        self._epochs.append(epoch)
        self._losses.append(loss)
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.plot(self._epochs, self._losses, color="#EA580C", linewidth=1.5)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss (MSE)")
        self.ax.set_title("학습 곡선")
        self.draw_idle()
```

#### `decision_boundary.py` — XOR 결정 경계

```python
class DecisionBoundaryChart(FigureCanvas):
    XOR_X = np.array([[0,0],[0,1],[1,0],[1,1]])
    XOR_Y = np.array([0,1,1,0])

    def update(self, mlp: MLP):
        xx, yy = np.meshgrid(np.linspace(-0.5,1.5,100), np.linspace(-0.5,1.5,100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = mlp.forward(grid).reshape(xx.shape)
        self.ax.clear()
        self.ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
        self.ax.scatter(self.XOR_X[:,0], self.XOR_X[:,1],
                        c=self.XOR_Y, cmap="bwr", edgecolors="k", zorder=5)
        self.draw_idle()
```

---

## 7. 시그널/슬롯 흐름 (Signal Flow)

```
[HyperparamPanel]
  └─ Signal: configChanged(config: dict)
       └─ TrainTab.on_config_changed()
            ├─ MLP 재구성 (layer_sizes, activation, lr)
            └─ NetworkDiagram.build(layer_sizes)

[TrainControlBar]
  └─ Signal: startRequested
       └─ TrainTab.on_start()
            ├─ TrainWorker 생성 및 시작
            └─ 버튼 상태 전환 (▶ 비활성화, ⏸⏹ 활성화)

  └─ Signal: pauseRequested / resumeRequested
       └─ worker.pause() / worker.resume()

  └─ Signal: stopRequested
       └─ worker.stop() → MLP.reset()

[TrainWorker]
  └─ Signal: epoch_updated(epoch, loss, a_cache, W_list)
       └─ TrainTab.on_epoch_updated()
            ├─ LossChart.append(epoch, loss)
            ├─ NetworkDiagram.update_activations(a_cache)
            ├─ NetworkDiagram.update_weights(W_list)
            ├─ WeightHeatmap.update(W_list)
            └─ DecisionBoundaryChart.update(mlp)

  └─ Signal: training_finished(final_loss, loss_history)
       └─ TrainTab.on_training_finished()
            ├─ ResultSummary.show(mlp, final_loss)
            └─ 버튼 상태 복원

[Toolbar]
  └─ Signal: languageToggled(lang: str)
       └─ MainWindow.on_language_change()
            └─ LanguageManager.load(lang) → retranslateUi() 전파
```

---

## 8. 국제화 (i18n) 설계

### 8.1 UI 레이블 (`i18n/ko.json`)

```json
{
  "app_title": "MLP 탐색기",
  "tab_concept": "개념 설명",
  "tab_code": "NumPy 구현",
  "tab_train": "학습 시각화",
  "btn_start": "▶ 학습 시작",
  "btn_pause": "⏸ 일시 정지",
  "btn_stop": "⏹ 초기화",
  "btn_fast": "⏩ 빠른 학습",
  "label_hidden_layers": "은닉층 수",
  "label_neurons": "뉴런 수",
  "label_lr": "학습률 (η)",
  "label_epochs": "에폭 수",
  "label_activation": "활성화 함수",
  "label_init": "가중치 초기화",
  "btn_copy_code": "복사",
  "btn_copy_all": "전체 코드 복사",
  "step_progress": "{current} / {total} 단계",
  "lang_toggle": "EN",
  "result_loss": "최종 Loss",
  "result_accuracy": "정확도",
  "result_success": "학습 성공 ✓",
  "result_fail": "학습 미완료 — 재시도 권장"
}
```

### 8.2 콘텐츠 언어 분기

- 탭 1·2의 콘텐츠는 `concept_ko.json` / `concept_en.json` 파일로 분리 관리
- `LanguageManager.load(lang)` 호출 시 `content_area.py`와 `code_tab.py`가 해당 언어 JSON 재로드 후 위젯 갱신

---

## 9. 성능 최적화 전략

| 요구사항 | 구현 방법 |
|----------|-----------|
| UI 블로킹 방지 | MLP 학습 루프를 `QThread` (TrainWorker)에서 실행 |
| UI 갱신 주기 제한 | 기본 10 epoch마다 `epoch_updated` Signal emit |
| Matplotlib 깜박임 방지 | `draw_idle()` 사용, `blit` 기법 적용 가능 시 적용 |
| 결정 경계 계산 최적화 | 100×100 그리드 (10,000 포인트) NumPy 배치 연산, `QThread` 내 수행 |
| 빠른 학습 모드 | `update_interval = epochs` 로 설정 → 마지막 epoch 결과만 emit |

---

## 10. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| 가중치 NaN / Inf 발생 (Gradient Exploding) | `np.isnan` / `np.isinf` 감지 → 학습 중단, 경고 메시지, 학습률 낮추기 권고 |
| 학습률 너무 큼 | Loss가 NaN이면 `TrainWorker.stop()` 자동 호출 |
| 잘못된 하이퍼파라미터 입력 | `QSpinBox` / `QDoubleSpinBox` min/max 제한으로 UI에서 사전 차단 |
| 언어 파일 누락 | `"ko"` fallback 후 경고 로그 출력 |
| 학습 중 탭 전환 | `TrainWorker`는 백그라운드에서 계속 실행, 탭 복귀 시 최신 상태 복원 |

---

## 11. 테스트 계획

### 11.1 단위 테스트 (`tests/`)

```python
# test_mlp.py
def test_xor_convergence():
    """XOR 학습 후 모든 샘플 예측 정확도 100% 검증"""
    from core.xor_dataset import get_xor_data
    X, y = get_xor_data()
    mlp = MLP([2, 4, 1], activation="sigmoid", learning_rate=0.5)
    for _ in range(5000):
        mlp.train_step(X, y)
    pred = mlp.forward(X)
    accuracy = np.mean((pred > 0.5) == y)
    assert accuracy == 1.0

def test_forward_cache_shape():
    """forward() 후 z_cache, a_cache shape 검증"""
    mlp = MLP([2, 4, 1])
    X, _ = get_xor_data()
    mlp.forward(X)
    assert mlp.a_cache[0].shape == (4, 2)   # 입력층
    assert mlp.a_cache[1].shape == (4, 4)   # 은닉층
    assert mlp.a_cache[2].shape == (4, 1)   # 출력층

def test_weight_update():
    """train_step 후 가중치가 변경됨을 검증"""
    mlp = MLP([2, 4, 1])
    X, y = get_xor_data()
    W_before = mlp.W[0].copy()
    mlp.train_step(X, y)
    assert not np.allclose(mlp.W[0], W_before)
```

### 11.2 통합 테스트

| 시나리오 | 검증 내용 |
|----------|-----------|
| 탭 전환 | 탭 간 이동 시 UI 상태 유지 |
| 하이퍼파라미터 변경 후 학습 시작 | 새 설정 반영된 MLP로 학습 진행 |
| 일시정지 → 재개 | 재개 후 loss가 정지 전과 연속적 |
| 초기화 버튼 | Loss 그래프·다이어그램·가중치 초기 상태 복원 |
| 언어 전환 (탭 1·2·3 각각) | 모든 텍스트 정상 변경, 수식·코드 유지 |

---

## 12. 의존성 (`requirements.txt`)

```
PySide6>=6.6.0
matplotlib>=3.8.0
numpy>=1.26.0
```

> MLP 연산에는 NumPy 외 어떠한 ML 라이브러리도 사용하지 않는다.

---

## 13. 개발 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 실행
python main.py

# 테스트
python -m pytest tests/ -v
```

---

## 14. 향후 기술 고려사항 (v2.0)

| 기능 | 기술 방안 |
|------|-----------|
| 추가 데이터셋 (사인 회귀, MNIST 서브셋) | `core/datasets.py` 확장, 입력 파이프라인 추상화 |
| GIF 내보내기 | Matplotlib `FuncAnimation` + `PillowWriter` |
| PDF 내보내기 | `reportlab` 또는 Matplotlib PDF 백엔드 |
| 학습 가중치 저장/불러오기 | `np.save()` / `np.load()` + `QFileDialog` |
| Mini-batch SGD | `xor_dataset.py` 배치 샘플링 확장 (XOR은 Full-batch가 적합하나 교육 목적) |

---

*이 문서는 PRD v1.0.0 기반으로 작성되었으며, 구현 착수 전 기술 검토가 필요합니다.*
