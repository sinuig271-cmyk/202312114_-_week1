# TRD: Universal Approximation Theorem 시각화 GUI

> **Technical Requirements Document**
> Version: 1.0.0
> Last Updated: 2026-03-25
> Status: Draft
> 참조 PRD: PRD_uat_gui.md v1.0.0

---

## 1. 기술 스택 (Tech Stack)

| 항목 | 선택 | 버전 | 비고 |
|------|------|------|------|
| **언어** | Python | 3.10+ | |
| **GUI 프레임워크** | PySide6 | 6.6+ | Qt 6 공식 Python 바인딩 |
| **수치 연산 / MLP** | NumPy | 1.26+ | 유일한 연산 라이브러리 |
| **그래프 렌더링** | Matplotlib | 3.8+ | Qt6Agg 백엔드, FigureCanvas |
| **수식 렌더링** | Matplotlib `mathtext` → PNG → QLabel | — | LaTeX 스타일 수식 |
| **멀티스레딩** | PySide6 `QThread` + `Signal` | — | 탭 3 MLP 학습 분리 |
| **의존성 관리** | pip / requirements.txt | — | |

---

## 2. 프로젝트 구조 (Project Structure)

```
uat_gui/
│
├── main.py                            # 진입점 (QApplication 실행)
│
├── ui/
│   ├── main_window.py                 # QMainWindow: 탭 구성, 툴바, 언어 전환
│   ├── toolbar.py                     # 앱 제목, KO/EN 버튼
│   │
│   ├── tab_theory/
│   │   ├── theory_tab.py              # 탭 1 루트 위젯 (사이드바 + 콘텐츠)
│   │   ├── section_sidebar.py         # QListWidget 섹션 목차
│   │   └── content_area.py            # 섹션별 텍스트/수식 스크롤 영역
│   │
│   ├── tab_slider/
│   │   ├── slider_tab.py              # 탭 2 루트 위젯
│   │   ├── function_selector.py       # 목표 함수 / 활성화 함수 / x범위 설정 바
│   │   ├── approx_chart.py            # Matplotlib 메인 근사 그래프 캔버스
│   │   ├── basis_panel.py             # 기저 함수 오버레이 패널
│   │   ├── quality_panel.py           # MSE / Max Error / 등급 표시 패널
│   │   └── neuron_slider.py           # 뉴런 수 QSlider + 수치 레이블
│   │
│   └── tab_train/
│       ├── train_tab.py               # 탭 3 루트 위젯
│       ├── hyperparam_panel.py        # 하이퍼파라미터 설정 패널
│       ├── train_approx_chart.py      # 실시간 근사 그래프 캔버스
│       ├── loss_chart.py              # 실시간 Loss 곡선 캔버스
│       └── train_control_bar.py       # ▶⏸⏹⏩ 버튼 + Epoch/Loss 표시
│
├── core/
│   ├── approximator.py                # 탭 2용 최소 제곱 기반 이상적 근사기
│   ├── mlp.py                         # 탭 3용 NumPy MLP (학습 포함)
│   ├── activations.py                 # Activation 함수 및 미분
│   ├── target_functions.py            # 목표 함수 정의 및 팩토리
│   └── formula_sandbox.py             # 사용자 정의 수식 안전 평가
│
├── engine/
│   └── train_worker.py                # QThread 기반 MLP 학습 워커
│
├── content/
│   ├── theory_ko.json                 # 탭 1 한국어 이론 콘텐츠
│   └── theory_en.json                 # 탭 1 영어 이론 콘텐츠
│
├── i18n/
│   ├── ko.json                        # UI 레이블 한국어
│   └── en.json                        # UI 레이블 영어
│
├── assets/
│   └── icons/
│
├── tests/
│   ├── test_approximator.py
│   ├── test_mlp.py
│   └── test_target_functions.py
│
└── requirements.txt
```

---

## 3. 핵심 모듈 상세 설계

### 3.1 `core/target_functions.py` — 목표 함수

```python
import numpy as np
from typing import Callable

TARGET_FUNCTIONS: dict[str, dict] = {
    "sin(x)": {
        "fn": np.sin,
        "x_range": (-np.pi, np.pi),
        "label_ko": "sin(x)",
        "label_en": "sin(x)",
    },
    "x²": {
        "fn": lambda x: x ** 2,
        "x_range": (-2.0, 2.0),
        "label_ko": "x²",
        "label_en": "x²",
    },
    "|x|": {
        "fn": np.abs,
        "x_range": (-2.0, 2.0),
        "label_ko": "|x|",
        "label_en": "|x|",
    },
    "step(x)": {
        "fn": lambda x: np.where(x >= 0, 1.0, 0.0),
        "x_range": (-2.0, 2.0),
        "label_ko": "계단 함수",
        "label_en": "Step Function",
    },
    "custom": {
        "fn": None,          # FormulaSandbox에서 동적 생성
        "x_range": (-2.0, 2.0),
        "label_ko": "사용자 정의",
        "label_en": "Custom",
    },
}

def get_target_fn(name: str) -> Callable:
    return TARGET_FUNCTIONS[name]["fn"]

def get_x_range(name: str) -> tuple[float, float]:
    return TARGET_FUNCTIONS[name]["x_range"]
```

---

### 3.2 `core/approximator.py` — 슬라이더 모드 이상적 근사기 (핵심)

탭 2의 슬라이더 모드는 학습 없이 **최소 제곱법(Least Squares)**으로 출력층 가중치를 해석적으로 계산한다.

```python
import numpy as np
from core.activations import apply_activation

class IdealApproximator:
    """
    단일 은닉층 MLP의 이상적 근사.
    입력 가중치는 균등 분포로 고정 배치,
    출력 가중치는 lstsq로 해석적 계산.
    """
    N_POINTS = 300   # x 샘플링 해상도

    def __init__(self):
        self.x: np.ndarray = np.array([])
        self.y_target: np.ndarray = np.array([])
        self.y_approx: np.ndarray = np.array([])
        self.basis_outputs: np.ndarray = np.array([])  # shape (N_POINTS, n_neurons)
        self.output_weights: np.ndarray = np.array([])

    def compute(
        self,
        target_fn,
        x_range: tuple[float, float],
        n_neurons: int,
        activation: str,
    ) -> "IdealApproximator":
        x_min, x_max = x_range
        self.x = np.linspace(x_min, x_max, self.N_POINTS)
        self.y_target = target_fn(self.x)

        # 입력 가중치: [-3, 3] 균등 배치 (뉴런마다 다른 "중심" 담당)
        centers = np.linspace(x_min, x_max, n_neurons)
        scale = (x_max - x_min) / n_neurons * 3.0

        # 은닉층 출력: shape (N_POINTS, n_neurons)
        Z = np.outer(self.x, np.ones(n_neurons))  # broadcast
        Z = (Z - centers) * scale                  # 각 뉴런별 이동/스케일
        H = apply_activation(Z, activation)        # shape (N_POINTS, n_neurons)
        self.basis_outputs = H

        # 출력 가중치: 최소 제곱 풀이
        # H @ w ≈ y_target → w = lstsq(H, y_target)
        H_bias = np.hstack([H, np.ones((self.N_POINTS, 1))])  # bias 추가
        result = np.linalg.lstsq(H_bias, self.y_target, rcond=None)
        w = result[0]
        self.output_weights = w[:-1]
        bias = w[-1]
        self.y_approx = H @ self.output_weights + bias
        return self

    @property
    def mse(self) -> float:
        return float(np.mean((self.y_target - self.y_approx) ** 2))

    @property
    def max_error(self) -> float:
        return float(np.max(np.abs(self.y_target - self.y_approx)))

    @property
    def quality_grade(self) -> str:
        mse = self.mse
        if mse < 0.001:   return "Excellent"
        elif mse < 0.01:  return "Good"
        elif mse < 0.05:  return "Fair"
        else:             return "Poor"
```

---

### 3.3 `core/mlp.py` — 탭 3 MLP (NumPy 전용)

```python
import numpy as np
from core.activations import apply_activation, apply_activation_derivative
from core.initializers import initialize_weights

class MLP:
    """
    단일 은닉층 MLP for 함수 근사 (1D → 1D).
    architecture: [1, n_hidden, 1]
    """
    def __init__(
        self,
        n_hidden: int,
        activation: str = "sigmoid",
        lr: float = 0.01,
        init_method: str = "xavier",
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.lr = lr

        self.W1 = initialize_weights(1, n_hidden, init_method)   # (1, n_hidden)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = initialize_weights(n_hidden, 1, init_method)   # (n_hidden, 1)
        self.b2 = np.zeros((1, 1))

        self.loss_history: list[float] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (N, 1)
        self.z1 = X @ self.W1 + self.b1           # (N, n_hidden)
        self.a1 = apply_activation(self.z1, self.activation)
        self.z2 = self.a1 @ self.W2 + self.b2     # (N, 1)
        self.a2 = self.z2                          # 출력층 Linear
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        N = X.shape[0]
        # 출력층 gradient (linear activation → grad = 1)
        dL_da2 = 2 * (self.a2 - y) / N            # (N, 1)
        dW2 = self.a1.T @ dL_da2                  # (n_hidden, 1)
        db2 = np.sum(dL_da2, axis=0, keepdims=True)

        # 은닉층 gradient
        da1 = dL_da2 @ self.W2.T                  # (N, n_hidden)
        dz1 = da1 * apply_activation_derivative(self.z1, self.activation)
        dW1 = X.T @ dz1                            # (1, n_hidden)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 가중치 업데이트
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        self.forward(X)
        loss = float(np.mean((self.a2 - y) ** 2))
        self.backward(X, y)
        self.loss_history.append(loss)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = apply_activation(z1, self.activation)
        return a1 @ self.W2 + self.b2

    def reset(self, init_method: str = "xavier"):
        self.W1 = initialize_weights(1, self.n_hidden, init_method)
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = initialize_weights(self.n_hidden, 1, init_method)
        self.b2 = np.zeros((1, 1))
        self.loss_history.clear()
```

---

### 3.4 `core/activations.py`

```python
import numpy as np

def apply_activation(z: np.ndarray, name: str) -> np.ndarray:
    match name:
        case "sigmoid": return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        case "relu":    return np.maximum(0, z)
        case "tanh":    return np.tanh(z)
        case "gelu":
            return z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715*z**3)))
        case _: raise ValueError(f"Unknown: {name}")

def apply_activation_derivative(z: np.ndarray, name: str) -> np.ndarray:
    match name:
        case "sigmoid":
            s = apply_activation(z, "sigmoid")
            return s * (1 - s)
        case "relu":  return np.where(z > 0, 1.0, 0.0)
        case "tanh":  return 1 - np.tanh(z) ** 2
        case "gelu":
            h = 1e-5
            return (apply_activation(z+h, "gelu") - apply_activation(z-h, "gelu")) / (2*h)
        case _: raise ValueError(f"Unknown: {name}")
```

---

### 3.5 `core/formula_sandbox.py` — 사용자 정의 수식 안전 평가

```python
import numpy as np
import ast

ALLOWED_NAMES = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "pi": np.pi, "e": np.e,
    "x": None,   # 호출 시 실제 x 배열로 대체
}
BLOCKED_KEYWORDS = ["import", "exec", "eval", "open", "__", "os", "sys"]

def safe_eval_formula(expr: str, x: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """
    Returns (result_array, None) on success,
            (None, error_message) on failure.
    """
    for kw in BLOCKED_KEYWORDS:
        if kw in expr:
            return None, f"허용되지 않는 키워드: '{kw}'"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return None, f"수식 문법 오류: {e}"
    try:
        ns = {**ALLOWED_NAMES, "x": x}
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, ns)
        return np.asarray(result, dtype=float), None
    except Exception as e:
        return None, f"계산 오류: {e}"
```

---

## 4. UI 모듈 상세 설계

### 4.1 `ui/tab_slider/approx_chart.py` — 메인 근사 그래프

```python
class ApproxChart(FigureCanvas):
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        super().__init__(self.fig)

    def update(self, approx: IdealApproximator, show_basis: bool = True):
        self.ax.clear()
        x = approx.x

        # 오차 영역 음영
        self.ax.fill_between(x, approx.y_target, approx.y_approx,
                             alpha=0.2, color="#FCA5A5", label="오차 영역")

        # 기저 함수 (뉴런별 기여)
        if show_basis:
            for i in range(approx.basis_outputs.shape[1]):
                scaled = approx.basis_outputs[:, i] * approx.output_weights[i]
                self.ax.plot(x, scaled, alpha=0.25, color="#86EFAC", linewidth=0.8)

        # 목표 함수
        self.ax.plot(x, approx.y_target, color="#1E293B",
                     linestyle="--", linewidth=2, label="목표 함수")

        # 근사 함수
        self.ax.plot(x, approx.y_approx, color="#2563EB",
                     linewidth=2, label=f"근사 함수 ({approx.basis_outputs.shape[1]}뉴런)")

        self.ax.legend(fontsize=8)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True, alpha=0.3)
        self.draw_idle()
```

### 4.2 `ui/tab_slider/neuron_slider.py` — 뉴런 수 슬라이더

```python
class NeuronSlider(QWidget):
    neuronCountChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 64)
        self.slider.setValue(4)
        self.label = QLabel("뉴런 수: 4")

        # 디바운싱: 50ms 내 추가 변경 시 타이머 재시작
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._emit)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, val: int):
        self.label.setText(f"뉴런 수: {val}")
        self._pending = val
        self._timer.start()

    def _emit(self):
        self.neuronCountChanged.emit(self._pending)
```

### 4.3 `ui/tab_slider/quality_panel.py` — 근사 품질 패널

```python
GRADE_COLORS = {
    "Excellent": "#16A34A",
    "Good":      "#CA8A04",
    "Fair":      "#EA580C",
    "Poor":      "#DC2626",
}

class QualityPanel(QWidget):
    def update(self, approx: IdealApproximator):
        self.mse_label.setText(f"MSE: {approx.mse:.5f}")
        self.max_err_label.setText(f"Max Error: {approx.max_error:.4f}")
        grade = approx.quality_grade
        self.grade_label.setText(grade)
        self.grade_label.setStyleSheet(
            f"color: {GRADE_COLORS[grade]}; font-weight: bold;"
        )
```

---

## 5. 학습 워커 (`engine/train_worker.py`)

```python
class TrainWorker(QThread):
    epoch_updated  = Signal(int, float, np.ndarray, np.ndarray)
    # epoch, loss, x_plot, y_pred

    training_finished = Signal(float, list)

    def __init__(self, mlp: MLP, X: np.ndarray, y: np.ndarray,
                 epochs: int, x_plot: np.ndarray, update_interval: int = 20):
        super().__init__()
        self.mlp = mlp
        self.X = X
        self.y = y
        self.epochs = epochs
        self.x_plot = x_plot         # 시각화용 조밀한 x 배열
        self.update_interval = update_interval
        self._paused = False
        self._stopped = False

    def run(self):
        for epoch in range(1, self.epochs + 1):
            while self._paused:
                self.msleep(30)
            if self._stopped:
                break

            loss = self.mlp.train_step(self.X, self.y)

            if epoch % self.update_interval == 0:
                y_pred = self.mlp.predict(self.x_plot.reshape(-1, 1)).ravel()
                self.epoch_updated.emit(epoch, loss,
                                        self.x_plot.copy(), y_pred)

        self.training_finished.emit(
            self.mlp.loss_history[-1] if self.mlp.loss_history else 0.0,
            self.mlp.loss_history.copy(),
        )

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._stopped = True
```

---

## 6. 시그널/슬롯 흐름 (Signal Flow)

```
── 탭 2 (슬라이더 모드) ───────────────────────────────────────────

[NeuronSlider]
  └─ Signal: neuronCountChanged(n: int)
       └─ SliderTab.on_neuron_changed()
            ├─ IdealApproximator.compute(target_fn, x_range, n, activation)
            ├─ ApproxChart.update(approx)
            └─ QualityPanel.update(approx)

[FunctionSelector]
  └─ Signal: targetFunctionChanged(name: str)
  └─ Signal: activationChanged(name: str)
  └─ Signal: xRangeChanged(x_min, x_max)
       └─ SliderTab.on_settings_changed()
            └─ (동일하게 IdealApproximator 재계산)

── 탭 3 (MLP 학습) ────────────────────────────────────────────────

[HyperparamPanel]
  └─ Signal: configChanged(config: dict)
       └─ TrainTab.on_config_changed()
            └─ MLP 재구성

[TrainControlBar]
  └─ Signal: startRequested / pauseRequested / stopRequested
       └─ TrainTab.on_start() / pause() / stop()
            └─ TrainWorker 생성·제어

[TrainWorker]
  └─ Signal: epoch_updated(epoch, loss, x_plot, y_pred)
       └─ TrainTab.on_epoch_updated()
            ├─ TrainApproxChart.update_approx(x_plot, y_pred)
            ├─ LossChart.append(epoch, loss)
            └─ TrainControlBar.set_status(epoch, loss)

── 공통 ────────────────────────────────────────────────────────────

[Toolbar]
  └─ Signal: languageToggled(lang: str)
       └─ MainWindow.on_language_change()
            └─ LanguageManager.load(lang) → retranslateUi()

[탭 2 ↔ 탭 3 설정 동기화]
  └─ 탭 변경 시 FunctionSelector 선택값을 HyperparamPanel 목표 함수에 반영
```

---

## 7. 국제화 (i18n) 설계

### 7.1 UI 레이블 (`i18n/ko.json`)

```json
{
  "app_title": "UAT 탐색기",
  "tab_theory": "UAT 이론",
  "tab_slider": "근사 시각화",
  "tab_train": "MLP 학습",
  "lang_toggle": "EN",
  "label_target_fn": "목표 함수",
  "label_activation": "활성화 함수",
  "label_x_range": "x 범위",
  "label_neuron_count": "뉴런 수",
  "label_mse": "MSE",
  "label_max_error": "최대 오차",
  "label_quality": "근사 품질",
  "grade_excellent": "우수",
  "grade_good": "양호",
  "grade_fair": "보통",
  "grade_poor": "미흡",
  "btn_start": "▶ 학습 시작",
  "btn_pause": "⏸ 일시 정지",
  "btn_stop": "⏹ 초기화",
  "btn_fast": "⏩ 빠른 학습",
  "label_lr": "학습률",
  "label_epochs": "에폭 수",
  "label_init": "가중치 초기화",
  "label_custom_fn": "수식 입력 (x 사용)",
  "error_invalid_formula": "수식 오류: {msg}",
  "fn_sin": "sin(x)",
  "fn_square": "x²",
  "fn_abs": "|x|",
  "fn_step": "계단 함수",
  "fn_custom": "사용자 정의"
}
```

---

## 8. 성능 최적화 전략

| 요구사항 | 구현 방법 |
|----------|-----------|
| 슬라이더 ≤ 100ms 갱신 | `QTimer` 50ms 디바운싱 → `IdealApproximator.compute()` (NumPy 벡터 연산, N=300 포인트) |
| `lstsq` 연산 최적화 | NumPy `np.linalg.lstsq` LAPACK 기반, n_neurons ≤ 64에서 충분히 빠름 |
| Matplotlib 깜박임 방지 | `draw_idle()` 사용, `blit` 적용 가능 시 적용 |
| 탭 3 UI 블로킹 방지 | `QThread` + 20 epoch마다 Signal emit |
| 빠른 학습 모드 | `update_interval = epochs`로 설정, 마지막 결과만 emit |
| 기저 함수 렌더링 최적화 | 뉴런 수 > 16: 대표 10개만 렌더링, 나머지 생략 |

---

## 9. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| 사용자 정의 수식 오류 | `formula_sandbox.py`에서 AST 파싱 → 오류 메시지를 `QLabel`에 빨간색 표시, 이전 함수 유지 |
| `lstsq` 수치 불안정 (n_neurons=1, 계단 함수) | `rcond=None`으로 안전 처리, 결과가 NaN이면 fallback (y_approx = 0) |
| MLP NaN 발생 (Gradient Exploding) | `np.isnan` 감지 → 학습 중단, 학습률 낮추기 안내 |
| x 범위 역전 입력 (min ≥ max) | `QDoubleSpinBox` 연동으로 min < max 보장 |
| 언어 파일 누락 | `"ko"` fallback |

---

## 10. 테스트 계획

### 10.1 단위 테스트

```python
# test_approximator.py
def test_sin_approximation_improves_with_neurons():
    """뉴런 수 증가 → MSE 감소 경향 검증"""
    from core.target_functions import TARGET_FUNCTIONS
    fn = TARGET_FUNCTIONS["sin(x)"]["fn"]
    x_range = TARGET_FUNCTIONS["sin(x)"]["x_range"]
    approx = IdealApproximator()

    mse_4  = approx.compute(fn, x_range, n_neurons=4,  activation="sigmoid").mse
    mse_16 = approx.compute(fn, x_range, n_neurons=16, activation="sigmoid").mse
    mse_32 = approx.compute(fn, x_range, n_neurons=32, activation="sigmoid").mse
    assert mse_4 > mse_16 > mse_32, "뉴런 수 증가 시 MSE가 감소해야 한다"

def test_quality_grade_excellent():
    """MSE < 0.001 → Excellent 등급"""
    approx = IdealApproximator()
    approx.y_target = np.zeros(300)
    approx.y_approx = np.zeros(300)
    assert approx.quality_grade == "Excellent"

# test_formula_sandbox.py
def test_valid_formula():
    x = np.linspace(-1, 1, 10)
    result, err = safe_eval_formula("sin(x) + x**2", x)
    assert err is None
    assert result is not None

def test_blocked_keyword():
    x = np.array([1.0])
    result, err = safe_eval_formula("__import__('os')", x)
    assert result is None
    assert err is not None

# test_mlp.py
def test_sin_convergence():
    """sin(x) 학습 후 MSE < 0.01 검증"""
    x_train = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    y_train = np.sin(x_train)
    mlp = MLP(n_hidden=16, activation="sigmoid", lr=0.05, init_method="xavier")
    for _ in range(3000):
        mlp.train_step(x_train, y_train)
    y_pred = mlp.predict(x_train)
    mse = float(np.mean((y_pred - y_train) ** 2))
    assert mse < 0.01
```

### 10.2 통합 테스트

| 시나리오 | 검증 내용 |
|----------|-----------|
| 뉴런 슬라이더 1→64 드래그 | MSE가 단조 감소하지 않을 수 있으나 전반적으로 감소 경향 |
| 목표 함수 전환 | 그래프·품질 지표 즉시 갱신 |
| 활성화 함수 전환 | 동일 뉴런 수에서 근사 곡선 변화 |
| 사용자 수식 입력 오류 | 오류 메시지 표시, 이전 함수 유지 |
| 탭 3 학습 → 일시정지 → 재개 | Loss 연속성 유지 |
| 언어 전환 (탭 1·2·3) | 전체 UI 텍스트 변경, 수식 유지 |

---

## 11. 의존성 (`requirements.txt`)

```
PySide6>=6.6.0
matplotlib>=3.8.0
numpy>=1.26.0
```

> MLP 연산 및 근사 계산에 NumPy 외 ML 라이브러리를 사용하지 않는다.

---

## 12. 개발 환경 설정

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
python -m pytest tests/ -v
```

---

## 13. 향후 기술 고려사항 (v2.0)

| 기능 | 기술 방안 |
|------|-----------|
| 2D 함수 근사 시각화 | Matplotlib 3D `Axes3D`, 입력 2개 → 출력 1개 곡면 |
| 뉴런 수 자동 재생 (GIF) | `FuncAnimation` + `PillowWriter` |
| 두 활성화 함수 동시 비교 | `plt.subplots(1,2)` 또는 별도 FigureCanvas 2개 배치 |
| 그래프 PNG 저장 | `fig.savefig()` + `QFileDialog` |

---

*이 문서는 PRD v1.0.0 기반으로 작성되었으며, 구현 착수 전 기술 검토가 필요합니다.*
