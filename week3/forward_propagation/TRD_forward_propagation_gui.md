# TRD: Forward Propagation 시각화 GUI

> **Technical Requirements Document**
> Version: 1.0.0
> Last Updated: 2026-03-25
> Status: Draft
> 참조 PRD: PRD_forward_propagation_gui.md v1.0.0

---

## 1. 기술 스택 (Tech Stack)

| 항목 | 선택 | 버전 | 비고 |
|------|------|------|------|
| **언어** | Python | 3.10+ | |
| **GUI 프레임워크** | PySide6 | 6.6+ | Qt 6 기반 공식 Python 바인딩 |
| **네트워크 다이어그램 렌더링** | PySide6 `QGraphicsScene` / `QGraphicsView` | — | 커스텀 노드·엣지 직접 드로잉 |
| **CNN 특징 맵 시각화** | Matplotlib (Qt6Agg 백엔드) | 3.8+ | Feature Map 그리드 렌더링 |
| **수치 연산** | NumPy | 1.26+ | 행렬 연산, Forward 계산 |
| **이미지 처리 (CNN)** | Pillow | 10.0+ | 입력 이미지 리사이즈 및 배열 변환 |
| **의존성 관리** | pip / requirements.txt | — | |

### 1.1 렌더링 전략 결정 근거

| 컴포넌트 | 렌더링 방법 | 이유 |
|----------|-------------|------|
| FC 네트워크 다이어그램 | `QGraphicsScene` | 노드·엣지 개별 항목 클릭/색상 동적 변경 용이 |
| CNN 특징 맵 | `Matplotlib FigureCanvas` | 픽셀 그리드 시각화에 최적화 |
| 계산 수식 표시 | `QLabel` + HTML | 간단한 인라인 수식 렌더링 |
| 가중치 행렬 뷰 | `QTableWidget` | 행렬 데이터 표현에 적합 |

---

## 2. 프로젝트 구조 (Project Structure)

```
forward_prop_gui/
│
├── main.py                          # 진입점 (QApplication 실행)
│
├── ui/
│   ├── main_window.py               # QMainWindow: 전체 레이아웃, 모드 전환
│   ├── toolbar.py                   # 모드 전환, 프리셋 선택, 언어 버튼
│   │
│   ├── panels/
│   │   ├── config_panel.py          # 레이어/뉴런 수 설정, Activation, 초기화 설정
│   │   ├── input_panel.py           # 입력값 설정 (FC 수치 입력 / CNN 이미지 선택)
│   │   ├── calc_detail_panel.py     # 현재 스텝 수식 + 계산 로그 + 행렬 뷰 탭
│   │   └── step_control_bar.py      # ◀ 이전 / ▶ 다음 / ↺ 초기화 / 자동재생 / Progress Bar
│   │
│   └── canvas/
│       ├── fc_canvas.py             # QGraphicsScene 기반 FC 다이어그램
│       ├── cnn_canvas.py            # Matplotlib 기반 CNN 특징 맵 뷰
│       └── graphics_items.py        # NeuronItem, EdgeItem (QGraphicsItem 서브클래스)
│
├── core/
│   ├── network_model.py             # 네트워크 구조 데이터 모델 (레이어, 가중치, 활성화값)
│   ├── forward_engine.py            # Forward Propagation 단계별 계산 엔진
│   ├── activations.py               # Activation Function 구현 (NumPy 기반)
│   ├── initializers.py              # 가중치 초기화 (Random, Xavier, He)
│   └── cnn_engine.py                # CNN Conv2D / Pooling 연산 구현
│
├── state/
│   └── app_state.py                 # 전역 애플리케이션 상태 관리 (현재 스텝, 모드 등)
│
├── i18n/
│   ├── ko.json                      # 한국어 문자열
│   └── en.json                      # 영어 문자열
│
├── presets/
│   ├── xor_solver.json              # XOR Solver 프리셋
│   ├── simple_classifier.json       # Simple Classifier 프리셋
│   └── mini_cnn.json                # Mini CNN 프리셋
│
├── assets/
│   ├── sample_images/               # CNN용 샘플 이미지 (8×8 ~ 32×32)
│   └── icons/                       # 툴바 아이콘
│
├── tests/
│   ├── test_forward_engine.py
│   ├── test_activations.py
│   └── test_cnn_engine.py
│
└── requirements.txt
```

---

## 3. 데이터 모델 설계

### 3.1 `core/network_model.py`

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class LayerConfig:
    layer_id: int
    layer_type: str          # "dense" | "conv2d" | "pooling" | "flatten"
    n_neurons: int           # Dense 전용
    activation: str          # "relu" | "sigmoid" | "tanh" | "linear" | "softmax"
    use_bias: bool = True
    # Conv2D 전용
    n_filters: int = 0
    kernel_size: tuple = (3, 3)
    stride: int = 1
    padding: str = "valid"   # "same" | "valid"
    # Pooling 전용
    pool_type: str = "max"   # "max" | "average"
    pool_size: tuple = (2, 2)

@dataclass
class NetworkModel:
    mode: str                          # "fc" | "cnn"
    layers: list[LayerConfig] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)   # W per layer
    biases: list[np.ndarray] = field(default_factory=list)    # b per layer
    # 런타임 상태 (Forward 계산 결과)
    z_values: list[np.ndarray] = field(default_factory=list)  # 가중합
    a_values: list[np.ndarray] = field(default_factory=list)  # 활성화값
    input_data: np.ndarray = field(default_factory=lambda: np.array([]))
```

---

## 4. 핵심 모듈 상세 설계

### 4.1 `core/forward_engine.py` — FC Forward 계산 엔진

Step-by-step 진행을 위해 **레이어 단위 제너레이터(Generator)** 방식으로 구현한다.

```python
class FCForwardEngine:
    def __init__(self, model: NetworkModel):
        self.model = model
        self._steps: list[StepResult] = []
        self._current_step: int = -1

    def compute_all_steps(self) -> None:
        """전체 Forward 계산을 미리 수행하고 각 스텝 결과를 저장"""
        self._steps.clear()
        a = self.model.input_data.copy()

        for i, layer in enumerate(self.model.layers):
            W = self.model.weights[i]
            b = self.model.biases[i] if layer.use_bias else 0
            z = W @ a + b
            a_new = apply_activation(z, layer.activation)
            self._steps.append(StepResult(
                layer_id=i,
                z=z.copy(),
                a=a_new.copy(),
                W=W.copy(),
                b=b if layer.use_bias else None,
                prev_a=a.copy(),
                activation=layer.activation,
            ))
            a = a_new

    def step_forward(self) -> StepResult | None:
        if self._current_step < len(self._steps) - 1:
            self._current_step += 1
            return self._steps[self._current_step]
        return None  # 마지막 스텝

    def step_backward(self) -> StepResult | None:
        if self._current_step > 0:
            self._current_step -= 1
            return self._steps[self._current_step]
        return None

    def reset(self) -> None:
        self._current_step = -1

    @property
    def progress(self) -> tuple[int, int]:
        return self._current_step + 1, len(self._steps)


@dataclass
class StepResult:
    layer_id: int
    z: np.ndarray           # 가중합 (활성화 함수 적용 전)
    a: np.ndarray           # 활성화값 (활성화 함수 적용 후)
    W: np.ndarray           # 해당 레이어 가중치 행렬
    b: np.ndarray | None    # 편향
    prev_a: np.ndarray      # 이전 레이어 활성화값 (입력)
    activation: str         # 적용된 활성화 함수 이름
```

### 4.2 `core/activations.py`

```python
import numpy as np

def apply_activation(z: np.ndarray, name: str) -> np.ndarray:
    dispatch = {
        "relu":    lambda x: np.maximum(0, x),
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
        "tanh":    lambda x: np.tanh(x),
        "linear":  lambda x: x,
        "softmax": lambda x: (e := np.exp(x - np.max(x))) / e.sum(),
    }
    fn = dispatch.get(name)
    if fn is None:
        raise ValueError(f"Unknown activation: {name}")
    return fn(z)
```

### 4.3 `core/initializers.py`

```python
def initialize_weights(n_in: int, n_out: int, method: str) -> np.ndarray:
    match method:
        case "random":
            return np.random.uniform(-1, 1, (n_out, n_in))
        case "xavier":
            std = np.sqrt(2 / (n_in + n_out))
            return np.random.normal(0, std, (n_out, n_in))
        case "he":
            std = np.sqrt(2 / n_in)
            return np.random.normal(0, std, (n_out, n_in))
        case _:
            raise ValueError(f"Unknown initializer: {method}")
```

### 4.4 `core/cnn_engine.py` — CNN Forward 계산

```python
class CNNForwardEngine:
    def compute_conv2d(
        self,
        input_map: np.ndarray,    # (H, W, C)
        kernels: np.ndarray,       # (n_filters, kH, kW, C)
        stride: int,
        padding: str,
    ) -> np.ndarray:               # (H', W', n_filters)
        ...  # NumPy 기반 순수 구현 (교육용 속도 충분)

    def compute_pooling(
        self,
        input_map: np.ndarray,    # (H, W, C)
        pool_size: tuple,
        pool_type: str,
    ) -> np.ndarray:
        ...
```

---

## 5. UI 모듈 상세 설계

### 5.1 `ui/canvas/fc_canvas.py` — FC 네트워크 다이어그램

`QGraphicsScene` + `QGraphicsView` 기반으로 커스텀 노드·엣지를 구현한다.

```python
class FCCanvas(QGraphicsView):
    def __init__(self):
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self._neuron_items: dict[tuple[int,int], NeuronItem] = {}
        self._edge_items: list[EdgeItem] = []

    def build_diagram(self, model: NetworkModel) -> None:
        """레이어 구성에 따라 뉴런 및 연결선을 배치"""
        self.scene.clear()
        # 레이어별 수직 중앙 정렬 배치
        # 뉴런 수 > 8 이면 대표 노드 + "..." 생략 표시

    def highlight_layer(self, layer_id: int) -> None:
        """현재 스텝의 레이어 강조 (노란 테두리)"""

    def update_neuron_values(self, layer_id: int, a_values: np.ndarray) -> None:
        """활성화값을 뉴런 내부 텍스트 및 배경 색상에 반영"""

    def update_edge_weights(self, layer_id: int, W: np.ndarray) -> None:
        """가중치에 따라 선 굵기 및 색상 업데이트"""
```

#### 뉴런/엣지 렌더링 규칙

| 속성 | 규칙 |
|------|------|
| 뉴런 반지름 | 24px (고정) |
| 뉴런 배경 밝기 | `활성화값`을 [0,1]로 정규화 후 HSL 밝기 매핑 |
| 엣지 색상 | W > 0 → `#2563EB` / W < 0 → `#DC2626` |
| 엣지 굵기 | `1 + abs(W) * 4` px (최대 5px) |
| 엣지 투명도 | 비활성 레이어 엣지: 30% 불투명도 |
| 현재 레이어 강조 | 노란 테두리 (`#FBBF24`, 3px) |
| 뉴런 수 생략 기준 | 레이어당 뉴런 > 8: 상위 4개 + `⋮` + 하위 4개 표시 |

### 5.2 `ui/canvas/graphics_items.py`

```python
class NeuronItem(QGraphicsEllipseItem):
    def __init__(self, layer_id: int, neuron_id: int):
        ...
    def set_activation(self, value: float): ...
    def set_highlighted(self, on: bool): ...

class EdgeItem(QGraphicsLineItem):
    def __init__(self, src: NeuronItem, dst: NeuronItem):
        ...
    def set_weight(self, w: float): ...
    def set_active(self, on: bool): ...
```

### 5.3 `ui/panels/calc_detail_panel.py` — 계산 상세 패널

```
┌──────────────────────────────────────┐
│ [탭: 수식 뷰] [탭: 행렬 뷰]           │
├──────────────────────────────────────┤
│ 현재 스텝: Layer 2 (ReLU)            │
│                                      │
│ z = W · a_prev + b                   │
│   = [0.3×1.0 + 0.7×0.5] + 0.1       │
│   = 1.25                             │
│                                      │
│ a = ReLU(1.25) = 1.25                │
│──────────────────────────────────────│
│ [계산 로그]                           │
│  ✓ Step 1: Layer 1 완료 → a=[0.8...] │
│  ► Step 2: Layer 2 진행 중...         │
└──────────────────────────────────────┘
```

- **수식 뷰 탭**: `QLabel` HTML 렌더링으로 수식 표시
- **행렬 뷰 탭**: `QTableWidget`으로 W, x, z, a 각 행렬 표시
- **계산 로그**: `QListWidget` (최근 50개 항목 유지)

### 5.4 `ui/panels/step_control_bar.py`

```python
class StepControlBar(QWidget):
    stepForwardRequested = Signal()
    stepBackwardRequested = Signal()
    resetRequested = Signal()
    autoPlayToggled = Signal(bool)
    speedChanged = Signal(float)

    # 구성 위젯
    btn_prev     # ◀ 이전
    btn_next     # ▶ 다음
    btn_reset    # ↺ 초기화
    btn_auto     # ▷ 자동재생 토글
    slider_speed # 재생 속도 0.5× ~ 3×
    progress_bar # QProgressBar (스텝 진행률)
    label_step   # "2 / 5" 텍스트
```

---

## 6. 상태 관리 (`state/app_state.py`)

```python
class AppState:
    """싱글턴 전역 상태"""
    mode: str = "fc"               # "fc" | "cnn"
    language: str = "ko"           # "ko" | "en"
    model: NetworkModel = None
    engine: FCForwardEngine | CNNForwardEngine = None
    current_step: int = -1
    total_steps: int = 0
    is_playing: bool = False
    play_speed: float = 1.0
```

---

## 7. 시그널/슬롯 흐름 (Signal Flow)

```
[ConfigPanel]
  └─ Signal: networkConfigChanged(config: dict)
       └─ MainWindow.on_config_changed()
            ├─ NetworkModel 재구성
            ├─ FCForwardEngine.compute_all_steps()
            └─ FCCanvas.build_diagram(model)

[InputPanel]
  └─ Signal: inputDataChanged(data: np.ndarray)
       └─ MainWindow.on_input_changed()
            ├─ model.input_data 갱신
            ├─ FCForwardEngine.compute_all_steps()  ← 재계산
            └─ FCCanvas.reset_values()

[StepControlBar]
  └─ Signal: stepForwardRequested
       └─ MainWindow.on_step_forward()
            ├─ engine.step_forward() → StepResult
            ├─ FCCanvas.highlight_layer(result.layer_id)
            ├─ FCCanvas.update_neuron_values(result.layer_id, result.a)
            ├─ FCCanvas.update_edge_weights(result.layer_id, result.W)
            ├─ CalcDetailPanel.update(result)
            └─ StepControlBar.set_progress(engine.progress)

  └─ Signal: resetRequested
       └─ MainWindow.on_reset()
            ├─ engine.reset()
            ├─ FCCanvas.reset_all_highlights()
            └─ CalcDetailPanel.clear_log()

[Toolbar]
  └─ Signal: languageToggled(lang: str)
       └─ MainWindow.on_language_change()
            └─ LanguageManager.load(lang) → retranslateUi()

  └─ Signal: modeChanged(mode: str)
       └─ MainWindow.on_mode_change()
            ├─ AppState.mode 갱신
            └─ 캔버스 전환 (FCCanvas ↔ CNNCanvas)

  └─ Signal: presetSelected(preset_name: str)
       └─ MainWindow.on_preset_selected()
            └─ preset JSON 로드 → networkConfigChanged emit
```

---

## 8. 국제화(i18n) 설계

### 8.1 문자열 파일 구조

```json
// i18n/ko.json
{
  "app_title": "순전파 시각화 탐색기",
  "mode_fc": "완전 연결 (FC)",
  "mode_cnn": "합성곱 (CNN)",
  "btn_next_step": "▶ 다음 단계",
  "btn_prev_step": "◀ 이전 단계",
  "btn_reset": "↺ 초기화",
  "btn_auto_play": "▷ 자동 재생",
  "label_layers": "레이어 수",
  "label_neurons": "뉴런 수",
  "label_activation": "활성화 함수",
  "label_init": "가중치 초기화",
  "label_input": "입력값",
  "btn_random_input": "랜덤 입력",
  "panel_formula": "수식 뷰",
  "panel_matrix": "행렬 뷰",
  "panel_log": "계산 로그",
  "lang_toggle": "EN",
  "step_progress": "{current} / {total} 단계",
  "no_params": "이 레이어는 파라미터가 없습니다."
}
```

### 8.2 LanguageManager

```python
class LanguageManager:
    _lang = "ko"
    _strings: dict = {}

    @classmethod
    def load(cls, lang: str):
        cls._lang = lang
        path = Path(f"i18n/{lang}.json")
        cls._strings = json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def t(cls, key: str, **kwargs) -> str:
        text = cls._strings.get(key, key)
        return text.format(**kwargs) if kwargs else text
```

---

## 9. 성능 최적화 전략

| 요구사항 | 구현 방법 |
|----------|-----------|
| 스텝 전환 ≤ 150ms | `compute_all_steps()`로 Forward 계산을 미리 완료, 스텝 전환 시 저장된 `StepResult`만 렌더링 |
| 다이어그램 갱신 최소화 | 전체 `scene.clear()` 대신 변경된 레이어의 아이템만 업데이트 (`update_neuron_values`, `update_edge_weights`) |
| 자동 재생 타이머 | `QTimer` 기반, `play_speed` 반영: `interval = int(1000 / speed)` ms |
| 뉴런 수 많을 때 생략 | 레이어당 뉴런 > 8: 대표 노드만 `QGraphicsItem`으로 생성, 나머지 생략 |

---

## 10. 에러 처리

| 상황 | 처리 방법 |
|------|-----------|
| 가중치 NaN / Inf 발생 | `np.nan_to_num()` 적용 후 경고 메시지 표시 |
| 입력값 미입력 | 스텝 진행 버튼 비활성화, 툴팁으로 안내 |
| 레이어 수 0 설정 | ConfigPanel에서 최솟값 1로 제한 (`QSpinBox.setMinimum(1)`) |
| CNN 이미지 크기 초과 | 업로드 시 자동 리사이즈 (32×32 이하로 Pillow 처리) |
| 프리셋 JSON 파싱 실패 | `try/except` 후 기본 구성(XOR Solver) fallback |

---

## 11. 테스트 계획

### 11.1 단위 테스트

```python
# test_forward_engine.py
def test_fc_single_layer():
    """단일 레이어 z = Wx + b 및 ReLU 적용 검증"""
    model = NetworkModel(mode="fc", ...)
    engine = FCForwardEngine(model)
    engine.compute_all_steps()
    result = engine.step_forward()
    assert result.layer_id == 0
    assert np.allclose(result.z, expected_z)
    assert np.all(result.a >= 0)  # ReLU 출력은 항상 >= 0

def test_step_backward():
    """step_backward 후 step_forward 결과 동일성 검증"""
    ...

def test_sigmoid_output_range():
    """Sigmoid 활성화값이 (0,1) 범위 내임을 검증"""
    ...
```

### 11.2 통합 테스트

| 시나리오 | 검증 내용 |
|----------|-----------|
| 레이어 구성 변경 | 다이어그램 즉시 갱신, 이전 스텝 상태 초기화 |
| Step-by-step 진행 | 매 스텝마다 올바른 레이어 강조 및 활성화값 표시 |
| 이전 단계 버튼 | 이전 `StepResult` 상태로 정확히 복원 |
| 언어 전환 | 전체 UI 텍스트 변경, 수식 유지 |
| 프리셋 로드 | JSON 파싱 후 네트워크 구성 및 다이어그램 정상 갱신 |
| CNN 모드 전환 | FC 캔버스 숨김, CNN 캔버스 표시 |

---

## 12. 의존성 (`requirements.txt`)

```
PySide6>=6.6.0
matplotlib>=3.8.0
numpy>=1.26.0
Pillow>=10.0.0
```

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
| Backpropagation 확장 | `FCForwardEngine`을 상속하는 `BackpropEngine` 추가, gradient 흐름 역방향 렌더링 |
| 네트워크 저장/불러오기 | `NetworkModel`을 JSON 직렬화 (`dataclasses-json` 또는 커스텀 직렬화) |
| 출력층 확률 분포 | Matplotlib 막대 그래프 위젯 추가 (Softmax 출력 시각화) |
| 뉴런 클릭 상세 팝업 | `NeuronItem.mousePressEvent` 오버라이드 → `QDialog` 팝업 |

---

*이 문서는 PRD v1.0.0 기반으로 작성되었으며, 구현 착수 전 기술 검토가 필요합니다.*
