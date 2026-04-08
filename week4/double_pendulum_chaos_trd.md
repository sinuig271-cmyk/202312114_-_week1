# 이중 진자 카오스 시뮬레이션 앱 TRD
# Technical Requirements Document

**문서 버전:** 1.0  
**작성일:** 2026-04-06  
**참조 문서:** double_pendulum_chaos_prd.md v1.0  
**대상 독자:** 개발자

---

## 1. 기술 스택 확정

| 구분 | 라이브러리 | 버전 | 용도 |
|------|-----------|------|------|
| UI 프레임워크 | PySide6 | ≥ 6.6 | 윈도우, 탭, 위젯, 이벤트 루프 |
| 수치 계산 | NumPy | ≥ 1.26 | RK4 벡터 연산, 상태 배열 |
| 과학 계산 | SciPy | ≥ 1.12 | 리아푸노프 지수 선형 회귀 보조 |
| 시각화 | Matplotlib | ≥ 3.8 | 2D/3D 플롯, Qt 백엔드 임베딩 |
| 애니메이션 루프 | PySide6 QTimer | — | 16ms 틱으로 프레임 구동 |
| 언어 | Python | ≥ 3.11 | 타입 힌트 전면 사용 |

> **3D 백엔드:** Matplotlib `Axes3D` (mpl_toolkits.mplot3d) — 외부 의존성 추가 없음.  
> **수치 적분:** NumPy 기반 직접 RK4 구현 — SciPy solve_ivp 미사용.

---

## 2. 프로젝트 파일 구조

```
chaoslab/
├── main.py                        # 앱 진입점 — QApplication 생성 및 실행
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py             # QMainWindow + QTabWidget 조립
│   ├── tab_pendulum.py            # Tab 1: 이중 진자 애니메이션
│   ├── tab_butterfly.py           # Tab 2: 나비효과
│   ├── tab_lorenz.py              # Tab 3: 로렌츠 어트랙터
│   └── tab_phase.py               # Tab 4: 위상공간
│
├── physics/
│   ├── __init__.py
│   ├── double_pendulum.py         # 운동방정식 + RK4 스테퍼
│   ├── lorenz.py                  # 로렌츠 방정식 + RK4 스테퍼
│   └── analysis.py                # 리아푸노프 지수, 발산 거리 계산
│
├── widgets/
│   ├── __init__.py
│   ├── mpl_canvas.py              # FigureCanvasQTAgg 래퍼 (공용)
│   └── param_slider.py            # 라벨+슬라이더+값 표시 복합 위젯
│
└── utils/
    ├── __init__.py
    └── export.py                  # CSV 저장 유틸리티
```

---

## 3. 물리 엔진 설계

### 3.1 이중 진자 운동방정식

라그랑지안 역학에서 유도된 연립 ODE를 상태벡터로 표현한다.

**상태벡터**

```
y = [θ1, ω1, θ2, ω2]   (ω = dθ/dt)
```

**유도된 각가속도 (분모 공통)**

```
Δ = (m1 + m2)·L1 - m2·L1·cos²(θ1 - θ2)

α1 = [ m2·L1·ω1²·sin(θ1-θ2)·cos(θ1-θ2)
      + m2·g·sin(θ2)·cos(θ1-θ2)
      + m2·L2·ω2²·sin(θ1-θ2)
      - (m1+m2)·g·sin(θ1) ] / Δ

α2 = [ -(m1+m2)·L1·ω1²·sin(θ1-θ2)·... (생략)
      ... ] / Δ
```

> 전체 수식은 `physics/double_pendulum.py`의 `derivatives()` 함수 주석에 포함.

### 3.2 RK4 스테퍼 인터페이스

```python
# physics/double_pendulum.py

import numpy as np
from dataclasses import dataclass

@dataclass
class PendulumParams:
    m1: float = 1.0   # kg
    m2: float = 1.0   # kg
    L1: float = 1.0   # m
    L2: float = 1.0   # m
    g:  float = 9.81  # m/s²

def derivatives(y: np.ndarray, p: PendulumParams) -> np.ndarray:
    """상태벡터 y = [θ1, ω1, θ2, ω2] → dy/dt 반환"""
    ...

def rk4_step(y: np.ndarray, p: PendulumParams, dt: float) -> np.ndarray:
    """단일 RK4 스텝. 새 상태벡터 반환."""
    k1 = derivatives(y,          p)
    k2 = derivatives(y + dt/2*k1, p)
    k3 = derivatives(y + dt/2*k2, p)
    k4 = derivatives(y + dt   *k3, p)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

### 3.3 로렌츠 방정식

```python
# physics/lorenz.py

@dataclass
class LorenzParams:
    sigma: float = 10.0
    rho:   float = 28.0
    beta:  float = 8/3

def lorenz_derivatives(y: np.ndarray, p: LorenzParams) -> np.ndarray:
    """y = [x, y, z] → [dx/dt, dy/dt, dz/dt]"""
    x, yy, z = y
    return np.array([
        p.sigma * (yy - x),
        x * (p.rho - z) - yy,
        x * yy - p.beta * z
    ])

def rk4_step(y: np.ndarray, p: LorenzParams, dt: float) -> np.ndarray:
    """로렌츠 전용 RK4 스텝"""
    ...
```

### 3.4 에너지 계산

수치 안정성 검증용. Tab 1에서 실시간 표시.

```python
def total_energy(y: np.ndarray, p: PendulumParams) -> tuple[float, float, float]:
    """(KE, PE, E_total) 반환 — SI 단위 (J)"""
    θ1, ω1, θ2, ω2 = y
    # 좌표 변환
    x1 =  p.L1 * np.sin(θ1)
    y1 = -p.L1 * np.cos(θ1)
    x2 = x1 + p.L2 * np.sin(θ2)
    y2 = y1 - p.L2 * np.cos(θ2)
    # KE
    v1x = p.L1 * ω1 * np.cos(θ1)
    v1y = p.L1 * ω1 * np.sin(θ1)
    v2x = v1x + p.L2 * ω2 * np.cos(θ2)
    v2y = v1y + p.L2 * ω2 * np.sin(θ2)
    ke = 0.5*p.m1*(v1x**2+v1y**2) + 0.5*p.m2*(v2x**2+v2y**2)
    # PE (원점 = 피벗)
    pe = -p.m1*p.g*p.L1*np.cos(θ1) - p.m2*p.g*(p.L1*np.cos(θ1)+p.L2*np.cos(θ2))
    return ke, pe, ke + pe
```

### 3.5 리아푸노프 지수 추정

```python
# physics/analysis.py

def lyapunov_estimate(distances: np.ndarray, times: np.ndarray) -> float:
    """
    발산 거리 d(t) = d0 * exp(λ·t) 에서 λ 추정.
    log(d(t)) ~ λ·t 선형 회귀 (scipy.stats.linregress 사용).
    """
    from scipy import stats
    log_d = np.log(np.maximum(distances, 1e-12))
    slope, *_ = stats.linregress(times, log_d)
    return slope  # λ (s⁻¹)
```

---

## 4. 애니메이션 아키텍처

### 4.1 QTimer + Matplotlib 루프 설계

```
QTimer (16ms)
    │
    ▼
on_tick()                          ← 각 탭 위젯의 슬롯
    ├── physics: rk4_step() × N    ← N = 서브스텝 수 (기본값 4)
    ├── state 버퍼 업데이트
    ├── canvas.axes.cla() / set_data()
    └── canvas.draw_idle()         ← Qt 이벤트 루프와 충돌 없는 비동기 드로우
```

**서브스텝 전략:** QTimer 1틱(16ms)당 물리 스텝을 N번 수행해 dt_physics < dt_render를 보장한다.

```python
DT_RENDER   = 0.016   # 16ms = ~60fps
N_SUBSTEPS  = 4
DT_PHYSICS  = DT_RENDER / N_SUBSTEPS  # 0.004s
```

### 4.2 공용 MplCanvas 위젯

```python
# widgets/mpl_canvas.py

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, projection: str | None = None):
        self.fig = Figure(tight_layout=True)
        self.axes = self.fig.add_subplot(111, projection=projection)
        super().__init__(self.fig)
        self.setParent(parent)
```

- `projection=None` → 2D 플롯 (Tab 1, 2, 4)
- `projection='3d'` → Axes3D (Tab 3)

### 4.3 렌더링 최적화 규칙

| 규칙 | 이유 |
|------|------|
| `set_data()` 우선, `cla()` 최소화 | axes 재생성 비용 절감 |
| `draw_idle()` 사용, `draw()` 금지 | Qt 이벤트 루프 블로킹 방지 |
| 궤적 잔상은 deque(maxlen=N) 사용 | 메모리 상한 보장 |
| 로렌츠 3D는 `set_data_3d()` 사용 | Line3D 객체 재사용 |
| blitting 미적용 (MVP) | Axes3D와 호환 불가, 향후 검토 |

---

## 5. UI 모듈 상세 설계

### 5.1 MainWindow

```python
# ui/main_window.py

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChaosLab — 이중 진자 카오스 시뮬레이터")
        self.resize(1200, 750)

        self.tabs = QTabWidget()
        self.tabs.addTab(PendulumTab(),  "🔵 이중 진자")
        self.tabs.addTab(ButterflyTab(), "🦋 나비효과")
        self.tabs.addTab(LorenzTab(),    "🌀 로렌츠 어트랙터")
        self.tabs.addTab(PhaseTab(),     "📊 위상공간")
        self.setCentralWidget(self.tabs)

        # 탭 전환 시 비활성 탭 타이머 일시정지
        self.tabs.currentChanged.connect(self._on_tab_changed)
```

### 5.2 공용 ParamSlider 위젯

```python
# widgets/param_slider.py

class ParamSlider(QWidget):
    """라벨 + QSlider + 현재값 표시를 하나의 위젯으로 묶음."""
    valueChanged = Signal(float)

    def __init__(self, label: str, min_v: float, max_v: float,
                 default: float, decimals: int = 2):
        ...
    # 내부적으로 정수 슬라이더를 float으로 매핑
    # 값 변경 시 valueChanged Signal emit
```

### 5.3 Tab 1 — PendulumTab 구조

```python
class PendulumTab(QWidget):
    def __init__(self):
        # 레이아웃: QHBoxLayout
        #   ├── MplCanvas (stretch=3)      ← 진자 캔버스
        #   └── QVBoxLayout (stretch=1)    ← 파라미터 패널
        #         ├── ParamSlider × 6
        #         └── QHBoxLayout          ← 버튼 행
        #               ├── QPushButton "▶ 시작"
        #               ├── QPushButton "⏸ 일시정지"
        #               └── QPushButton "↺ 초기화"
        #
        # QLabel × 3: KE / PE / E 실시간 표시
        ...

    def _on_tick(self):
        # 1. N_SUBSTEPS번 rk4_step
        # 2. 직교좌표 변환
        # 3. Line2D.set_data() 업데이트
        # 4. 궤적 deque append
        # 5. 에너지 라벨 갱신
        # 6. canvas.draw_idle()
        ...
```

### 5.4 Tab 2 — ButterflyTab 구조

```python
class ButterflyTab(QWidget):
    N_PENDULUMS: int = 5
    COLORS = plt.cm.plasma(np.linspace(0, 1, N_PENDULUMS))

    def __init__(self):
        # 레이아웃: QVBoxLayout
        #   ├── QHBoxLayout
        #   │     ├── MplCanvas "진자 캔버스" (stretch=2)
        #   │     └── QVBoxLayout 파라미터
        #   └── MplCanvas "발산 그래프" (stretch=1, height 고정)
        ...

    def _build_initial_states(self, base_theta1: float, delta: float) -> list[np.ndarray]:
        """Δθ 간격으로 N개 초기 상태벡터 생성"""
        ...

    def _on_tick(self):
        # N개 진자 개별 rk4_step
        # 발산 거리 계산 → deque append
        # 리아푸노프 추정 (100 포인트마다 재계산)
        ...
```

### 5.5 Tab 3 — LorenzTab 구조

```python
class LorenzTab(QWidget):
    def __init__(self):
        # MplCanvas(projection='3d')
        # 슬라이더: sigma, rho, beta, 속도
        # 마우스 회전: Matplotlib 기본 Axes3D 인터랙션 활용
        ...

    def _on_tick(self):
        # 매 틱마다 BATCH_SIZE(=20) 포인트 생성
        # Line3D.set_data_3d() 업데이트
        # ax.set_xlim/ylim/zlim 고정 (뷰 안정화)
        ...
```

### 5.6 Tab 4 — PhaseTab 구조

```python
class PhaseTab(QWidget):
    def __init__(self):
        # QHBoxLayout
        #   ├── MplCanvas "θ1-ω1" (subplot 121)
        #   └── MplCanvas "θ2-ω2" (subplot 122)
        # Tab 1 시뮬레이션 상태를 공유 참조로 수신
        # 포앙카레 단면: θ2 부호 변환 감지 시 scatter 점 추가
        ...

    def _export_csv(self):
        """QFileDialog → utils/export.py 호출"""
        ...
```

---

## 6. 데이터 흐름

```
[ParamSlider] ─ valueChanged ──► [Tab 위젯]
                                      │
                              PendulumParams 갱신
                                      │
                              [QTimer 16ms tick]
                                      │
                         ┌────────────┴──────────────┐
                    rk4_step()                  rk4_step() × N
                    (단일 진자)                  (나비효과)
                         │                            │
                   state: np.ndarray          states: list[np.ndarray]
                         │                            │
                 ┌───────┴────────┐           ┌───────┴────────┐
           직교좌표 변환       에너지 계산   발산거리 계산   λ 추정
                 │                                    │
           Line2D.set_data()                  Line2D.set_data()
                 │                            scatter.set_offsets()
           canvas.draw_idle()                 canvas.draw_idle()
```

---

## 7. 상태 공유 전략

Tab 1과 Tab 4는 동일한 시뮬레이션 상태를 참조해야 한다.

```python
# ui/main_window.py

self.pendulum_tab = PendulumTab()
self.phase_tab    = PhaseTab(state_source=self.pendulum_tab)

# PhaseTab은 pendulum_tab.state_history (deque) 를 직접 참조
# 별도 복사 없이 메모리 공유 — thread-safe 불필요 (단일 Qt 스레드)
```

---

## 8. 파라미터 범위 및 수치 안정성

### 8.1 슬라이더 범위

| 파라미터 | 최소 | 최대 | 기본값 | 단위 |
|---------|------|------|--------|------|
| m1, m2 | 0.1 | 5.0 | 1.0 | kg |
| L1, L2 | 0.2 | 2.0 | 1.0 | m |
| θ1, θ2 | -180 | 180 | 120 / -20 | deg |
| g | 1.0 | 20.0 | 9.81 | m/s² |
| Δθ (나비효과) | 0.001 | 1.0 | 0.01 | deg |
| σ | 1 | 20 | 10 | — |
| ρ | 1 | 50 | 28 | — |
| β | 0.1 | 5.0 | 2.667 | — |

### 8.2 수치 발산 방지

```python
def rk4_step(y, p, dt):
    y_new = ...
    # 각도 클리핑 불필요 (진자는 자유 회전 허용)
    # 각속도 상한 클리핑
    MAX_OMEGA = 50.0  # rad/s
    y_new[1] = np.clip(y_new[1], -MAX_OMEGA, MAX_OMEGA)
    y_new[3] = np.clip(y_new[3], -MAX_OMEGA, MAX_OMEGA)
    # NaN 감지 시 초기 상태로 복구
    if np.any(np.isnan(y_new)):
        return y  # 이전 상태 유지
    return y_new
```

### 8.3 에너지 드리프트 허용 기준

| 조건 | 기준 |
|------|------|
| 1분 미만 | ΔE / E₀ < 0.1% |
| 1시간 연속 | ΔE / E₀ < 1.0% |
| 허용 초과 시 | 상태바에 경고 메시지 표시 |

---

## 9. CSV 내보내기 스펙

```python
# utils/export.py

def export_phase_csv(path: str, history: list[np.ndarray]) -> None:
    """
    columns: time, theta1, omega1, theta2, omega2
    """
```

**CSV 형식 예시**

```
time,theta1_rad,omega1_rad_s,theta2_rad,omega2_rad_s
0.000,2.094,-0.000,-0.349,0.000
0.016,2.093,-0.041,-0.348,0.023
...
```

---

## 10. 의존성 및 설치

### 10.1 requirements.txt

```
PySide6>=6.6.0
numpy>=1.26.0
matplotlib>=3.8.0
scipy>=1.12.0
```

### 10.2 실행 방법

```bash
pip install -r requirements.txt
python main.py
```

### 10.3 플랫폼별 주의사항

| 플랫폼 | 주의사항 |
|--------|---------|
| macOS | `matplotlib` 백엔드를 `QtAgg`로 명시 필요 |
| Windows | PySide6 DLL 경로 자동 해결됨 |
| Ubuntu | `libxcb-*` 패키지 사전 설치 필요 |

```python
# main.py 상단에 백엔드 명시 (import matplotlib 직후)
import matplotlib
matplotlib.use("QtAgg")
```

---

## 11. 테스트 전략

### 11.1 단위 테스트 대상 (physics/ 모듈)

| 테스트 | 검증 내용 |
|--------|----------|
| `test_rk4_energy` | 단진자(해석해 존재) 에너지 보존 오차 < 0.01% |
| `test_lorenz_attractor` | σ=10, ρ=28, β=8/3에서 어트랙터 수렴 확인 |
| `test_lyapunov_positive` | 카오스 조건에서 λ > 0 |
| `test_nan_guard` | 극단 파라미터 입력 시 NaN 미반환 |
| `test_energy_components` | KE + PE = E_total 항등식 |

### 11.2 수동 검증 항목

| 항목 | 방법 |
|------|------|
| 60fps 달성 여부 | QTimer 실제 간격 로깅 (개발 모드) |
| 나비효과 발산 가시성 | Δθ=0.01°에서 30초 내 육안 발산 확인 |
| 로렌츠 어트랙터 형태 | 나비 날개 쌍 궤도 육안 확인 |

---

## 12. 향후 확장 고려사항 (MVP 외)

| 항목 | 기술적 접근 |
|------|------------|
| Blitting 최적화 | `canvas.copy_from_bbox()` + `restore_region()` (2D 전용) |
| 멀티스레딩 | QThread로 물리 연산 분리, Signal로 상태 전달 |
| GIF/MP4 저장 | `matplotlib.animation.FFMpegWriter` |
| 웹 포팅 | PyScript + Pyodide (SciPy 미지원 시 직접 RK4 유지) |

---

*ChaosLab TRD v1.0 — PySide6 + NumPy + Matplotlib 기반 기술 설계 문서*
