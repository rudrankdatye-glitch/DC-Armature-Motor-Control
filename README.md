# DC Motor Speed Control - Interactive P / PD / PI / PID Simulator

An interactive Python simulation of **classical controllers (P, PD, PI, PID) for DC motor speed control**.
Pick a controller, move the gain sliders, and watch the three **state variables** of the motor -
**angular position, angular velocity and armature current** - update in real time.
A green / red banner tells you instantly whether the closed loop is stable.

![Stable PID response](images/main_window_stable.png)

## Features

- Select **P, PD, PI or PID**; only the gain sliders that apply (Kp, Ki, Kd) are enabled
- **Real-time** plots of all three states: angular position θ(t), angular velocity ω(t), armature current i(t)
- Adjustable armature inductance **L**, reference speed and simulation time
- **Green banner = stable, red banner = unstable** (computed from the closed-loop poles)
- Performance values written on the plot: rise time, settling time, overshoot, steady-state error,
  peak current, steady-state current, final position
- Hover over any plot to read exact (time, value) pairs
- **Compare** button: one window with P vs PD vs PI vs PID for all three states, a metrics table and an
  automatic "most suitable controller" recommendation
- **Export CSV** of the plotted time response

![Unstable PI example](images/main_window_unstable.png)

![Controller comparison](images/comparison.png)

## Motor model

State vector `x = [θ, ω, i]`, input = armature voltage `V`:

```
dθ/dt = ω
J dω/dt = K i - b ω
L di/dt = V - R i - K ω
```

Parameters: `J = 0.01 kg·m²`, `b = 0.1 N·m·s`, `K = Kt = Ke = 0.01`, `R = 1 Ω`, `L = 0.5 H` (adjustable).

Speed transfer function:

```
G(s) = ω(s)/V(s) = K / [ (J s + b)(L s + R) + K² ]
```

Control law: `V = C(s)·(ω_ref − ω)` with `C(s) = Kp + Ki/s + Kd·s` (terms removed according to the selected controller).
With `C = Nc/Dc` and `D(s) = Dc·[(Js+b)(Ls+R)+K²] + Nc·K`:

```
ω/ref = Nc·K / D        θ/ref = (ω/ref) / s        i/ref = Nc·(J s + b) / D
```

The closed-loop stability is decided by the roots of `D(s)`. Position θ integrates ω, so it keeps growing
while the motor runs (it does not settle) - this is expected.

## Installation

Requires **Python 3.9+** (tested with 3.11).

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

## Run

```bash
python dc_motor_3state_controllers.py
```

Or open the file in **IDLE** and press **F5**. Run it as a normal script (not inside a Jupyter notebook), because the
interactive widgets need a desktop window.

## How to use

1. Choose a controller on the left.
2. Drag **Kp / Ki / Kd** (and **L**, reference, simulation time) - the plots update immediately.
3. Select each controller in turn and tune it, then press **Compare P vs PD vs PI vs PID**.
4. Press **Export CSV** to save the current time response.

## Example results (reference 1 rad/s, L = 0.5 H)

| Controller | Gains | Rise time | Settling time | Overshoot | Steady-state error | Peak current |
|---|---|---|---|---|---|---|
| P | Kp = 100 | 0.099 s | 0.568 s | 24.9 % | 9.1 % | 14.7 A |
| PD | Kp = 100, Kd = 1 | 0.105 s | 0.548 s | 18.8 % | 9.1 % | 13.8 A |
| PI | Kp = 100, Ki = 100 | 0.104 s | 1.033 s | 22.3 % | 0 % | 15.8 A |
| PID | Kp = 100, Ki = 200, Kd = 10 | 0.132 s | 0.258 s | 1.0 % | 0 % | 20.0 A |

**Observations**

- **P** is fast but leaves a steady-state error and overshoots.
- **PD** reduces the overshoot but not the steady-state error.
- **PI** removes the steady-state error but is slow to settle.
- **PID** gives zero steady-state error, almost no overshoot and the shortest settling time - the most suitable
  controller for this motor model.
- The PID peak current comes from the ideal derivative term acting on a step reference ("derivative kick"). A real
  drive would limit the current or filter the derivative.

## Stability notes

- For this second-order motor model, **P and PD are stable for any positive gains**.
- **PI and PID** can become unstable when `Ki` is too large for `Kp` (and a larger `L` makes this easier). Example:
  Kp = 10 and Ki = 500 on PI gives the red banner.

## Project structure

```
.
├── dc_motor_3state_controllers.py   # simulator + interactive GUI
├── requirements.txt
├── README.md
└── images/                          # screenshots used in this README
```
