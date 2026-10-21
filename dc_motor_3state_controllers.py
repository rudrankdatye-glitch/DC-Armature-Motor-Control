#!/usr/bin/env python3
"""
DC Motor Speed Control - Interactive P / PD / PI / PID Simulator  (3 state variables)
=====================================================================================

Requirements :  pip install numpy scipy matplotlib
Run          :  python dc_motor_3state_controllers.py      (or open in IDLE and press F5)

WHAT IS NEW IN THIS VERSION
    * Three state-variable plots:  angular position  theta(t),
                                   angular velocity  omega(t)   (the controlled output),
                                   armature current  i(t).
    * The armature inductance L is an adjustable slider (all other motor data are constants).
    * Peak current, steady-state current and final position are reported.
    * The comparison window (P vs PD vs PI vs PID) shows all three state variables.

DC MOTOR STATE-SPACE MODEL  (states x = [theta, omega, i],  input = armature voltage V)
    d(theta)/dt = omega
    J d(omega)/dt = K i - b omega
    L d(i)/dt     = V - R i - K omega

    J = 0.01 kg.m^2   b = 0.1 N.m.s   K = Kt = Ke = 0.01   R = 1 ohm   L = adjustable (default 0.5 H)

CONTROL LOOP        V = C(s) * ( omega_ref - omega )      with  C(s) = Kp + Ki/s + Kd s
    Closed-loop transfer functions (D(s) = Dc*DEN + Nc*K, DEN = (Js+b)(Ls+R)+K^2, C = Nc/Dc):
        omega/ref = Nc*K / D          theta/ref = omega/ref / s          i/ref = Nc*(Js+b) / D
    The position theta is an integrator of omega, so it keeps growing while the motor runs
    (it does not settle).  Stability of the loop is decided by the roots of D(s).
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider
from scipy import signal

# ----------------------------------------------------------------------------
# 1. DC MOTOR MODEL  (L is adjustable)
# ----------------------------------------------------------------------------
J, B, K, R = 0.01, 0.1, 0.01, 1.0
L_DEFAULT = 0.5


def plant_den(L):
    """(J s + b)(L s + R) + K^2"""
    return np.polyadd(np.polymul([J, B], [L, R]), [K * K])


# ----------------------------------------------------------------------------
# 2. CONTROLLERS + CLOSED LOOP
# ----------------------------------------------------------------------------
USED = {"P": ("Kp",), "PD": ("Kp", "Kd"), "PI": ("Kp", "Ki"), "PID": ("Kp", "Ki", "Kd")}
DEFAULTS = {
    "P": dict(Kp=100.0, Ki=0.0, Kd=0.0),
    "PD": dict(Kp=100.0, Ki=0.0, Kd=1.0),
    "PI": dict(Kp=100.0, Ki=100.0, Kd=0.0),
    "PID": dict(Kp=100.0, Ki=200.0, Kd=10.0),
}
COLORS = {"P": "#1f77b4", "PD": "#ff7f0e", "PI": "#17a589", "PID": "#8e44ad"}
GREEN, RED = "#1e8e3e", "#d93025"
LABELS = {
    "P": "P    - Proportional",
    "PD": "PD  - Proportional-Derivative",
    "PI": "PI   - Proportional-Integral",
    "PID": "PID - Prop.-Integral-Derivative",
}


def _trim(p):
    p = np.trim_zeros(np.atleast_1d(np.asarray(p, dtype=float)), "f")
    return p if p.size else np.array([0.0])


def effective_gains(kind, g):
    return {k: (g[k] if k in USED[kind] else 0.0) for k in ("Kp", "Ki", "Kd")}


def controller_tf(kind, g):
    """(num, den) of C(s). A zero Ki / Kd simply removes that term."""
    e = effective_gains(kind, g)
    has_i, has_d = e["Ki"] > 1e-12, e["Kd"] > 1e-12
    if has_i and has_d:
        return [e["Kd"], e["Kp"], e["Ki"]], [1.0, 0.0]
    if has_i:
        return [e["Kp"], e["Ki"]], [1.0, 0.0]
    if has_d:
        return [e["Kd"], e["Kp"]], [1.0]
    return [e["Kp"]], [1.0]


def controller_string(kind, g):
    e = effective_gains(kind, g)
    s = f"Kp = {e['Kp']:.2f}"
    if "Ki" in USED[kind]:
        s += f", Ki = {e['Ki']:.2f}"
    if "Kd" in USED[kind]:
        s += f", Kd = {e['Kd']:.2f}"
    return s


def closed_loop(kind, g, L):
    """Return numerators of omega/ref and i/ref and the common denominator D(s)."""
    nc, dc = controller_tf(kind, g)
    D = _trim(np.polyadd(np.polymul(dc, plant_den(L)), np.polymul(nc, [K])))
    num_w = _trim(np.polymul(nc, [K]))
    num_i = _trim(np.polymul(nc, [J, B]))
    return num_w, num_i, D


# ----------------------------------------------------------------------------
# 3. SIMULATION + METRICS
# ----------------------------------------------------------------------------
def simulate(kind, g, ref, tend, L, n=1500):
    num_w, num_i, D = closed_loop(kind, g, L)
    poles = np.roots(D)
    max_re = float(poles.real.max()) if poles.size else -1.0
    stable = max_re < -1e-9
    t = np.linspace(0.0, tend, n)

    def step(num, den):
        with np.errstate(all="ignore"):
            _, y = signal.step(signal.TransferFunction(num, den), T=t)
        return np.clip(np.nan_to_num(y * ref, nan=1e9, posinf=1e9, neginf=-1e9), -1e9, 1e9)

    w = step(num_w, D)                          # angular velocity
    th = step(num_w, _trim(np.polymul(D, [1, 0])))   # angular position = integral of omega
    i = step(num_i, D)                          # armature current
    return dict(t=t, w=w, y=w, th=th, i=i, ref=ref, num=num_w, num_i=num_i, den=D, poles=poles,
                max_re=max_re, stable=stable, kind=kind, g=dict(g), L=L)


def _first_cross(t, y, level):
    idx = np.where(y >= level)[0]
    if idx.size == 0:
        return np.nan
    k = idx[0]
    if k == 0:
        return t[0]
    return float(np.interp(level, [y[k - 1], y[k]], [t[k - 1], t[k]]))


def compute_metrics(sim):
    """Speed metrics: rise (10-90 %), settling (2 %), overshoot, steady-state error."""
    nan = np.nan
    m = dict(tr=nan, t10=nan, t90=nan, ts=nan, os=nan, tp=nan, yp=nan, yss=nan, ess=nan, ess_pct=nan)
    if not sim["stable"]:
        return m
    t, y, ref = sim["t"], sim["w"], sim["ref"]
    yss = ref * sim["num"][-1] / sim["den"][-1]  # final-value theorem
    m["yss"], m["ess"] = yss, ref - yss
    m["ess_pct"] = 100.0 * (ref - yss) / ref
    if abs(yss) < 1e-9 * max(ref, 1):
        return m
    m["t10"], m["t90"] = _first_cross(t, y, 0.1 * yss), _first_cross(t, y, 0.9 * yss)
    if not (np.isnan(m["t10"]) or np.isnan(m["t90"])):
        m["tr"] = m["t90"] - m["t10"]
    ip = int(np.argmax(y))
    m["yp"], m["tp"] = float(y[ip]), float(t[ip])
    m["os"] = max(0.0, 100.0 * (y[ip] - yss) / abs(yss))
    out = np.where(np.abs(y - yss) > 0.02 * abs(yss))[0]
    if out.size == 0:
        m["ts"] = 0.0
    elif out[-1] >= len(t) - 1:
        m["ts"] = nan
    else:
        m["ts"] = float(t[out[-1] + 1])
    return m


def current_metrics(sim):
    """Peak current and steady-state current (valid for stable loops)."""
    t, i = sim["t"], sim["i"]
    out = dict(ip=np.nan, tip=np.nan, iss=np.nan)
    if not sim["stable"]:
        return out
    k = int(np.argmax(np.abs(i)))
    out["ip"], out["tip"] = float(i[k]), float(t[k])
    out["iss"] = sim["ref"] * sim["num_i"][-1] / sim["den"][-1]
    return out


def fmt(v, unit="", nd=4, na="N/A"):
    return na if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}{unit}"


def stability_text(sim, kind):
    if sim["stable"]:
        return (f"STABLE  -  all closed-loop poles are in the left half-plane "
                f"(rightmost pole Re = {sim['max_re']:.3f})"), GREEN
    reason = ""
    if kind in ("PI", "PID"):
        reason = "\nKi is too large for the current Kp / L: reduce Ki or L, or increase Kp."
    return (f"UNSTABLE  -  closed-loop pole in the right half-plane / imaginary axis "
            f"(Re = {sim['max_re']:+.3f}){reason}"), RED


def robust_ylim(y):
    """y-limits for diverging (unstable) signals: look at the first quarter of the run."""
    q = y[: max(10, len(y) // 4)]
    m = max(float(np.abs(q).max()), 1e-6)
    return min(float(q.min()), 0.0) - 0.5 * m, max(float(q.max()), 0.0) + 1.5 * m


def choose_best(sims, mets, cmets):
    stable = [k for k in sims if sims[k]["stable"]]
    if not stable:
        return None, "no controller is stable with the current gains."

    def key(k):
        return np.inf if np.isnan(mets[k]["ts"]) else mets[k]["ts"]

    for os_lim in (10.0, 25.0):
        cand = [k for k in stable if abs(mets[k]["ess_pct"]) <= 2.0 and mets[k]["os"] <= os_lim]
        if cand:
            b = min(cand, key=key)
            m = mets[b]
            return b, (f"stable, ss error {m['ess_pct']:.2f} %, overshoot {m['os']:.2f} %, "
                       f"settling {fmt(m['ts'], ' s', 3)}, peak current {fmt(cmets[b]['ip'], ' A', 2)}")
    b = min(stable, key=lambda k: abs(mets[k]["ess_pct"]))
    return b, f"smallest steady-state error ({mets[b]['ess_pct']:.2f} %) among the stable controllers"


# ----------------------------------------------------------------------------
# 4. INTERACTIVE GUI
# ----------------------------------------------------------------------------
class MotorGUI:
    def __init__(self):
        self.kind = "PID"
        self.gains = {k: dict(v) for k, v in DEFAULTS.items()}
        self.ref, self.tend, self.L = 1.0, 3.0, L_DEFAULT
        self._busy = False
        self.dyn = []
        self.sim = None
        self.compare_fig = None
        self._build()
        self._refresh_controls()
        self.update()

    # ------------------------------------------------------------------ layout
    def _build(self):
        fig = self.fig = plt.figure(figsize=(15, 9))
        try:
            fig.canvas.manager.set_window_title("DC Motor Speed Control - P / PD / PI / PID (3 states)")
        except Exception:
            pass
        fig.patch.set_facecolor("#f4f6f8")

        ax_info = fig.add_axes([0.02, 0.885, 0.31, 0.10])
        ax_info.axis("off")
        self.info = ax_info.text(0, 1, "", va="top", ha="left", family="monospace", fontsize=8.5,
                                 color="#202124")

        ax_radio = fig.add_axes([0.02, 0.635, 0.27, 0.22], facecolor="#ffffff")
        ax_radio.set_title("1) Select controller", fontsize=11, fontweight="bold", loc="left")
        self.radio = RadioButtons(ax_radio, [LABELS[k] for k in ("P", "PD", "PI", "PID")], active=3,
                                  activecolor=COLORS["PID"])
        for lab in self.radio.labels:
            lab.set_fontsize(10.5)
        self.radio.on_clicked(self._on_radio)

        fig.text(0.02, 0.600, "2) Tune the controller gains (live)", fontsize=11, fontweight="bold")

        def mk(y, label, vmin, vmax, init, color, fmt_="%.2f", step=None):
            ax = fig.add_axes([0.085, y, 0.155, 0.028], facecolor="#e8eaed")
            return Slider(ax, label, vmin, vmax, valinit=init, color=color, valfmt=fmt_, valstep=step)

        self.s_kp = mk(0.560, "Kp", 0.1, 300.0, 100.0, "#1f77b4")
        self.s_ki = mk(0.515, "Ki", 0.0, 2000.0, 0.0, "#17a589")
        self.s_kd = mk(0.470, "Kd", 0.0, 30.0, 0.0, "#ff7f0e")
        fig.text(0.02, 0.432, "Motor parameter & simulation settings", fontsize=10, fontweight="bold")
        self.s_L = mk(0.395, "L (H)", 0.01, 3.0, L_DEFAULT, "#d93025", "%.2f", 0.01)
        self.s_ref = mk(0.350, "Ref\n(rad/s)", 0.5, 100.0, 1.0, "#5f6368")
        self.s_tend = mk(0.305, "T end\n(s)", 0.5, 20.0, 3.0, "#5f6368")

        self.s_kp.on_changed(lambda v: self._on_gain("Kp", v))
        self.s_ki.on_changed(lambda v: self._on_gain("Ki", v))
        self.s_kd.on_changed(lambda v: self._on_gain("Kd", v))
        for s in (self.s_L, self.s_ref, self.s_tend):
            s.on_changed(self._on_sim)

        ax_cmp = fig.add_axes([0.02, 0.200, 0.27, 0.07])
        self.b_cmp = Button(ax_cmp, "3) Compare  P vs PD vs PI vs PID", color="#d2e3fc", hovercolor="#aecbfa")
        self.b_cmp.label.set_fontsize(11)
        self.b_cmp.label.set_fontweight("bold")
        self.b_cmp.on_clicked(lambda _e: self.show_comparison())
        self.b_csv = Button(fig.add_axes([0.02, 0.125, 0.13, 0.05]), "Export CSV", color="#e6f4ea",
                            hovercolor="#ceead6")
        self.b_csv.on_clicked(lambda _e: self.export_csv())
        self.b_rst = Button(fig.add_axes([0.16, 0.125, 0.13, 0.05]), "Reset gains", color="#fce8e6",
                            hovercolor="#f6c7c2")
        self.b_rst.on_clicked(lambda _e: self.reset())
        self.msg = fig.text(0.02, 0.095, "", fontsize=8, color="#5f6368")
        fig.text(0.02, 0.03, "State variables:  x = [ theta (rad),  omega (rad/s),  i (A) ]\n"
                 "Controlled output: omega.   Input: armature voltage V.", fontsize=8.5, color="#5f6368")

        # banner
        self.ax_banner = fig.add_axes([0.35, 0.925, 0.63, 0.055])
        self.ax_banner.set_xticks([])
        self.ax_banner.set_yticks([])
        self.banner = self.ax_banner.text(0.5, 0.5, "", ha="center", va="center", color="white",
                                          fontsize=10.5, fontweight="bold", transform=self.ax_banner.transAxes)

        # the three state plots
        self.ax_w = fig.add_axes([0.37, 0.530, 0.61, 0.370])
        self.ax_th = fig.add_axes([0.37, 0.355, 0.61, 0.135], sharex=self.ax_w)
        self.ax_i = fig.add_axes([0.37, 0.185, 0.61, 0.135], sharex=self.ax_w)
        for ax in (self.ax_w, self.ax_th, self.ax_i):
            ax.grid(True, alpha=0.35)
        plt.setp(self.ax_w.get_xticklabels(), visible=False)
        plt.setp(self.ax_th.get_xticklabels(), visible=False)
        self.ax_w.set_ylabel("omega  (rad/s)")
        self.ax_th.set_ylabel("theta  (rad)")
        self.ax_i.set_ylabel("i  (A)")
        self.ax_i.set_xlabel("Time (s)")
        self.ax_th.set_title("State 1: angular position  theta(t)", fontsize=9.5, loc="left", pad=3)
        self.ax_i.set_title("State 3: armature current  i(t)", fontsize=9.5, loc="left", pad=3)

        (self.line_w,) = self.ax_w.plot([], [], lw=2.4, label="omega")
        self.ref_line = self.ax_w.axhline(1, color="k", ls="--", lw=1.2, label="Reference")
        (self.line_th,) = self.ax_th.plot([], [], lw=2.2, label="theta")
        (self.line_i,) = self.ax_i.plot([], [], lw=2.2, label="i")

        # hover read-out on all three plots
        self.hv = {}
        for ax, key, name, unit in ((self.ax_w, "w", "omega", "rad/s"), (self.ax_th, "th", "theta", "rad"),
                                    (self.ax_i, "i", "i", "A")):
            dot, = ax.plot([], [], "ko", ms=6, zorder=10)
            ann = ax.annotate("", xy=(0, 0), xytext=(18, -34), textcoords="offset points", fontsize=9,
                              bbox=dict(boxstyle="round", fc="#fffde7", ec="#999"),
                              arrowprops=dict(arrowstyle="->", color="#555"), visible=False, zorder=11)
            self.hv[ax] = (dot, ann, key, name, unit)
        fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

        # metrics panel
        self.ax_m = fig.add_axes([0.35, 0.008, 0.63, 0.12])
        self.ax_m.axis("off")
        self.metrics_txt = self.ax_m.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.8,
                                          bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#c4c7c5"),
                                          transform=self.ax_m.transAxes)

    # ---------------------------------------------------------------- controls
    def _set_slider_enabled(self, s, enabled):
        s.active = enabled
        s.poly.set_alpha(1.0 if enabled else 0.2)
        s.label.set_alpha(1.0 if enabled else 0.35)
        s.valtext.set_alpha(1.0 if enabled else 0.35)
        s.ax.set_facecolor("#e8eaed" if enabled else "#f1f3f4")

    def _refresh_controls(self):
        self._busy = True
        g = self.gains[self.kind]
        for name, s in (("Kp", self.s_kp), ("Ki", self.s_ki), ("Kd", self.s_kd)):
            s.set_val(g[name])
            self._set_slider_enabled(s, name in USED[self.kind])
        self._busy = False
        self.fig.canvas.draw_idle()

    def _on_radio(self, label):
        self.kind = label.split()[0]
        self._refresh_controls()
        self.update()

    def _on_gain(self, name, val):
        if self._busy:
            return
        self.gains[self.kind][name] = float(val)
        self.update()

    def _on_sim(self, _val):
        if self._busy:
            return
        self.L, self.ref, self.tend = float(self.s_L.val), float(self.s_ref.val), float(self.s_tend.val)
        self.update()

    def reset(self):
        self.gains[self.kind] = dict(DEFAULTS[self.kind])
        self._refresh_controls()
        self.update()

    def export_csv(self):
        if self.sim is None:
            return
        name = f"dc_motor_{self.kind}_response.csv"
        s = self.sim
        with open(name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "angular_position_rad", "angular_velocity_rad_s", "current_A",
                        "reference_rad_s"])
            for k in range(len(s["t"])):
                w.writerow([f"{s['t'][k]:.6f}", f"{s['th'][k]:.6f}", f"{s['w'][k]:.6f}",
                            f"{s['i'][k]:.6f}", f"{s['ref']:.6f}"])
        self.msg.set_text(f"Saved: {os.path.abspath(name)}")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------- hover
    def _on_hover(self, event):
        if self.sim is None:
            return
        changed = False
        for ax, (dot, ann, key, name, unit) in self.hv.items():
            if event.inaxes is ax and event.xdata is not None:
                t, y = self.sim["t"], self.sim[key]
                k = int(np.argmin(np.abs(t - event.xdata)))
                if abs(y[k]) > 1e6:
                    continue
                dot.set_data([t[k]], [y[k]])
                ann.xy = (t[k], y[k])
                ann.set_text(f"t = {t[k]:.4f} s\n{name} = {y[k]:.4f} {unit}")
                ann.set_visible(True)
                changed = True
            elif ann.get_visible():
                ann.set_visible(False)
                dot.set_data([], [])
                changed = True
        if changed:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ update
    def _clear_dyn(self):
        for a in self.dyn:
            try:
                a.remove()
            except Exception:
                pass
        self.dyn = []

    def _tag(self, ax, xy, text, color="#202124", ha="left", va="bottom", offset=(8, 8)):
        a = ax.annotate(text, xy=xy, xytext=offset, textcoords="offset points", fontsize=8.5, color=color,
                        ha=ha, va=va, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.92),
                        zorder=9)
        self.dyn.append(a)

    def update(self):
        sim = simulate(self.kind, self.gains[self.kind], self.ref, self.tend, self.L)
        self.sim = sim
        m, cm = compute_metrics(sim), current_metrics(sim)
        t, w, th, cur, ref, kind = sim["t"], sim["w"], sim["th"], sim["i"], sim["ref"], sim["kind"]
        self._clear_dyn()

        txt, color = stability_text(sim, kind)
        self.ax_banner.set_facecolor(color)
        self.banner.set_text(txt)
        for ax in (self.ax_w, self.ax_th, self.ax_i):
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.0)

        den = plant_den(self.L)
        self.info.set_text(
            "DC MOTOR   G(s) = w(s)/V(s) = K / [(Js+b)(Ls+R)+K^2]\n"
            f"= {K} / ({den[0]:.4g} s^2 + {den[1]:.4g} s + {den[2]:.4g})\n"
            f"J={J}, b={B}, K={K}, R={R}, L={self.L:.2f}\n"
            "states: theta' = w ; J w' = K i - b w ; L i' = V - R i - K w")

        c = COLORS[kind] if sim["stable"] else RED
        self.line_w.set_data(t, w)
        self.line_w.set_color(c)
        self.line_th.set_data(t, th)
        self.line_th.set_color(c)
        self.line_i.set_data(t, cur)
        self.line_i.set_color(c)
        self.ref_line.set_ydata([ref, ref])
        self.ax_w.set_xlim(0, t[-1])
        self.ax_w.set_title(f"State 2: angular velocity omega(t) - {kind} controller, reference = {ref:g} rad/s",
                            fontsize=10.5, fontweight="bold", loc="left", pad=4)

        if sim["stable"]:
            lo = min(0.0, float(w.min()))
            hi = max(1.2 * ref, float(w.max()) * 1.12)
            self.ax_w.set_ylim(lo - 0.03 * (hi - lo), hi)
            # theta
            hi_t = max(float(th.max()), 1e-9)
            self.ax_th.set_ylim(min(0.0, float(th.min())) - 0.05 * hi_t, hi_t * 1.18)
            self._tag(self.ax_th, (t[-1], th[-1]),
                      f"theta({t[-1]:.1f} s) = {th[-1]:.3f} rad  ({th[-1] / (2 * np.pi):.3f} rev)", "#5f6368",
                      ha="right", va="top", offset=(-8, -10))
            self.dyn.append(self.ax_th.plot([t[-1]], [th[-1]], "o", color="#5f6368", ms=7, zorder=8)[0])
            # current
            ilo, ihi = min(0.0, float(cur.min())), max(float(cur.max()), cm["iss"], 1e-9)
            padi = 0.18 * max(ihi - ilo, 1e-9)
            self.ax_i.set_ylim(ilo - 0.12 * padi, ihi + padi)
            self.dyn.append(self.ax_i.plot([t[0], t[-1]], [cm["iss"]] * 2, ":", color="#5f6368", lw=1.2)[0])
            self._tag(self.ax_i, (t[-1] * 0.99, cm["iss"]), f"steady-state i = {cm['iss']:.3f} A", "#5f6368",
                      ha="right", va="top", offset=(0, -6))
            self.dyn.append(self.ax_i.plot([cm["tip"]], [cm["ip"]], "v", color=RED, ms=9, zorder=8)[0])
            self._tag(self.ax_i, (cm["tip"], cm["ip"]), f"peak i = {cm['ip']:.3f} A @ {cm['tip']:.3f} s", RED,
                      offset=(10, -4), va="top")

            yss = m["yss"]
            self.dyn.append(self.ax_w.fill_between(t, 0.98 * yss, 1.02 * yss, color="#9aa0a6", alpha=0.18,
                                                   label="+/-2 % band"))
            self.dyn.append(self.ax_w.plot([t[0], t[-1]], [yss, yss], ":", color="#5f6368", lw=1.2)[0])
            self._tag(self.ax_w, (t[-1] * 0.98, yss), f"final value = {yss:.4f}", "#5f6368", ha="right", va="top")
            if not np.isnan(m["tr"]):
                self.dyn.append(self.ax_w.plot([m["t10"], m["t90"]], [0.1 * yss, 0.9 * yss], "s",
                                               color="#1a73e8", ms=7, zorder=8)[0])
                self._tag(self.ax_w, (m["t90"], 0.9 * yss), f"rise time = {m['tr']:.4f} s", "#1a73e8", va="top")
            if m["os"] > 0.05:
                self.dyn.append(self.ax_w.plot([m["tp"]], [m["yp"]], "v", color=RED, ms=9, zorder=8)[0])
                self._tag(self.ax_w, (m["tp"], m["yp"]),
                          f"overshoot = {m['os']:.2f} %\npeak {m['yp']:.3f} @ {m['tp']:.3f} s", RED)
            if not np.isnan(m["ts"]) and m["ts"] > 0:
                self.dyn.append(self.ax_w.axvline(m["ts"], color="#e37400", ls="--", lw=1.3))
                self.dyn.append(self.ax_w.plot([m["ts"]], [float(np.interp(m["ts"], t, w))], "o",
                                               color="#e37400", ms=7, zorder=8)[0])
                self._tag(self.ax_w, (m["ts"], self.ax_w.get_ylim()[0]), f"settling (2 %) = {m['ts']:.4f} s",
                          "#e37400")
            if abs(m["ess"]) > 0.005 * ref:
                xe = t[-1] * 0.9
                self.dyn.append(self.ax_w.annotate("", xy=(xe, ref), xytext=(xe, yss),
                                                   arrowprops=dict(arrowstyle="<->", color="k", lw=1.4)))
                self._tag(self.ax_w, (xe, (ref + yss) / 2), f"ss error = {m['ess']:.4f}\n({m['ess_pct']:.2f} %)",
                          "k", ha="right", va="center")
        else:
            self.ax_w.set_ylim(-1.0 * ref, 3.0 * ref)
            self.ax_th.set_ylim(*robust_ylim(th))
            self.ax_i.set_ylim(*robust_ylim(cur))
            self._tag(self.ax_w, (t[-1] * 0.5, 2.6 * ref), "UNSTABLE - response diverges", RED, ha="center",
                      va="center")

        handles, labels = self.ax_w.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
        self.ax_w.legend(*zip(*keep), loc="lower right" if sim["stable"] else "upper left", fontsize=8.5)

        # metrics panel
        pol = ", ".join(f"{p.real:.2f}{p.imag:+.2f}j" if abs(p.imag) > 1e-9 else f"{p.real:.2f}"
                        for p in sim["poles"])
        head = f"Controller : {kind}  ({controller_string(kind, sim['g'])}),  L = {self.L:.2f} H\n"
        if sim["stable"]:
            ts_txt = fmt(m["ts"], " s") if not np.isnan(m["ts"]) else "not settled (raise T end)"
            body = (head +
                    f"Speed      : rise (10-90%) {fmt(m['tr'], ' s')} | settling (2%) {ts_txt} | "
                    f"overshoot {fmt(m['os'], ' %', 2)}\n"
                    f"Speed ss   : value {fmt(m['yss'], ' rad/s')} | error {fmt(m['ess'], ' rad/s')} "
                    f"({fmt(m['ess_pct'], ' %', 2)} of ref)\n"
                    f"Current    : peak {fmt(cm['ip'], ' A', 3)} at {fmt(cm['tip'], ' s', 3)} | "
                    f"steady-state {fmt(cm['iss'], ' A', 3)}\n"
                    f"Position   : theta({t[-1]:.1f} s) = {th[-1]:.3f} rad ({th[-1] / (2 * np.pi):.3f} rev)  "
                    f"[keeps growing while the motor runs]\n"
                    f"Poles      : {pol}")
        else:
            body = (head + "Status     : UNSTABLE -> performance metrics are not defined\n"
                    f"Poles      : {pol}\nHint       : keep every pole's real part negative.")
        self.metrics_txt.set_text(body)
        self.metrics_txt.get_bbox_patch().set_edgecolor(color)
        self.metrics_txt.get_bbox_patch().set_linewidth(2)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------- comparison
    def show_comparison(self):
        if self.compare_fig is not None and plt.fignum_exists(self.compare_fig.number):
            plt.close(self.compare_fig)
        ref, tend, L = self.ref, self.tend, self.L
        kinds = ["P", "PD", "PI", "PID"]
        sims = {k: simulate(k, self.gains[k], ref, tend, L) for k in kinds}
        mets = {k: compute_metrics(sims[k]) for k in kinds}
        cmets = {k: current_metrics(sims[k]) for k in kinds}

        fig = plt.figure(figsize=(14, 10))
        self.compare_fig = fig
        try:
            fig.canvas.manager.set_window_title("Comparison: P vs PD vs PI vs PID")
        except Exception:
            pass
        axw = fig.add_axes([0.07, 0.675, 0.90, 0.265])
        axt = fig.add_axes([0.07, 0.470, 0.90, 0.170], sharex=axw)
        axi = fig.add_axes([0.07, 0.270, 0.90, 0.170], sharex=axw)
        for ax in (axw, axt, axi):
            ax.grid(True, alpha=0.35)
        axw.axhline(ref, color="k", ls="--", lw=1.3, label=f"Reference ({ref:g} rad/s)")

        lim = {"w": [ref * 1.25, 0.0], "th": [1e-9, 0.0], "i": [1e-9, 0.0]}  # [max, min]
        for k in kinds:
            s = sims[k]
            lab = f"{k}  ({controller_string(k, s['g'])})"
            kw = dict(lw=2.3, color=COLORS[k], label=lab) if s["stable"] else \
                dict(lw=1.8, color=COLORS[k], ls=(0, (4, 2)), label=lab + "  - UNSTABLE")
            for ax, key in ((axw, "w"), (axt, "th"), (axi, "i")):
                ax.plot(s["t"], s[key], **kw)
                if s["stable"]:
                    lim[key][0] = max(lim[key][0], float(s[key].max()) * 1.1)
                    lim[key][1] = min(lim[key][1], float(s[key].min()))
        axw.set_xlim(0, tend)
        for ax, key in ((axw, "w"), (axt, "th"), (axi, "i")):
            ax.set_ylim(lim[key][1] - 0.04 * lim[key][0], lim[key][0])
        plt.setp(axw.get_xticklabels(), visible=False)
        plt.setp(axt.get_xticklabels(), visible=False)
        axw.set_ylabel("omega (rad/s)")
        axt.set_ylabel("theta (rad)")
        axi.set_ylabel("current i (A)")
        axi.set_xlabel("Time (s)")
        axw.set_title(f"DC motor - P vs PD vs PI vs PID  (L = {L:.2f} H):  angular velocity, position and current",
                      fontsize=12.5, fontweight="bold")
        axw.legend(loc="lower right", fontsize=8.5)

        axtab = fig.add_axes([0.02, 0.075, 0.96, 0.155])
        axtab.axis("off")
        cols = ["Controller", "Gains", "Rise (s)", "Settling (s)", "Overshoot (%)", "ss error",
                "Peak i (A)", "ss i (A)", "Stability"]
        rows, cc = [], []
        for k in kinds:
            m, cm, s = mets[k], cmets[k], sims[k]
            if s["stable"]:
                ts = fmt(m["ts"]) if not np.isnan(m["ts"]) else "> T end"
                rows.append([k, controller_string(k, s["g"]), fmt(m["tr"]), ts, fmt(m["os"], nd=2),
                             f"{m['ess_pct']:.2f} %", fmt(cm["ip"], nd=2), fmt(cm["iss"], nd=2), "STABLE"])
                cc.append(["#ffffff"] * 8 + ["#ceead6"])
            else:
                rows.append([k, controller_string(k, s["g"])] + ["N/A"] * 6 + ["UNSTABLE"])
                cc.append(["#ffffff"] * 8 + ["#f6c7c2"])
        tb = axtab.table(cellText=rows, colLabels=cols, cellColours=cc, loc="center", cellLoc="center",
                         colColours=["#dadce0"] * len(cols))
        tb.auto_set_font_size(False)
        tb.set_fontsize(9.5)
        tb.scale(1, 1.7)
        tb.auto_set_column_width(list(range(len(cols))))

        best, why = choose_best(sims, mets, cmets)
        col = GREEN if best else RED
        fig.text(0.5, 0.032, f"Most suitable controller:  {best or 'none'}  -  {why}", ha="center", fontsize=10.5,
                 fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.5", fc=col, ec=col))
        fig.text(0.5, 0.006, "Uses the gains currently set for each controller (select one, tune it, press Compare "
                 "again).  theta grows without limit while the motor runs - this is expected.",
                 ha="center", fontsize=8, color="#5f6368")
        fig.show()


def main():
    gui = MotorGUI()  # keep a reference so the widgets are not garbage-collected
    plt.show()
    return gui


if __name__ == "__main__":
    main()
