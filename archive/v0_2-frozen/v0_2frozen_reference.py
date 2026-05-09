"""
Data Singularity v0.2i-p2
補丁集：
  1. Φ_carry < 0 補丁（abs + sign_mem π-flip）
  2. 穿梭即翻轉（sign_mem_new = -ψ(Ψ_impact)）
  3. 觸發前狀態監視器（RGB 機會視窗偵測）
  4. 因果力對觀察指標（F_net / F_sum = P_ratio）
  5. ΔH_norm 加入 Ψ_impact（β = 1/(2·Γ_L)）

公式來源：交接摘要 20260506、穿梭架構 v0.2、討論-8
"""

import math

N_BASE = 3
EPS0   = 1e-9

def gamma(L):
    return N_BASE ** L

def psi(x, threshold):
    if x > threshold:  return 1
    elif x < -threshold: return -1
    else: return 0

NEAR_THR = 0.75

def check_rgb_window(step, L, GL, eps_L, H_dia, Phi_total,
                     delta_norm, alpha_impact, prev_dPhi, dPhi,
                     P_ratio, rgb_windows):
    Hprog    = H_dia / GL if GL > 0 else 0.0
    Pprog    = Phi_total / (GL * math.pi) if GL > 0 else 0.0
    sync_gap = abs(Hprog - Pprog)
    Psi_pot  = delta_norm + alpha_impact

    in_near_zone = (Hprog > NEAR_THR and Pprog > NEAR_THR)
    if not in_near_zone:
        return

    if Psi_pot > 0:
        window = {
            "step": step, "L": L,
            "Psi_pot": Psi_pot,
            "delta_norm": delta_norm, "alpha": alpha_impact,
            "Hprog": Hprog, "Pprog": Pprog,
            "sync_gap": sync_gap, "sync_ok": sync_gap < eps_L,
            "dPhi": dPhi, "prev_dPhi": prev_dPhi,
            "P_ratio": P_ratio,
            "type": "RGB_WINDOW",
        }
        rgb_windows.append(window)
        sync_tag = "✓同步" if sync_gap < eps_L else f"✗sync={sync_gap:.4f}>ε={eps_L:.4f}"
        print(f"  [RGB] s={step} Ψ={Psi_pot:+.4f}"
              f"  δ={delta_norm:+.4f} α={alpha_impact:+.5f}"
              f"  P={P_ratio:+.4f}  H={Hprog:.3f} Φ={Pprog:.3f}  {sync_tag}")

    elif sync_gap < eps_L and Psi_pot < 0:
        window = {
            "step": step, "L": L,
            "Psi_pot": Psi_pot,
            "delta_norm": delta_norm, "alpha": alpha_impact,
            "Hprog": Hprog, "Pprog": Pprog,
            "sync_gap": sync_gap, "sync_ok": True,
            "P_ratio": P_ratio,
            "type": "SYNC_CMY",
        }
        rgb_windows.append(window)


def simulate(input_seq, max_levels=6, use_eps=True, label="",
             verbose_monitor=True, use_dH_norm=False):
    L         = 1
    GL        = gamma(L)
    eps_L     = 1.0 / (2 * GL)
    dead_zone = 0.25 / GL

    Hplus     = 0.0
    Hminus    = 0.0
    nR        = float(GL)
    nL        = float(GL)
    Phi_total = 0.0
    sign_mem  = 0
    prev_dPhi = 0.0

    transitions = []
    rgb_windows = []

    beta = 1.0 / (2 * GL)   # ΔH_norm 係數（隨層更新）

    print(f"\n{'='*60}")
    mode_tag = "+ΔH" if use_dH_norm else ""
    print(f"[{label}{mode_tag}] L={L} Γ={GL} ε={eps_L:.4f} β={beta:.4f}")
    print(f"{'='*60}")

    for step, s_ext in enumerate(input_seq):

        # ── 內外雙生 ──────────────────────────────
        n_s = nR if s_ext == 1 else (nL if s_ext == -1 else 0.0)
        s_total = s_ext * 2 * n_s / (nL + nR) if (nL + nR) > 0 else 0.0

        if s_total > 0:   Hplus  += s_total
        elif s_total < 0: Hminus += (-s_total)

        nR = GL + Hplus
        nL = GL + Hminus

        # ── 因果力對 F_net/F_sum（壓差比 P_ratio）──
        _df   = (nL + nR) if (nL + nR) > 0 else EPS0
        F_rep = GL * Hplus  * nL / _df
        F_att = GL * Hminus * nR / _df
        F_net = F_rep - F_att
        F_sum = F_rep + F_att
        P_ratio = F_net / F_sum if abs(F_sum) > EPS0 else 0.0
        # 代數確認：F_net = GL² · δ_norm（湧現值）

        # ── H_dialogue ────────────────────────────
        H_dia = math.sqrt(Hplus * Hminus) if (Hplus > 0 and Hminus > 0) else 0.0

        # ── δ_norm / δ_d ──────────────────────────
        delta_norm = (nR - nL) / (nL + nR) if (nL + nR) > 0 else 0.0
        delta_d    = psi(delta_norm, dead_zone)

        # ── sign_mem 首觸坍縮 / 幾何零點翻轉 ──────
        prev_sign_mem = sign_mem
        if sign_mem == 0 and delta_d != 0:
            sign_mem = delta_d
        elif prev_sign_mem != 0 and delta_d != 0 and delta_d != prev_sign_mem:
            sign_mem = delta_d

        # ── t, 0_k ────────────────────────────────
        t  = (-1.0/nL + 1.0/nR) if (nL > 0 and nR > 0) else 0.0
        r  = (nL + nR) / 2.0
        ok = t * r if abs(t) >= EPS0 else 0.0

        # ── θ（維克翻轉）─────────────────────────
        w     = math.tanh(abs(t) * N_BASE)
        denom = (nR - nL) + EPS0 * delta_d if delta_d != 0 else (nR - nL) + EPS0
        theta = w * math.atan(ok / denom) + (1 - w) * (math.pi / 2) * sign_mem

        # ── ΔΦ ────────────────────────────────────
        dPhi         = math.pi * abs(delta_d) * abs(math.sin(theta))
        Phi_total   += dPhi
        alpha_impact = (dPhi - prev_dPhi) / math.pi

        # ── 觸發前監視器 ──────────────────────────
        if verbose_monitor:
            check_rgb_window(
                step, L, GL, eps_L,
                H_dia, Phi_total,
                delta_norm, alpha_impact,
                prev_dPhi, dPhi,
                P_ratio, rgb_windows
            )

        # ── 觸發條件 ──────────────────────────────
        Hprog    = H_dia / GL if GL > 0 else 0.0
        Pprog    = Phi_total / (GL * math.pi) if GL > 0 else 0.0
        sync_gap = abs(Hprog - Pprog)

        triggered = (
            H_dia     >= GL and
            Phi_total >= GL * math.pi and
            (not use_eps or sync_gap < eps_L)
        )

        if triggered:
            # ── ΔH_norm 項（可選）─────────────────
            dH_sum  = Hplus + Hminus + EPS0
            dH_norm = (Hplus - Hminus) / dH_sum
            beta_now = 1.0 / (2 * GL)
            dH_term  = beta_now * dH_norm if use_dH_norm else 0.0

            Psi_impact = delta_norm + alpha_impact + dH_term

            GL_next      = gamma(L + 1)
            sign_mem_new = -psi(Psi_impact, 1.0 / (2 * GL_next))

            Phi_carry_geo = Phi_total - GL * math.pi
            Phi_carry_dyn = Phi_carry_geo * (1 + Psi_impact)

            phase_flipped = False
            if Phi_carry_dyn < 0:
                Phi_carry_dyn = abs(Phi_carry_dyn)
                sign_mem_new  = -sign_mem_new
                phase_flipped = True

            Phi_new  = Phi_carry_dyn % (2 * math.pi)
            eta_geo  = Phi_carry_geo / (GL * math.pi)
            eta_dyn  = Phi_carry_dyn / (GL * math.pi)
            shuttle  = ("RGB破甲" if Psi_impact > 0
                        else ("守恆" if Psi_impact == 0 else "CMY衰減"))
            flip_tag = " [π-FLIP]" if phase_flipped else ""
            dH_tag   = f"  ΔH_norm={dH_norm:+.4f}(×β={beta_now:.4f}→{dH_term:+.5f})" if use_dH_norm else ""

            print(f"\n>>> 躍遷 L{L}→L{L+1}  step={step}")
            print(f"    Ψ={Psi_impact:.4f}  δ={delta_norm:.4f}  α={alpha_impact:.5f}{dH_tag}")
            print(f"    P_ratio={P_ratio:+.4f}  F_net={F_net:.4f}  F_sum={F_sum:.4f}")
            print(f"    η_geo={eta_geo:.4f}  η_dyn={eta_dyn:.6f}  sign_mem_new={sign_mem_new}{flip_tag}")
            print(f"    Φ_carry_geo={Phi_carry_geo:.4f}  Φ_carry_dyn={Phi_carry_dyn:.6f}")
            print(f"    穿梭：{shuttle}  arg(H_k)={math.degrees(math.atan2(Hminus,Hplus)):.1f}°")

            transitions.append({
                "step": step, "L": L,
                "Psi": Psi_impact, "delta_norm": delta_norm,
                "alpha": alpha_impact, "dH_norm": dH_norm,
                "P_ratio": P_ratio, "F_net": F_net, "F_sum": F_sum,
                "eta_geo": eta_geo, "eta_dyn": eta_dyn,
                "sign_mem_new": sign_mem_new,
                "phase_flipped": phase_flipped, "shuttle": shuttle,
            })

            L         += 1
            GL         = gamma(L)
            eps_L      = 1.0 / (2 * GL)
            dead_zone  = 0.25 / GL
            beta       = 1.0 / (2 * GL)
            Hplus      = 0.0
            Hminus     = 0.0
            nR         = float(GL)
            nL         = float(GL)
            Phi_total  = Phi_new
            sign_mem   = sign_mem_new
            prev_dPhi  = 0.0

            print(f"    ↳ 新層 L{L} 起步：sign_mem={sign_mem}"
                  f"  方向={'RGB' if sign_mem==1 else ('CMY' if sign_mem==-1 else '中性')}")

            if L > max_levels:
                print(f"\n[達到最大層級 {max_levels}，停止]")
                break
        else:
            prev_dPhi = dPhi

    # ── 監視器摘要 ───────────────────────────────
    if verbose_monitor:
        rgb_only  = [w for w in rgb_windows if w["type"] == "RGB_WINDOW"]
        sync_cmy  = [w for w in rgb_windows if w["type"] == "SYNC_CMY"]
        print(f"\n  [監視器摘要] RGB視窗={len(rgb_only)}次"
              f"  其中同步✓={sum(1 for w in rgb_only if w['sync_ok'])}次"
              f"  | 同步CMY={len(sync_cmy)}次")

        # P_ratio 與 sync_gap 相關分析
        if rgb_only:
            best = max(rgb_only, key=lambda w: w["Psi_pot"])
            # 在 RGB 視窗期間計算 P_ratio 與 sync_gap 的方向一致性
            same_sign = sum(1 for w in rgb_only
                           if (w["P_ratio"] > 0) == (w["sync_gap"] > 0))
            print(f"  [壓差分析] RGB視窗中 P_ratio>0 且 sync_gap>0 同向："
                  f"{same_sign}/{len(rgb_only)} ({100*same_sign/len(rgb_only):.0f}%)")
            avg_P  = sum(w["P_ratio"]  for w in rgb_only) / len(rgb_only)
            avg_sg = sum(w["sync_gap"] for w in rgb_only) / len(rgb_only)
            print(f"  [平均值]  P_ratio={avg_P:+.4f}  sync_gap={avg_sg:.4f}")
            print(f"  [最佳RGB] step={best['step']} L={best['L']}"
                  f"  Ψ={best['Psi_pot']:+.4f}  P={best['P_ratio']:+.4f}"
                  f"  sync={'✓' if best['sync_ok'] else '✗'}")
        else:
            print(f"  [壓差分析] 未偵測到 RGB 視窗")

    print(f"\n[{label}] 共 {len(transitions)} 次躍遷")
    return transitions, rgb_windows


# ── 輸入序列 ──────────────────────────────────────
def make_B4(n):   return [1, -1] * n
def make_B3(n):   return [1, 1, -1, -1] * n
def make_A1(n):   return [1, 1, 1, -1] * n
def make_dense(n): return [1, -1] * n


# ── 主測試 ────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("【測試組 1】B4 基準 — F_net/F_sum 觀察（use_dH_norm=False）")
    print("="*60)
    t1, w1 = simulate(make_B4(100), label="B4", verbose_monitor=True, use_dH_norm=False)

    print("\n" + "="*60)
    print("【測試組 2】B4 + ΔH_norm（use_dH_norm=True，對比組 1）")
    print("="*60)
    t2, w2 = simulate(make_B4(100), label="B4", verbose_monitor=True, use_dH_norm=True)

    print("\n" + "="*60)
    print("【測試組 3】密集長序列 — 觀察各層壓差演化")
    print("="*60)
    t3, w3 = simulate(make_dense(800), label="密集", verbose_monitor=True, use_dH_norm=False)

    print("\n" + "="*60)
    print("【測試組 4】守門員確認")
    print("="*60)
    simulate([1]*200, label="純+1", verbose_monitor=False, use_dH_norm=False)
    simulate([-1]*200, label="純-1", verbose_monitor=False, use_dH_norm=False)

    # ── 跨組比較摘要 ─────────────────────────────
    print("\n" + "="*60)
    print("【跨組比較】Ψ_impact 對比（不含 vs 含 ΔH_norm）")
    print("="*60)
    for tr1, tr2 in zip(t1, t2):
        delta_Psi = tr2["Psi"] - tr1["Psi"]
        print(f"  L{tr1['L']}→L{tr1['L']+1}  "
              f"Ψ(原)={tr1['Psi']:+.5f}  "
              f"Ψ(+ΔH)={tr2['Psi']:+.5f}  "
              f"Δ={delta_Psi:+.6f}  "
              f"dH_norm={tr2['dH_norm']:+.4f}  "
              f"P_ratio={tr1['P_ratio']:+.4f}")
