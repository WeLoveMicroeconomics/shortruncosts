#import streamlit as st
from sympy import (
    symbols,
    Eq,
    solve,
    diff,
    latex,
    Min,
    Pow,
    lambdify,
    Piecewise,
    nan,
    Rational,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import numpy as np
import matplotlib.pyplot as plt
# import os

# 1) Symbolic setup
x, y = symbols("x y", real=True, nonnegative=True)
w, r, Q = symbols("w r Q", positive=True)

st.title("Short-Run Cost Minimization")

# 2) User inputs
f_input = st.text_input(
    "Enter production function f(x, y):",
    value="x**0.5 * y**0.5",  # default
)
y0 = st.number_input("Fixed value of y (e.g., capital)", value=1.0, min_value=0.0)
w_val = st.number_input("Value for w (wage)", value=1.0)
r_val = st.number_input("Value for r (capital cost)", value=1.0)
Q_val = st.number_input("Value for Q (output level)", value=1.0)

transformations = standard_transformations + (implicit_multiplication_application,)

try:
    # 3) Parse the raw input (do NOT simplify yet)
    raw_f = parse_expr(
        f_input.replace("^", "**"),
        transformations=transformations,
        local_dict={"x": x, "y": y, "min": Min},
    )
except Exception as e:
    st.error(f"Error parsing production function: {e}")
    st.stop()

# 4) Helper: attempt to match exactly (x^rho + y^rho)^(1/rho)
def match_ces(expr):
    """
    If expr is exactly (x^rho + y^rho)^(1/rho), return (rho, gamma=1).
    Otherwise return None.
    """
    if not isinstance(expr, Pow):
        return None

    base = expr.base
    exp = expr.exp
    # We want base = x**rho + y**rho, and exp = 1/rho.
    if not base.is_Add:
        return None

    terms = list(base.args)
    # find term containing x and term containing y
    xt = next((t for t in terms if t.has(x)), None)
    yt = next((t for t in terms if t.has(y)), None)
    if xt is None or yt is None:
        return None

    # xt should be x**rho, yt should be y**rho
    if not (xt.is_Pow and yt.is_Pow):
        return None

    xb, xe = xt.as_base_exp()
    yb, ye = yt.as_base_exp()
    if xb != x or yb != y:
        return None

    # check that xe == ye  (same exponent on x and on y)
    if (xe - ye).simplify() != 0:
        return None

    rho_candidate = xe
    # outer exponent must equal 1/rho
    # i.e. exp * rho_candidate == 1
    if (exp * rho_candidate).simplify() != 1:
        return None

    # set gamma = 1 (since exp = 1/rho => exp * rho = 1)
    gamma_candidate = Rational(1, 1)
    return (rho_candidate, gamma_candidate)

# 5) Try a direct CES match on the raw parse
detected_ces = False
rho = None
gamma = None

ces_params = match_ces(raw_f)
if ces_params is not None:
    rho, gamma = ces_params
    detected_ces = True

# 6) If that failed, look inside any Pow subexpressions of raw_f
if not detected_ces:
    for sub in raw_f.atoms(Pow):
        ces_params = match_ces(sub)
        if ces_params is not None:
            rho, gamma = ces_params
            detected_ces = True
            break

if detected_ces:
    st.markdown(
        f"$$\\text{{CES detected: }}\\rho = {latex(rho)},\\quad \\gamma = {latex(gamma)}$$"
    )

# 7) Now substitute y → y0
f_sub = raw_f.subs(y, y0)
st.markdown(f"Parsed production function: $f(x, y_0) = {latex(f_sub)}$")

# 8) Solve f(x, y0) = Q for x
try:
    # (a) Leontief / Min-case
    if f_sub.has(Min):
        args = f_sub.args
        if len(args) != 2:
            raise ValueError("min() must have exactly two arguments.")
        a_expr, b_expr = args
        if a_expr.has(x) and not b_expr.has(x):
            a = a_expr.coeff(x)
            b_val = float(b_expr.evalf())
        elif b_expr.has(x) and not a_expr.has(x):
            a = b_expr.coeff(x)
            b_val = float(a_expr.evalf())
        else:
            raise ValueError("min(...) must be between a function of x and a constant.")

        # if Q > maximum producible b_val, stop
        if Q_val > b_val:
            st.error(
                f"Cannot produce Q = {Q_val:.3f}. "
                f"Maximum output = {b_val:.3f} under the min(...) constraint."
            )
            st.stop()
        else:
            x_star = Q / a

    # (b) “Linear-in-x” case (affine: f_sub = a*x + b)
    elif f_sub.is_Add and all(
        (term.as_poly(x) is not None and term.as_poly(x).total_degree() <= 1)
        for term in f_sub.args
    ):
        a = f_sub.coeff(x)
        b = f_sub.subs(x, 0)

        if a == 0:
            # f_sub ≡ b is constant
            b_val = float(b.evalf()) if b.is_number else None
            if b_val is None:
                raise ValueError("Cannot interpret constant term b.")
            if Q_val > b_val:
                st.error(f"Cannot produce Q = {Q_val:.3f}; maximum is {b_val:.3f}.")
                st.stop()
            else:
                x_star = 0
        else:
            x_star = Piecewise(((Q - b) / a, Q >= b), (0, True))

    # (c) CES case (we now know f_sub = (x^rho + y0^rho)^(1/rho) up to our match)
    elif detected_ces:
        # For a two-input CES without weights:
        #   f(x,y0) = (x^rho + y0^rho)^(1/rho)
        # Solve  (x^rho + y0^rho)^(1/rho) = Q
        #  ↦ x^rho + y0^rho = Q^rho
        #  ↦ x^rho = Q^rho − y0^rho
        #  ↦ x = (Q^rho − y0^rho)^(1/rho).
        #
        # Domain:
        #  • If rho > 0, need Q^rho ≥ y0^rho  ⇔  Q ≥ y0  ⇒  x = 0 for Q ≤ y0, interior for Q > y0.
        #  • If rho < 0, need Q^rho ≥ y0^rho  ⇔  Q ≤ y0  ⇒  x = 0 for Q = y0, interior for Q < y0,
        #    and infeasible if Q > y0.

        if rho > 0:
            x_star = Piecewise(
                (0, Q <= y0),                            # capital alone suffices if Q ≤ y0
                ((Q ** rho - y0**rho) ** (1 / rho), True)  # interior solution if Q > y0
            )
        else:
            # rho < 0
            x_star = Piecewise(
                (nan, Q > y0),                            # infeasible if Q > y0
                (0, Q == y0),                             # at Q=y0, x=0
                ((Q ** rho - y0**rho) ** (1 / rho), True)  # interior if Q < y0
            )

    # (d) General solve
    else:
        sol = solve(Eq(f_sub, Q), x, dict=True)
        x_solutions = [s[x] for s in sol if (s[x].is_real and s[x] >= 0)]
        if not x_solutions:
            raise ValueError("No non-negative real solution for x.")
        x_star = x_solutions[0]

except Exception as e:
    st.error(f"Failed to solve f(x, y₀) = Q: {e}")
    st.stop()

# 9) Short-run cost, efficiency, marginal cost
cost_expr = r * y0 + w * x_star
eff_expr = cost_expr / Q
dc_dq_expr = diff(cost_expr, Q)

st.markdown("### Symbolic Results")
st.latex(f"x^* = {latex(x_star)}")
st.latex(f"c(Q) = {latex(cost_expr)}")
st.latex(f"c(Q)/Q = {latex(eff_expr)}")
st.latex(f"dc/dQ = {latex(dc_dq_expr)}")

# 10) Numeric lambdify & evaluation
cost_func = lambdify((w, r, Q), cost_expr, modules=["numpy", "sympy"])
eff_func = lambdify((w, r, Q), eff_expr, modules=["numpy", "sympy"])
marg_func = lambdify((w, r, Q), dc_dq_expr, modules=["numpy", "sympy"])

try:
    cost_val = float(cost_func(w_val, r_val, Q_val))
    eff_val = float(eff_func(w_val, r_val, Q_val))
    marg_val = float(marg_func(w_val, r_val, Q_val))
    st.markdown("### Numerical Evaluation")
    st.latex(f"c(Q) = {cost_val:.4f}")
    st.latex(f"c(Q)/Q = {eff_val:.4f}")
    st.latex(f"dc/dQ = {marg_val:.4f}")
except Exception as e:
    st.error(f"Numerical evaluation failed: {e}")

# 11) Plot settings
file_name1 = st.text_input(
    "Filename for c(Q) plot (e.g. short_run_cost.png):",
    value="short_run_cost.png",
)
file_name2 = st.text_input(
    "Filename for Efficiency/Marginal plot (e.g. efficiency_marginal_short_run.png):",
    value="efficiency_marginal_short_run.png",
)
Q_max = st.number_input("Maximum Q for plots:", value=10.0, min_value=0.1, key="Q_max")
Y_max1 = st.number_input("Max vertical axis (c(Q) plot):", value=20.0, min_value=0.1, key="Y_max1")
Y_max2 = st.number_input(
    "Max vertical axis (Efficiency/Marginal plot):",
    value=20.0,
    min_value=0.1,
    key="Y_max2",
)

# 12) Build Q grid (respecting Min-case infeasibility)
if f_sub.has(Min):
    try:
        args = f_sub.args
        a_expr, b_expr = args
        if a_expr.has(x) and not b_expr.has(x):
            b_term = b_expr.subs(y, y0)
        elif b_expr.has(x) and not a_expr.has(x):
            b_term = a_expr.subs(y, y0)
        else:
            b_term = None

        if b_term is not None:
            Q_cutoff = float(b_term.evalf())
            Q_vals = np.linspace(0.1, min(Q_max, Q_cutoff), 200)
        else:
            Q_vals = np.linspace(0.1, Q_max, 200)
    except Exception:
        Q_vals = np.linspace(0.1, Q_max, 200)
else:
    Q_vals = np.linspace(0.1, Q_max, 200)

# 13) Evaluate c, c/Q, dc/dQ on the grid (nan where it fails)
c_vals, eff_vals, marg_vals = [], [], []
for q in Q_vals:
    try:
        c = float(cost_func(w_val, r_val, q))
        e = float(eff_func(w_val, r_val, q))
        m = float(marg_func(w_val, r_val, q))
    except Exception:
        c, e, m = np.nan, np.nan, np.nan
    c_vals.append(c)
    eff_vals.append(e)
    marg_vals.append(m)

# 14) Plot 1: c(Q) vs Q
fig1, ax1 = plt.subplots()
if f_sub.has(Min):
    args = f_sub.args
    a_expr, b_expr = args
    if a_expr.has(x) and not b_expr.has(x):
        b_term = b_expr.subs(y, y0)
    elif b_expr.has(x) and not a_expr.has(x):
        b_term = a_expr.subs(y, y0)
    else:
        b_term = None
    try:
        if b_term is not None:
            Q_cutoff = float(b_term.evalf())
            ax1.axvline(Q_cutoff, color="red", linestyle="--", label=f"Max Q = {Q_cutoff:.2f}")
    except Exception:
        pass

ax1.plot(Q_vals, c_vals, label="c(Q)")
ax1.set_xlabel("Q")
ax1.set_ylabel("Cost")
ax1.set_title("Short-Run Cost vs Output")
ax1.set_ylim(0, Y_max1)
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# if st.button("Save c(Q) plot"):
 #   base_name1 = os.path.basename(file_name1)
  #  try:
   #     fig1.savefig(base_name1, dpi=300, bbox_inches="tight")
    #    with open(base_name1, "rb") as f:
     #       st.download_button(
      #          "Download c(Q) Plot", f, file_name=base_name1, mime="image/png"
       #     )
  #  except Exception as e:
   #     st.error(f"Failed to save or download c(Q) plot: {e}")

# 15) Plot 2: Efficiency and Marginal Cost
fig2, ax2 = plt.subplots()

# Only draw the vertical MC line if it's a Min-case and not (rho < 0 CES)
if f_sub.has(Min) and not (detected_ces and rho < 0):
    args = f_sub.args
    a_expr, b_expr = args
    if a_expr.has(x) and not b_expr.has(x):
        b_term = b_expr.subs(y, y0)
    elif b_expr.has(x) and not a_expr.has(x):
        b_term = a_expr.subs(y, y0)
    else:
        b_term = None

    try:
        if b_term is not None:
            Q_cutoff = float(b_term.evalf())
            mc_val = float(marg_func(w_val, r_val, Q_cutoff))
            ax2.vlines(
                Q_cutoff,
                mc_val,
                Y_max2,
                color="orange",
                linestyle="-",
                label=f"Vertical MC at Q = {Q_cutoff:.2f}",
            )
    except Exception:
        pass

ax2.plot(Q_vals, eff_vals, label="c(Q)/Q")
ax2.plot(Q_vals, marg_vals, label="dc/dQ", color="orange")

ax2.set_xlabel("Q")
ax2.set_ylabel("Cost Derivatives")
ax2.set_title("Efficiency and Marginal Cost")
ax2.set_ylim(0, Y_max2)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# if st.button("Save Efficiency/Marginal plot"):
 #   base_name2 = os.path.basename(file_name2)
  #  try:
   #     fig2.savefig(base_name2, dpi=300, bbox_inches="tight")
    #    with open(base_name2, "rb") as f:
     #       st.download_button(
      #          "Download Efficiency/Marginal Plot",
       #         f,
        #        file_name=base_name2,
         #       mime="image/png",
          #  )
   # except Exception as e:
    #    st.error(f"Failed to save or download Efficiency/Marginal plot: {e}")
