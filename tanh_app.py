"""
BSD3513 Lab Report 5 - App 3: Hyperbolic Tangent (Tanh) activation function visualiser.

Run locally:  streamlit run tanh_app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def tanh(x):
    """Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x), squashes any input into (-1, 1)."""
    return np.tanh(x)


def tanh_derivative(x):
    """Gradient of tanh: 1 - f(x)^2; maximum of 1 at x = 0."""
    return 1.0 - np.tanh(x) ** 2


def make_plot(x, show_derivative, x_point):
    """Build the matplotlib figure for f(x), optional f'(x) and a marked input point."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, tanh(x), color="#1f77b4", linewidth=2.5, label="Tanh  f(x)")
    if show_derivative:
        ax.plot(x, tanh_derivative(x), color="#d62728", linestyle="--", linewidth=2, label="Derivative  f'(x)")
    ax.scatter([x_point], [tanh(np.array(x_point))], color="black", zorder=5, label=f"x = {x_point:.2f}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Input x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("Hyperbolic Tangent (Tanh)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def main():
    st.set_page_config(page_title="Tanh Visualiser", page_icon="📈")
    st.title("📈 Tanh Activation Function")
    st.caption("BSD3513 Introduction to AI - Lab Report 5 (Neural Networks)")

    st.latex(r"f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}")
    st.write(
        "Tanh is a rescaled sigmoid whose output lies between -1 and 1. Being zero-centred, "
        "it usually trains hidden layers faster than sigmoid."
    )

    st.sidebar.header("Controls")
    x_min, x_max = st.sidebar.slider("Input range", -20.0, 20.0, (-10.0, 10.0), step=1.0)
    show_derivative = st.sidebar.checkbox("Show derivative", value=True)
    x_point = st.sidebar.slider("Test input x", float(x_min), float(x_max), 2.0, step=0.1)

    x = np.linspace(x_min, x_max, 600)
    st.pyplot(make_plot(x, show_derivative, x_point))

    col1, col2 = st.columns(2)
    col1.metric("f(x)", f"{float(tanh(np.array(x_point))):.4f}")
    col2.metric("f'(x)", f"{float(tanh_derivative(np.array(x_point))):.4f}")

    with st.expander("Key properties"):
        st.markdown(
            "- **Output range:** (-1, 1), zero-centred\n"
            "- **Smooth and differentiable** everywhere\n"
            "- **Pros:** zero-centred, stronger gradients than sigmoid (max 1 vs 0.25)\n"
            "- **Cons:** still saturates for large |x| (vanishing gradient)"
        )


if __name__ == "__main__":
    main()
