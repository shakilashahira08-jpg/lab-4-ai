"""
BSD3513 Lab Report 5 - App 1: ReLU activation function visualiser.

Run locally:  streamlit run relu_app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def relu(x):
    """ReLU: f(x) = max(0, x)."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Gradient of ReLU: 1 for x > 0, otherwise 0 (value at x = 0 set to 0 by convention)."""
    return (x > 0).astype(float)


def make_plot(x, show_derivative, x_point):
    """Build the matplotlib figure for f(x), optional f'(x) and a marked input point."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, relu(x), color="#1f77b4", linewidth=2.5, label="ReLU  f(x)")
    if show_derivative:
        ax.plot(x, relu_derivative(x), color="#d62728", linestyle="--", linewidth=2, label="Derivative  f'(x)")
    ax.scatter([x_point], [relu(np.array(x_point))], color="black", zorder=5, label=f"x = {x_point:.2f}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Input x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("Rectified Linear Unit (ReLU)")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def main():
    st.set_page_config(page_title="ReLU Visualiser", page_icon="📈")
    st.title("📈 ReLU Activation Function")
    st.caption("BSD3513 Introduction to AI - Lab Report 5 (Neural Networks)")

    st.latex(r"f(x) = \max(0, x)")
    st.write(
        "ReLU outputs the input directly when it is positive and outputs zero otherwise. "
        "It is cheap to compute and helps avoid the vanishing-gradient problem in deep networks."
    )

    st.sidebar.header("Controls")
    x_min, x_max = st.sidebar.slider("Input range", -20.0, 20.0, (-10.0, 10.0), step=1.0)
    show_derivative = st.sidebar.checkbox("Show derivative", value=True)
    x_point = st.sidebar.slider("Test input x", float(x_min), float(x_max), 2.0, step=0.1)

    x = np.linspace(x_min, x_max, 600)
    st.pyplot(make_plot(x, show_derivative, x_point))

    col1, col2 = st.columns(2)
    col1.metric("f(x)", f"{float(relu(np.array(x_point))):.4f}")
    col2.metric("f'(x)", f"{float(relu_derivative(np.array(x_point))):.0f}")

    with st.expander("Key properties"):
        st.markdown(
            "- **Output range:** [0, infinity)\n"
            "- **Non-linear** (piecewise linear), so stacked layers can learn complex patterns\n"
            "- **Pros:** very fast, sparse activation, no saturation for x > 0\n"
            "- **Cons:** 'dying ReLU' - neurons stuck at x < 0 get zero gradient and stop learning"
        )


if __name__ == "__main__":
    main()
