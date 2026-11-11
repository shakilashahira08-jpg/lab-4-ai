"""
BSD3513 Lab Report 5 - App 2: Sigmoid activation function visualiser.

Run locally:  streamlit run sigmoid_app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def sigmoid(x):
    """Sigmoid: f(x) = 1 / (1 + e^-x), squashes any input into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Gradient of sigmoid: f(x) * (1 - f(x)); maximum of 0.25 at x = 0."""
    s = sigmoid(x)
    return s * (1.0 - s)


def make_plot(x, show_derivative, x_point):
    """Build the matplotlib figure for f(x), optional f'(x) and a marked input point."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, sigmoid(x), color="#1f77b4", linewidth=2.5, label="Sigmoid  f(x)")
    if show_derivative:
        ax.plot(x, sigmoid_derivative(x), color="#d62728", linestyle="--", linewidth=2, label="Derivative  f'(x)")
    ax.scatter([x_point], [sigmoid(np.array(x_point))], color="black", zorder=5, label=f"x = {x_point:.2f}")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Input x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.set_title("Sigmoid")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def main():
    st.set_page_config(page_title="Sigmoid Visualiser", page_icon="📈")
    st.title("📈 Sigmoid Activation Function")
    st.caption("BSD3513 Introduction to AI - Lab Report 5 (Neural Networks)")

    st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
    st.write(
        "Sigmoid maps any real number to a value between 0 and 1, which makes it a natural choice "
        "for probabilities in binary classification output layers."
    )

    st.sidebar.header("Controls")
    x_min, x_max = st.sidebar.slider("Input range", -20.0, 20.0, (-10.0, 10.0), step=1.0)
    show_derivative = st.sidebar.checkbox("Show derivative", value=True)
    x_point = st.sidebar.slider("Test input x", float(x_min), float(x_max), 2.0, step=0.1)

    x = np.linspace(x_min, x_max, 600)
    st.pyplot(make_plot(x, show_derivative, x_point))

    col1, col2 = st.columns(2)
    col1.metric("f(x)", f"{float(sigmoid(np.array(x_point))):.4f}")
    col2.metric("f'(x)", f"{float(sigmoid_derivative(np.array(x_point))):.4f}")

    with st.expander("Key properties"):
        st.markdown(
            "- **Output range:** (0, 1), centred at 0.5\n"
            "- **Smooth and differentiable** everywhere\n"
            "- **Pros:** probability-like output, clean gradient\n"
            "- **Cons:** saturates for large |x| (vanishing gradient), outputs are not zero-centred"
        )


if __name__ == "__main__":
    main()
