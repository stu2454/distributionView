import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

# --- Sidebar controls ---
st.sidebar.header("Simulation settings")

# Distribution selector (only Normal and Log-normal)
dist_name = st.sidebar.selectbox(
    "Distribution type",
    ("Normal", "Log-normal")
)

# Sample size
n = st.sidebar.number_input(
    "Sample size (N)",
    min_value=10,
    max_value=1_000_000,
    value=1000,
    step=100
)

# Always get mean and std for Normal
mu = st.sidebar.number_input(
    "Mean (orig-scale)",
    value=0.0,
    format="%.2f"
)
sigma = st.sidebar.number_input(
    "Std dev (orig-scale)",
    min_value=0.01,
    value=1.0,
    format="%.2f"
)

# Configure sampler based on distribution
if dist_name == "Normal":
    sampler = lambda size: np.random.normal(loc=mu, scale=sigma, size=size)
elif dist_name == "Log-normal":
    # Prefill log-normal orig-scale inputs with Normal values
    orig_mean = st.sidebar.number_input(
        "Mean (orig-scale) for log-normal",
        value=mu,
        format="%.2f"
    )
    orig_std = st.sidebar.number_input(
        "Std dev (orig-scale) for log-normal",
        min_value=0.0,
        value=sigma,
        format="%.2f"
    )
    # Compute log-space parameters
    if orig_mean > 0:
        variance = orig_std ** 2
        logsigma = math.sqrt(math.log(1 + variance / orig_mean**2))
        logmu = math.log(orig_mean) - 0.5 * logsigma**2
    else:
        logmu = 0.0
        logsigma = 1.0
    st.sidebar.write(f"Calculated log-mean: {logmu:.2f}, log-std dev: {logsigma:.2f}")
    sampler = lambda size: np.random.lognormal(mean=logmu, sigma=logsigma, size=size)

# Absolute percentile payment values
p25_val = st.sidebar.number_input(
    "25th percentile payment value (P25)",
    value=0.0,
    format="%.2f"
)
p75_val = st.sidebar.number_input(
    "75th percentile payment value (P75)",
    value=0.0,
    format="%.2f"
)
p90_val = st.sidebar.number_input(
    "90th percentile payment value (P90)",
    value=0.0,
    format="%.2f"
)

# User-provided summary stats
median_val = st.sidebar.number_input(
    "Median (orig-scale)",
    value=0.0,
    format="%.2f"
)
skew_val = st.sidebar.number_input(
    "Skewness",
    format="%.2f"
)
kurt_val = st.sidebar.number_input(
    "Kurtosis",
    format="%.2f"
)

# Benchmarks inputs
benchmark = st.sidebar.number_input(
    "Current benchmark",
    value=0.0,
    format="%.2f"
)
new_benchmark = st.sidebar.number_input(
    "New benchmark",
    value=benchmark,
    format="%.2f"
)

# --- Generate data & compute stats ---
data = sampler(n)
mean = np.mean(data)
median = np.median(data)
std = np.std(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# --- Plotting ---
fig, ax = plt.subplots()
ax.hist(data, bins=50, alpha=0.7, edgecolor='white')
# Plot benchmarks and percentiles
ax.axvline(benchmark, color='black', linestyle='--', linewidth=2,
           label=f'Benchmark: {benchmark:.2f}')
ax.axvline(new_benchmark, color='red', linestyle='--', linewidth=2,
           label=f'New BM: {new_benchmark:.2f}')
if p25_val:
    ax.axvline(p25_val, color='blue', linestyle='-.', linewidth=2,
               label=f'P25: {p25_val:.2f}')
if p75_val:
    ax.axvline(p75_val, color='green', linestyle='-.', linewidth=2,
               label=f'P75: {p75_val:.2f}')
if p90_val:
    ax.axvline(p90_val, color='orange', linestyle='-.', linewidth=2,
               label=f'P90: {p90_val:.2f}')
# User-provided median line
if median_val:
    ax.axvline(median_val, color='purple', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.2f}')
ax.legend()
st.pyplot(fig)

# --- Display computed and user summary statistics ---
st.markdown(f"""
**Computed Statistics:**  
- Mean: {mean:.2f}  
- Median: {median:.2f}  
- Std Dev: {std:.2f}  
- Skewness: {skewness:.2f}  
- Kurtosis: {kurtosis:.2f}  

**User-entered:**  
- Median: {median_val:.2f}  
- Skewness: {skew_val:.2f}  
- Kurtosis: {kurt_val:.2f}  
""")
