import streamlit as st
import numpy as np
from scipy import stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math

# --- Helper for truncated normal sampling ---
def truncate_normal(mean, std, min_val, max_val, size):
    """
    Draw samples from a truncated normal distribution between min_val and max_val.
    """
    # If bounds are valid, compute standardized a, b
    if min_val < max_val:
        a, b = (min_val - mean) / std, (max_val - mean) / std
        return truncnorm(a, b, loc=mean, scale=std).rvs(size)
    # Fallback to standard normal if invalid bounds
    return np.random.normal(loc=mean, scale=std, size=size)

# --- Sidebar controls ---
st.sidebar.header("Simulation settings")

# Distribution selector (only Normal and Log-normal)
dist_name = st.sidebar.selectbox(
    "Distribution type",
    ("Normal", "Log-normal")
)

# Primary sample size
n = st.sidebar.number_input(
    "Sample size for primary distribution (N)",
    min_value=10,
    max_value=1_000_000,
    value=1000,
    step=100
)

# Primary distribution bounds
middle_min = st.sidebar.number_input(
    "Middle distribution min",
    value=0.0,
    format="%.2f"
)
middle_max = st.sidebar.number_input(
    "Middle distribution max",
    value=0.0,
    format="%.2f"
)

# Always get mean and std for Normal
display_mean = st.sidebar.number_input(
    "Mean (orig-scale)",
    value=0.0,
    format="%.2f"
)
display_std = st.sidebar.number_input(
    "Std dev (orig-scale)",
    min_value=0.01,
    value=1.0,
    format="%.2f"
)

# Configure sampler based on distribution
if dist_name == "Normal":
    # Use truncated normal to avoid spikes
    sampler = lambda size: truncate_normal(display_mean, display_std, middle_min, middle_max, size)
elif dist_name == "Log-normal":
    # Prefill log-normal orig-scale inputs with Normal values
    orig_mean = st.sidebar.number_input(
        "Mean (orig-scale) for log-normal",
        value=display_mean,
        format="%.2f"
    )
    orig_std = st.sidebar.number_input(
        "Std dev (orig-scale) for log-normal",
        min_value=0.0,
        value=display_std,
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
    # Rejection sampling for log-normal within bounds
    def log_sampler(size):
        samples = []
        while len(samples) < size:
            batch = np.random.lognormal(mean=logmu, sigma=logsigma, size=size)
            if middle_min < middle_max:
                batch = batch[(batch >= middle_min) & (batch <= middle_max)]
            samples.extend(batch.tolist())
        return np.array(samples[:size])
    sampler = log_sampler

# Additional distributions: <10% and >180%
st.sidebar.header("Additional distributions settings for extremes")
# Sample sizes for extremes
n_below = st.sidebar.number_input(
    "Sample size for <10% distribution (N_below)",
    min_value=0,
    max_value=1_000_000,
    value=n,
    step=100
)
n_above = st.sidebar.number_input(
    "Sample size for >180% distribution (N_above)",
    min_value=0,
    max_value=1_000_000,
    value=n,
    step=100
)
# Means, std devs, and bounds for extremes
below_mean = st.sidebar.number_input("<10% dist mean", value=0.0, format="%.2f")
below_std = st.sidebar.number_input("<10% dist std dev", min_value=0.0, value=1.0, format="%.2f")
below_min = st.sidebar.number_input("<10% dist min", value=0.0, format="%.2f")
below_max = st.sidebar.number_input("<10% dist max", value=0.0, format="%.2f")
above_mean = st.sidebar.number_input(
    ">180% dist mean", value=0.0, format="%.2f"
)
above_std = st.sidebar.number_input(
    ">180% dist std dev", min_value=0.0, value=1.0, format="%.2f"
)
above_min = st.sidebar.number_input(
    ">180% dist min", value=0.0, format="%.2f"
)
above_max = st.sidebar.number_input(
    ">180% dist max", value=0.0, format="%.2f"
)
# Colour pickers for extremes
below_color = st.sidebar.color_picker("Colour for <10% distribution", '#047703')
above_color = st.sidebar.color_picker("Colour for >180% distribution", '#AF1034')

# User-entered median/skew/kurt
median_val = st.sidebar.number_input("Median (orig-scale)", value=0.0, format="%.2f")
skew_val = st.sidebar.number_input("Skewness", value=0.0, format="%.2f")
kurt_val = st.sidebar.number_input("Kurtosis", value=0.0, format="%.2f")

# Benchmarks inputs
benchmark = st.sidebar.number_input("Current benchmark", value=0.0, format="%.2f")
new_benchmark = st.sidebar.number_input("New benchmark", value=benchmark, format="%.2f")

# --- Generate data & compute stats ---
# Primary data
data = sampler(n)
# Compute stats
mean = np.mean(data)
median = np.median(data)
std = np.std(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)
# Extremes data using truncation to avoid spikes
below_data = truncate_normal(below_mean, below_std, below_min, below_max, n_below)
above_data = truncate_normal(above_mean, above_std, above_min, above_max, n_above)

# --- Plotting ---
fig, ax = plt.subplots()
ax.hist(data, bins=50, alpha=1.0, edgecolor='white', label='Primary')
ax.hist(below_data, bins=50, alpha=0.7, color=below_color, edgecolor='white', label='<10%')
ax.hist(above_data, bins=50, alpha=0.7, color=above_color, edgecolor='white', label='>180%')
# Benchmark lines
ax.axvline(benchmark, color='black', linestyle='--', linewidth=1, label=f'Benchmark: {benchmark:.2f}')
ax.axvline(new_benchmark, color='red', linestyle='--', linewidth=1, label=f'New BM: {new_benchmark:.2f}')
# User median
if median_val:
    ax.axvline(median_val, color='purple', linestyle=':', linewidth=1, label=f'Median: {median_val:.2f}')
ax.legend()
st.pyplot(fig)

# --- Display statistics ---
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
""" )
