import streamlit as st
import numpy as np
from scipy import stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math

# --- Helper for truncated normal sampling ---
def truncate_normal(mean, std, min_val, max_val, size):
    if min_val < max_val:
        a, b = (min_val - mean) / std, (max_val - mean) / std
        return truncnorm(a, b, loc=mean, scale=std).rvs(size)
    return np.random.normal(loc=mean, scale=std, size=size)

# --- Sidebar controls ---
st.sidebar.header("Simulation settings")
# Distribution selector
dist_name = st.sidebar.selectbox("Distribution type", ("Normal", "Log-normal"), key="dist_type")

# --- MidRange Parameters ---
st.sidebar.header("MidRange Parameters")
n = st.sidebar.number_input("Sample size (N)", min_value=10, max_value=1_000_000, value=92, step=1, key="mid_n")
middle_min = st.sidebar.number_input("Minimum", value=1190.0, format="%.2f", key="mid_min")
middle_max = st.sidebar.number_input("Maximum", value=14392.0, format="%.2f", key="mid_max")
display_mean = st.sidebar.number_input("Mean", value=3564.0, format="%.2f", key="mid_mean")
median_val = st.sidebar.number_input("Median", value=2690.0, format="%.2f", key="mid_median")
display_std = st.sidebar.number_input("Std dev", min_value=0.01, value=2499.0, format="%.2f", key="mid_std")
skew_val = st.sidebar.number_input("Skew", value=2.19, format="%.2f", key="mid_skew")
kurt_val = st.sidebar.number_input("Kurtosis", value=5.91, format="%.2f", key="mid_kurt")

# Configure sampler
if dist_name == "Normal":
    sampler = lambda size: truncate_normal(display_mean, display_std, middle_min, middle_max, size)
elif dist_name == "Log-normal":
    orig_mean, orig_std = display_mean, display_std
    if orig_mean > 0:
        logsigma = math.sqrt(math.log(1 + (orig_std**2)/(orig_mean**2)))
        logmu = math.log(orig_mean) - 0.5*logsigma**2
    else:
        logmu, logsigma = 0.0, 1.0
    sampler = lambda size: np.random.lognormal(mean=logmu, sigma=logsigma, size=size)

# --- LT 10% Parameters ---
st.sidebar.header("LT 10% Parameters")
n_below = st.sidebar.number_input("Sample size for <10% dist", min_value=0, max_value=1_000_000, value=60, step=1, key="below_n")
below_mean = st.sidebar.number_input("Mean", value=531.0, format="%.2f", key="below_mean")
below_std = st.sidebar.number_input("Std dev", value=284.0, format="%.2f", key="below_std")
below_min = st.sidebar.number_input("Min", value=0.0, format="%.2f", key="below_min")
below_max = st.sidebar.number_input("Max", value=1130.0, format="%.2f", key="below_max")
below_color = st.sidebar.color_picker("Colour", '#1f77b4', key="below_color")

# --- GT 180% Parameters ---
st.sidebar.header("GT 180% Parameters")
n_above = st.sidebar.number_input("Sample size for >180% dist", min_value=0, max_value=1_000_000, value=0, step=1, key="above_n")
above_mean = st.sidebar.number_input("Mean", value=0.0, format="%.2f", key="above_mean")
above_std = st.sidebar.number_input("Std dev", value=1.0, format="%.2f", key="above_std")
above_min = st.sidebar.number_input("Min", value=0.0, format="%.2f", key="above_min")
above_max = st.sidebar.number_input("Max", value=0.0, format="%.2f", key="above_max")
above_color = st.sidebar.color_picker("Colour", '#ff7f0e', key="above_color")

# --- Benchmarks & Bin ---
st.sidebar.header("Benchmarks & Bin")
benchmark = st.sidebar.number_input("Current benchmark", value=0.0, format="%.2f", key="benchmark")
new_benchmark = st.sidebar.number_input("New benchmark", value=0.0, format="%.2f", key="new_benchmark")
bin_width = st.sidebar.number_input("Bin width", min_value=1.0, value=100.0, step=1.0, format="%.2f", key="bin_width")

# --- Generate data & compute stats ---
data = sampler(n)
below_data = truncate_normal(below_mean, below_std, below_min, below_max, n_below)
above_data = truncate_normal(above_mean, above_std, above_min, above_max, n_above)
mean = np.mean(data)
median = np.median(data)
std = np.std(data)
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

# Compute bins
all_vals = np.concatenate([data, below_data, above_data])
bins = np.arange(all_vals.min(), all_vals.max() + bin_width, bin_width)

# --- Combined Plot ---
fig, ax = plt.subplots()
ax.hist(data, bins=bins, alpha=0.3, edgecolor='white', label='Primary', zorder=1)
ax.hist(above_data, bins=bins, alpha=1.0, color=above_color, edgecolor='white', label='>180%', zorder=3)
ax.hist(below_data, bins=bins, alpha=1.0, color=below_color, edgecolor='white', label='<10%', zorder=4)
ax.axvline(benchmark, color='black', linestyle='--', linewidth=1, label=f'Benchmark: {benchmark:.2f}', zorder=5)
ax.axvline(new_benchmark, color='red', linestyle='--', linewidth=1, label=f'New BM: {new_benchmark:.2f}', zorder=5)
if median_val:
    ax.axvline(median_val, color='purple', linestyle=':', linewidth=1, label=f'Median: {median_val:.2f}', zorder=5)
ax.legend()
st.pyplot(fig)

# --- Separate Primary Plot ---
st.header('Primary Distribution Only')
fig1, ax1 = plt.subplots()
ax1.hist(data, bins=bins, color='skyblue', edgecolor='white')
ax1.set_xlabel('Payment Value')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Primary Distribution (n={n})')
st.pyplot(fig1)

# --- Separate LT10% Plot ---
st.header('LT 10% Distribution Only')
fig2, ax2 = plt.subplots()
ax2.hist(below_data, bins=bins, color=below_color, edgecolor='white')
ax2.set_xlabel('Payment Value')
ax2.set_ylabel('Frequency')
ax2.set_title(f'<10% Distribution (n={n_below})')
st.pyplot(fig2)

# --- Separate GT180% Plot ---
st.header('>180% Distribution Only')
fig3, ax3 = plt.subplots()
ax3.hist(above_data, bins=bins, color=above_color, edgecolor='white')
ax3.set_xlabel('Payment Value')
ax3.set_ylabel('Frequency')
ax3.set_title(f'>180% Distribution (n={n_above})')
st.pyplot(fig3)

# --- Display statistics ---
st.markdown(f"""
**Computed Statistics (Primary):**
- Mean: {mean:.2f}
- Median: {median:.2f}
- Std Dev: {std:.2f}
- Skewness: {skewness:.2f}
- Kurtosis: {kurtosis:.2f}

**User-entered (Primary median/skew/kurt):**
- Median: {median_val:.2f}
- Skewness: {skew_val:.2f}
- Kurtosis: {kurt_val:.2f}
""" )
