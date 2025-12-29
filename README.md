## 📖 Project Overview

This project is an industrial-grade quantitative finance pipeline designed to construct, visualize, and utilize **Implied Volatility Surfaces** for the S&P 500 ETF (SPY).

Unlike simple interpolation scripts, this engine implements a rigorous **SVI (Stochastic Volatility Inspired)** calibration on real-time market data, ensuring arbitrage-free smoothing of volatility smiles. The system is decoupled into a robust backend (ETL & Calculation Engine) and an interactive frontend (Streamlit Dashboard).

**Core Objective:** To bridge the gap between raw, noisy market options data and a tradeable volatility surface suitable for exotic derivative pricing.

---

## 🚀 Key Features

### 1. Robust Data ETL (`src/data_loader.py`)
* **Real-time Connection:** Fetches live Option Chain data via Yahoo Finance API.
* **Smart Cleaning:** * Filters for liquidity (Volume/OI checks).
    * Handles "dirty data" and zero-bid contracts.
    * Precisely calculates Time-to-Maturity ($T$) using trading calendars.
* **Rate Bootstrapping:** Dynamically fetches the risk-free rate ($r$) using the 13-Week Treasury Bill (`^IRX`) as a proxy.

### 2. SVI Calibration Engine (`src/svi_model.py`)
* Implements the **Raw SVI Parameterization** model to fit volatility smiles for each expiration slice.
* **Optimization:** Uses `scipy.optimize` with bounded constraints to ensure model stability ($b > 0$, $|\rho| < 1$, $\sigma > 0$).
* **Formula:**
    $$w(k) = a + b \left\{ \rho(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right\}$$
    *Where $w$ is total variance and $k$ is log-moneyness.*

### 3. Volatility Surface Construction (`src/vol_surface.py`)
* **Time Interpolation:** Constructs a dense grid by linearly interpolating Total Variance ($w$) in the time dimension.
* **Arbitrage Checks:** Implicitly handles calendar arbitrage through variance interpolation.
* **3D Mesh Generation:** Exports vectorized coordinates for visualization.

### 4. Interactive Dashboard (`app.py`)
* **3D Visualization:** Fully interactive Plotly 3D surface to inspect Skew and Term Structure.
* **Smile Inspection:** Drill-down capability to view raw market data vs. fitted SVI curves for specific maturities.

---

## 🛠️ Project Structure

```text
volatility-surface-modelling/
├── src/                    # Core Quantitative Library
│   ├── __init__.py         # Package initializer
│   ├── data_loader.py      # ETL pipeline for options data
│   ├── rates.py            # Risk-free rate service
│   ├── svi_model.py        # SVI calibration logic (Optimizer)
│   ├── vol_surface.py      # Surface construction & interpolation
│   └── pricer.py           # [Planned] Monte Carlo Pricing Engine
├── notebooks/              # Research & Prototyping (Jupyter)
├── data/                   # Local data cache
├── app.py                  # Streamlit Frontend Entry Point
├── requirements.txt        # Python dependencies
└── README.md               # Documentation

```

---

## ⚡ Quick Start

### Prerequisites

* Python 3.8+
* Pip or Conda

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd volatility-surface-modelling

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the Dashboard:**
```bash
streamlit run app.py

```



---

## 📊 Methodology Highlight: Why SVI?

We chose the **SVI (Stochastic Volatility Inspired)** parameterization over cubic splines or polynomial regression for several reasons critical in a sell-side context:

1. **Asymptotic Behavior:** SVI ensures that variance is linear in the wings (), preventing the "Runge Phenomenon" (wild oscillations) common in polynomial fits.
2. **Arbitrage Constraints:** SVI parameters can be constrained to ensure the density function remains positive (no static arbitrage).
3. **Interpretability:** Parameters like  directly map to the market skew (correlation between spot and vol), offering traders intuitive insights.

---

## 🔜 Roadmap

* **Phase 1-3:** Data ETL, SVI Calibration, 3D Surface (✅ Completed)
* **Phase 4:** **Local Volatility (Dupire)** extraction from the Implied Vol Surface.
* **Phase 5:** **Exotic Pricing Engine**. Implementing a Monte Carlo pricer for Barrier Options using the generated Local Volatility surface.
