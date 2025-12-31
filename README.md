# Volatility Surface Modelling & Pricing Engine

## Project Overview

This project implements an industrial-grade quantitative finance pipeline designed to construct, visualize, and utilize **Implied Volatility Surfaces** for the S&P 500 ETF (SPY).

Going beyond simple interpolation, this engine implements a rigorous **SVI (Stochastic Volatility Inspired)** calibration on real-time market data to ensure arbitrage-free smoothing. It derives the **Local Volatility Surface** using Dupire's formula—adjusted for **Log-Forward Moneyness** coordinates—and employs it in a **Monte Carlo Pricing Engine** to price path-dependent exotic derivatives (specifically Barrier Options) and analyze **Hedging Risks** (Delta Skew).

**Core Objective:** To bridge the gap between raw, noisy market options data and a tradeable, arbitrage-free volatility surface suitable for pricing and hedging.

---

## Key Features

### 1. Robust Data ETL (`src/data_loader.py`)

* **Real-time Connection:** Fetches live Option Chain data via Yahoo Finance API.
* **Smart Cleaning:**
* **Liquidity Filtering:** Filters quotes based on **Open Interest** and **Volume** to remove stale prices.
* **Data Hygiene:** Eliminates "dirty data" (e.g., zero-volatility quotes, extreme bid-ask spreads).
* **Precise Time-Keeping:** Calculates Time-to-Maturity () using trading calendars, handling intraday granularity.


* **Yield Curve Bootstrapping:** Dynamically bootstraps the risk-free rate curve () using US Treasury yields (13W, 5Y, 10Y, 30Y) to ensure accurate discounting for long-dated options.

### 2. SVI Calibration Engine (`src/svi_model.py`)

The engine fits the **Raw SVI (Stochastic Volatility Inspired)** parameterization to market implied volatilities for each expiration slice.

* **Why SVI?** Unlike cubic splines, SVI guarantees correct asymptotic behavior in the wings (), preventing the "wiggling" that causes massive pricing errors in tail risk products.
* **The Formula:**



Where  is total variance, and  is log-moneyness.
* **Constrained Optimization:**
* **Optimizer:** Uses `SLSQP` (Sequential Least SQuares Programming) with **Multi-Start** heuristics to avoid local minima in highly skewed slices.
* **Arbitrage Enforcement:** Implements **Penalty Functions** to enforce static no-arbitrage constraints:
* **Vertical Arbitrage:** , , .
* **Butterfly Arbitrage:** Enforces strictly positive density ( and convexity checks).





> **Visual Proof:** The model fits the raw market data (red dots) with high precision, ensuring a smooth, arbitrage-free curve (blue line) even in the presence of noise.

> *Figure 1: Real-time SVI model calibration against SPY market data. Note the tight fit in the liquid region and stable extrapolation in the wings.*

### 3. Surface Construction & Local Volatility (`src/vol_surface.py`)

This module transforms the 2D SVI slices into a coherent 3D Implied Volatility Surface and extracts the Local Volatility Surface .

* **Coordinate System Transformation (Critical Industry Standard):**
* Instead of Spot Moneyness (), the engine internally converts all coordinates to **Log-Forward Moneyness**:


* **Why?** This absorbs the drift terms (), significantly simplifying Dupire's formula and making the surface invariant to interest rate shifts.


* **Dupire’s Local Volatility:**
* Implements **Gatheral’s formulation** of Dupire in terms of Total Variance :


* **Numerical Derivatives:** Uses central finite differences on the dense interpolated grid to compute partial derivatives  and .
* **Regularization:** Includes calendar arbitrage checks () and ratio caps to prevent numerical explosions in deep OTM regions.



> **Topology Comparison:** Notice how the Local Volatility surface (right) exhibits steeper skew and sharper features compared to the smoother Implied Volatility surface (left). This "leverage effect" (volatility increasing as spot drops) is crucial for pricing barriers accurately.

> *Figure 2: 3D Visualization of the Implied Volatility Surface (Left) vs. Dupire Local Volatility Surface (Right).*

### 4. Exotic Pricing Engine (`src/pricer.py`)

* **Monte Carlo Simulation:** Simulates 50,000+ asset price paths using Geometric Brownian Motion with Local Volatility.


* **Sticky Local Volatility:** The pricer performs a vectorized lookup on the pre-computed Local Volatility grid at every time step , ensuring the smile dynamics are respected.
* **Barrier Option Pricing:**
* Prices **Down-and-Out Call** options.
* **Model Comparison:** Calculates the spread between the **Local Vol Price** and **Black-Scholes Price**, quantifying the "Model Risk" hidden in flat-volatility assumptions.



### 5. Risk Management & Hedging Analysis

* **Finite Difference Greeks:** Calculates Delta () via "Bump and Revalue" method using **Common Random Numbers (CRN)** to minimize variance.
* **Delta Skew Profiling:**
* Visualizes how the hedge ratio changes as the spot price approaches the barrier.
* **Insight:** Local Volatility models typically suggest a significantly different hedging strategy near barriers compared to Black-Scholes, often requiring larger hedges due to the correlation between Spot and Volatility.



### 6. Interactive Dashboard (`app.py`)

* **Tech Stack:** Streamlit + Plotly.
* **Capabilities:**
* Real-time 3D surface rotation and inspection.
* Interactive "Pricing Playground" to test different Strikes, Barriers, and Maturities.
* Live view of calibration error (RMSE) and arbitrage constraints.



---

## Project Structure

```text
volatility-surface-modelling/
├── src/
│   ├── __init__.py         # Package initializer
│   ├── data_loader.py      # ETL: Yahoo Finance API, Cleaning, Yield Curve Bootstrap
│   ├── rates.py            # RateProvider: US Treasury Interpolation
│   ├── svi_model.py        # Optimizer: Raw SVI Parametrization & Penalty Functions
│   ├── vol_surface.py      # Mathematics: Log-Forward Coords, SVI Interpolation, Dupire Formula
│   └── pricer.py           # Engine: Monte Carlo Simulation & Greeks Calculation
├── images/                 # Documentation assets
├── app.py                  # Streamlit Frontend Entry Point
├── demo.ipynb              # Jupyter Notebook for Step-by-Step Research Walkthrough
├── requirements.txt        # Python dependencies
└── README.md               # Documentation

```

---

## Quick Start

### Prerequisites

* Python 3.8+
* Pip or Conda

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ericwang0321/volatility-surface-modelling.git
cd volatility-surface-modelling

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



### How to Run

**Option 1: Interactive Dashboard**
To explore the data, surfaces, and pricing engine via a web interface:

```bash
streamlit run app.py

```

**Option 2: Jupyter Notebook Walkthrough**
For a transparent, code-first walkthrough of the entire quantitative pipeline (without launching the web UI), refer to `demo.ipynb`. This notebook is designed for research and code review, allowing you to step through the logic block-by-block.

---

## Technical Deep Dive: Why This Architecture?

### 1. Handling the "Smile" with SVI

Standard polynomial interpolation of implied volatility often leads to **arbitrage** (negative probability density) in the wings. We use SVI because it is **asymptotically linear**, which is consistent with the Heston model and ensures no static arbitrage exists if parameters are constrained correctly. Our implementation explicitly penalizes parameters that violate density constraints.

### 2. Log-Forward Moneyness for Dupire

Implementing Dupire's formula using standard Spot Moneyness () is numerically unstable when interest rates are non-zero, requiring complex drift-adjustment terms. By transforming the surface to **Log-Forward Moneyness** () before calibration:

1. We remove the drift term  from the partial differential equation.
2. The SVI fit becomes more symmetric and robust.
3. The extraction of  becomes purely a function of the shape of the surface, decoupled from the discount curve.

### 3. Hedging the "Crash" Risk

In Equity markets, the Skew is negative (Spot , Vol ). A simple Black-Scholes model assumes Vol is constant.

* **Scenario:** If SPY drops 10%, a BS model assumes Vol stays at 15%. Our Local Vol model knows (via the surface) that Vol will spike to 25%.
* **Result:** The Local Vol model assigns a higher probability to hitting the Down-and-Out Barrier, resulting in a **lower** theoretical price for the option compared to BS. This project empirically proves this behavior.

---

## Roadmap

* [x] Phase 1: Real-time Data ETL & Yield Curve Bootstrapping
* [x] Phase 2: Arbitrage-Free SVI Calibration (Raw SVI + SLSQP)
* [x] Phase 3: Log-Forward Coordinate Transformation
* [x] Phase 4: Local Volatility (Dupire) Extraction
* [x] Phase 5: Monte Carlo Pricing Engine for Exotics (Down-and-Out)
* [x] Phase 6: Greeks Analysis (Delta Skew)
* [ ] Future: Implement SSVI (Surface SVI) for guaranteed calendar arbitrage freedom.
* [ ] Future: Add Heston Model Calibration (Stochastic Volatility).