# Studio 08 - Benign Overfitting in Ridgeless Regression

This is the *studio8* branch, rather than the *main* branch.

## Overview
This project reproduces Figure 2 from Hastie et al. (2022), demonstrating the "double descent" phenomenon in ridgeless least squares regression.

## Installation

Clone using the web URL:

```bash
git clone https://github.com/Celestine97/STATS607-Week7.git
cd STATS607-Week7
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the virtual environment:
```bash
# On Windows (Command Prompt)
venv\Scripts\activate

# On Windows (Git Bash or PowerShell)
source venv/Scripts/activate

# On macOS/Linux
source venv/bin/activate
```

Required packages:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python run_simulations.py  # Generate data
python analyze_results.py  # Create Figure 2
```

## Files
- `Data_Generation.py` - Generate synthetic data with y = Xβ + σε
- `Models.py` - Ridgeless regression implementation
- `run_simulations.py` - Run simulations across 3 regimes (nsim=1,50,1000)
- `analyze_results.py` - Generate Figure 2 visualization

## Key Results
- Reproduces double descent: risk peaks at γ=1 then decreases
- Shows benign overfitting in overparameterized regime (γ>1)


