# Oscillator Simulation Suite

[![Test Suite](https://github.com/rtfisher/forced_damped_oscillator/actions/workflows/test.yml/badge.svg)](https://github.com/rtfisher/forced_damped_oscillator/actions/workflows/test.yml)

A unified Python toolkit for simulating and visualizing driven, damped oscillator systems including harmonic oscillators and nonlinear Duffing oscillators with chaotic dynamics.

## Overview

This package provides an interactive animation of oscillator systems with three synchronized visualization panels:
- **Top panel**: Potential energy landscape with particle position
- **Middle panel**: Physical mass-spring system animation
- **Bottom panel**: Phase portrait (velocity vs. position) showing trajectory history

## Differential Equation

The simulation solves the general driven, damped oscillator equation:

```
x'' + δx' + αx + βx³ = γcos(ωt)
```

**Parameters:**
- `δ` (delta) — damping coefficient
- `α` (alpha) — linear restoring force coefficient
- `β` (beta) — cubic (nonlinear) restoring force coefficient
- `γ` (gamma) — driving force amplitude
- `ω` (omega) — driving angular frequency (rad/s)

Different parameter combinations produce different behaviors:
- **Harmonic oscillator**: β = 0
- **Duffing oscillator**: β ≠ 0 (nonlinear)
- **Chaotic dynamics**: specific parameter regimes with driving and damping

## Quick Start

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- Conda environment `npscipy` (or another environment with the above packages)

### Basic Usage

```bash
# Simple harmonic oscillator (no damping, no driving)
conda run -n npscipy python oscillator.py --mode harmonic-simple

# Driven damped harmonic oscillator
conda run -n npscipy python oscillator.py --mode harmonic-driven

# Chaotic Duffing oscillator
conda run -n npscipy python oscillator.py --mode duffing-chaotic
```

### View All Options

```bash
conda run -n npscipy python oscillator.py --help
```

## Preset Modes

The unified `oscillator.py` script includes six preset configurations:

| Mode | Description | Parameters |
|------|-------------|------------|
| `harmonic-simple` | Simple harmonic oscillator (undamped, undriven) | δ=0, α=1, β=0, γ=0 |
| `harmonic-damped` | Damped harmonic oscillator | δ=0.1, α=1, β=0, γ=0 |
| `harmonic-driven` | Driven damped harmonic oscillator | δ=0.2, α=1, β=0, γ=1, ω=1 |
| `duffing-damped` | Damped Duffing oscillator (nonlinear) | δ=0.1, α=1, β=5, γ=0 |
| `duffing-driven` | Driven damped Duffing oscillator | δ=0.02, α=1, β=5, γ=8, ω=0.5 |
| `duffing-chaotic` | Duffing oscillator in chaotic regime | δ=0.02, α=1, β=5, γ=8, ω=0.5 |

## Advanced Usage

### Override Preset Parameters

```bash
# Use harmonic-driven preset but with different driving amplitude
conda run -n npscipy python oscillator.py --mode harmonic-driven --gamma 2.0

# Longer simulation time
conda run -n npscipy python oscillator.py --mode duffing-chaotic --time 200
```

### Custom Parameters

```bash
# Fully custom configuration
conda run -n npscipy python oscillator.py --custom \
    --delta 0.2 \
    --alpha 1.0 \
    --beta 5.0 \
    --gamma 1.5 \
    --omega 1.0 \
    --time 100
```

### Initial Conditions

```bash
# Start with different initial position and velocity
conda run -n npscipy python oscillator.py --mode harmonic-driven \
    --x0 0.5 \
    --v0 1.0
```

### Save Animation

```bash
# Save as MP4 (requires ffmpeg)
conda run -n npscipy python oscillator.py --mode duffing-chaotic \
    --save-mp4 --output my_chaotic_system

# Save as GIF (requires pillow)
conda run -n npscipy python oscillator.py --mode harmonic-simple \
    --save-gif --output harmonic_motion
```

## Visualization Features

### Three-Panel Display

1. **Potential Energy Panel (Top)**
   - Shows V(x) = (α/2)x² + (β/4)x⁴
   - Marks equilibrium points (stable in green, unstable in red)
   - Red dot indicates current particle position
   - Blue arrow shows velocity direction

2. **Physical System Panel (Middle)**
   - Animated mass-spring oscillator
   - Reference wall at x = 0
   - Green arrow shows driving force γcos(ωt) (when present)
   - Time display in upper left

3. **Phase Portrait Panel (Bottom)**
   - **Blue line**: Complete trajectory history from start to current time
   - **Red line**: Recent trajectory (last 1 time unit)
   - **Red dot**: Current state (x, v)
   - Equilibrium points marked
   - Shows strange attractors in chaotic regimes

### Progressive History Display

The phase portrait displays the trajectory progressively:
- Starts empty at t = 0
- Blue line grows as the simulation progresses, showing accumulated history
- Red line highlights recent motion
- This allows you to watch the phase space structure emerge in real-time

## Command-Line Options Reference

### Simulation Parameters
- `--delta DELTA` — Damping coefficient
- `--alpha ALPHA` — Linear restoring force coefficient
- `--beta BETA` — Cubic restoring force coefficient
- `--gamma GAMMA` — Driving force amplitude
- `--omega OMEGA` — Driving force frequency

### Initial Conditions
- `--x0 X0` — Initial position (default: 0.2)
- `--v0 V0` — Initial velocity (default: 0.0)

### Simulation Settings
- `--time TIME` — Simulation duration in time units (default: 60.0)
- `--dt DT` — Integration time step (default: 0.02)
- `--stride STRIDE` — Animation frame stride (default: 2)

### Output Options
- `--save-gif` — Save animation as GIF
- `--save-mp4` — Save animation as MP4 (requires ffmpeg)
- `--output OUTPUT` — Output filename without extension

## Technical Details

### Integration Method
- 4th-order Runge-Kutta (RK4) method for numerical integration
- Default time step: dt = 0.02
- Adaptive x-axis scaling based on actual motion range

### Performance
- Simulation is precomputed before animation
- Frame stride controls animation speed (default: 2 = every 2nd frame)
- For longer simulations, increase `--stride` for better performance

### Installing ffmpeg (for MP4 export)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Legacy Scripts

The `_old/` directory contains previous standalone scripts for specific configurations:
- `harmonic_undamped_unforced.py`
- `harmonic_forced_damped.py`
- `harmonic_forced_damped2.py`
- `duffing_damped.py`
- `duffing_chaotic.py`
- `phase_oscillator.py`

These are preserved for reference but are superseded by the unified `oscillator.py` script.

## Examples

### Explore Parameter Space

```bash
# Compare different damping values
conda run -n npscipy python oscillator.py --mode harmonic-driven --delta 0.05
conda run -n npscipy python oscillator.py --mode harmonic-driven --delta 0.5

# Investigate resonance
conda run -n npscipy python oscillator.py --mode harmonic-driven --omega 0.9
conda run -n npscipy python oscillator.py --mode harmonic-driven --omega 1.0
conda run -n npscipy python oscillator.py --mode harmonic-driven --omega 1.1
```

### Study Nonlinear Dynamics

```bash
# Vary cubic nonlinearity strength
conda run -n npscipy python oscillator.py --mode duffing-driven --beta 1.0
conda run -n npscipy python oscillator.py --mode duffing-driven --beta 5.0
conda run -n npscipy python oscillator.py --mode duffing-driven --beta 10.0

# Explore route to chaos
conda run -n npscipy python oscillator.py --mode duffing-driven --gamma 2.0 --time 200
conda run -n npscipy python oscillator.py --mode duffing-driven --gamma 5.0 --time 200
conda run -n npscipy python oscillator.py --mode duffing-driven --gamma 8.0 --time 200
```

## Notes

- Close the animation window to exit
- The simulation completes before animation starts
- Phase portrait history builds progressively during animation
- For high-resolution phase portraits, use longer simulation times (`--time`)
- Chaotic systems may require `--time 100` or more to show full attractor structure

## Educational Use

This toolkit is designed for teaching classical mechanics, nonlinear dynamics, and chaos theory. Students can:
- Visualize energy conservation in undamped systems
- Observe damping effects and energy dissipation
- Study resonance phenomena in driven systems
- Explore nonlinear oscillations and multiple equilibria
- Discover strange attractors and chaotic dynamics
- Investigate bifurcations and routes to chaos

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 PHY313 Course Materials

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
