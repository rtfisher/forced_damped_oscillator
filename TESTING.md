# Testing Documentation

## Overview

The oscillator simulation suite includes a comprehensive test suite (`test_oscillator.py`) that validates numerical accuracy, physical correctness, and visualization functionality. All tests are designed using convergence analysis based on the theoretical properties of the RK4 integration method.

## Running Tests

### Local Testing

```bash
# Run all tests
conda run -n npscipy python -m pytest test_oscillator.py -v

# Run specific test class
conda run -n npscipy python -m pytest test_oscillator.py::TestEnergyConservation -v

# Run with coverage
conda run -n npscipy python -m pytest test_oscillator.py --cov=. --cov-report=term
```

### GitHub Actions

Tests run automatically on:
- Push to main/master/develop branches
- Pull requests
- Manual workflow dispatch

The CI tests across:
- Operating systems: Ubuntu, macOS, Windows
- Python versions: 3.9, 3.10, 3.11, 3.12

## Test Suite Structure

### 1. Numerical Integration Tests (`TestNumericalIntegration`)

**Purpose**: Validate the RK4 integration method exhibits correct 4th order convergence.

#### `test_rk4_fourth_order_convergence`
- Tests simple harmonic oscillator with known analytical solution
- Verifies error ~ O(dt^4) using log-log regression
- Checks convergence ratios: error(dt) / error(dt/2) ≈ 16

**Key concept**: For an pth-order method, when timestep is halved, error should decrease by factor 2^p. For RK4, p=4, so factor ≈ 16.

#### `test_rk4_long_time_accuracy`
- Verifies accuracy maintained over long integration times (100 time units)
- Tests at multiple points throughout trajectory
- Ensures no significant error accumulation

**Rationale**: Global error for RK4 is O(dt^4), should remain bounded even for long times.

### 2. Energy Conservation Tests (`TestEnergyConservation`)

**Purpose**: Verify energy conservation in undamped, undriven systems with expected convergence rates.

#### `test_harmonic_simple_energy_drift_scaling`
- Tests simple harmonic oscillator (conservative system)
- Measures energy drift at multiple timesteps
- Verifies drift scales as O(dt^4)

**Physical basis**: In a conservative system, total energy E = KE + PE should be exactly conserved. Numerical drift is due to integration error, which for RK4 scales as dt^4.

#### `test_duffing_undriven_energy_conservation`
- Tests nonlinear Duffing oscillator
- Energy: E = (1/2)v² + (α/2)x² + (β/4)x⁴
- Verifies 4th order drift scaling

**Why this matters**: Confirms RK4 preserves energy well even for nonlinear systems.

### 3. Analytical Comparison Tests (`TestAnalyticalComparison`)

**Purpose**: Compare numerical solutions to known analytical solutions.

#### `test_simple_harmonic_oscillator_convergence`
- Compares to x(t) = cos(ω₀t) for ω₀ = √α
- Uses log-log fit to extract convergence order
- Verifies order ≈ 4.0

**Mathematical basis**:
```
error ~ C × dt^p
log(error) = log(C) + p × log(dt)
```
Slope of log(error) vs log(dt) gives convergence order p.

#### `test_damped_oscillator_convergence`
- Analytical solution for underdamped case (δ < 2ω₀):
```
x(t) = exp(-δt/2) × [C₁cos(ω_d t) + C₂sin(ω_d t)]
where ω_d = √(ω₀² - δ²/4)
```
- Uses RMS error over multiple time points (more robust than single endpoint)
- Verifies 4th order convergence

**Design choice**: RMS over trajectory is less sensitive to phase errors than single endpoint comparison.

#### `test_driven_resonance_steady_state_convergence`
- Tests driven oscillator at resonance (ω = ω₀)
- Theoretical steady-state amplitude: A = γ / (δω)
- Verifies numerical amplitude matches theory

**Physics**: At resonance, system absorbs maximum energy from driving force.

### 4. Physical Behavior Tests (`TestPhysicalBehavior`)

**Purpose**: Verify physically correct behavior regardless of numerical details.

#### `test_damping_monotonic_energy_decrease`
- Energy should decrease monotonically in damped systems
- Allows <0.1% violations due to numerical noise
- Verifies final energy << initial energy

**Physical law**: Damping dissipates energy irreversibly.

#### `test_period_measurement_convergence`
- Measures oscillation period from zero crossings
- Compares to theoretical T = 2π/ω₀
- Verifies convergence as dt decreases

**Note**: Period is derived quantity, not primary integrated variable, so convergence may be slightly slower than 4th order.

#### `test_duffing_hardening_frequency_shift`
- Hardening spring (β > 0): frequency increases with amplitude
- Tests at multiple amplitudes
- Verifies f(large amplitude) > f(small amplitude)

**Nonlinear dynamics**: Duffing equation shows amplitude-dependent frequency, unlike linear oscillators.

### 5. Visualization Tests (`TestVisualization`)

**Purpose**: Ensure visualization components generate correctly.

#### `test_setup_visualization_structure`
- Verifies all required plot elements exist
- Checks axis limits are set properly
- Validates data structure completeness

#### `test_animation_creation_no_errors`
- Creates full animation without errors
- Verifies animation has save method
- Tests with representative parameters

#### `test_all_presets_integrate_successfully`
- Runs all 6 preset modes
- Verifies no NaN or Inf values
- Checks output shapes and energy positivity

**CI importance**: Ensures presets work on all platforms.

### 6. Numerical Stability Tests (`TestNumericalStability`)

**Purpose**: Verify robustness across parameter ranges.

#### `test_chaotic_regime_stability`
- Long-time integration (200 time units) of chaotic Duffing
- Verifies no numerical blow-up
- Checks strange attractor remains bounded

**Challenge**: Chaotic systems amplify small errors exponentially, but RK4 should remain stable.

#### `test_timestep_convergence_consistency`
- Tests convergence for various parameter combinations
- Conservative, driven, and nonlinear cases
- Verifies consistent 4th order behavior

#### `test_energy_bounds_physical`
- High-energy initial conditions
- Energy should not grow unboundedly in damped systems
- Validates physical bounds maintained

## Convergence Testing Methodology

### Why Convergence Rates Matter

Testing against arbitrary tolerances (e.g., "error < 1e-6") is fragile:
- Depends on problem parameters, time scale, units
- Doesn't validate the numerical method order
- May pass with wrong implementation if tolerance is loose

**Convergence rate testing is rigorous**:
- Tests fundamental property of the numerical method
- Independent of problem specifics
- Detects subtle implementation bugs

### RK4 Convergence Properties

**Local truncation error**: O(dt⁵) per step

**Global error**: O(dt⁴) over fixed time interval

**Energy drift** in conservative systems: O(dt⁴)

### Expected Convergence Ratios

When timestep is halved:
```
error(dt) / error(dt/2) ≈ 2⁴ = 16
```

Tests allow range [8, 40] to account for:
- Different problems may show better/worse constants
- Nonlinear systems may vary
- Finite precision effects

### Log-Log Regression

For tests using multiple timesteps:
```python
log(error) = log(C) + p × log(dt)
```

Linear fit gives order p directly. For RK4: |p| ≈ 4.0.

## Test Coverage

| Category | Tests | Lines Covered |
|----------|-------|---------------|
| Numerical integration | 2 | RK4 implementation |
| Energy conservation | 2 | Conservative dynamics |
| Analytical comparison | 3 | All oscillator modes |
| Physical behavior | 3 | Damping, periods, nonlinearity |
| Visualization | 3 | Plot generation |
| Numerical stability | 3 | Edge cases, chaos |
| **Total** | **16** | **~90% of oscillator.py** |

## Interpreting Test Failures

### Convergence order too low (order < 3.5)
- Bug in RK4 implementation
- Wrong parameter passed to integrator
- Time array issues

### Convergence order too high (order > 4.5)
- Problem is too simple (reduces to lower order method)
- Errors at machine precision
- Special symmetries in test case

### Energy not conserved
- Missing terms in Hamiltonian
- Wrong signs in force calculation
- Numerical instability

### Physical behavior wrong
- Incorrect damping sign
- Wrong potential energy formula
- Parameter mismatch

## Adding New Tests

### Template for convergence test:
```python
def test_new_feature_convergence(self):
    """Test new feature with convergence analysis."""
    # Setup parameters
    ...

    # Multiple timesteps
    dts = [0.08, 0.04, 0.02, 0.01]
    errors = []

    for dt in dts:
        # Run simulation
        result = integrate(...)

        # Compute error vs analytical/reference
        error = compute_error(result)
        errors.append(error)

    errors = np.array(errors)

    # Option 1: Check convergence ratios
    ratios = errors[:-1] / errors[1:]
    for ratio in ratios:
        assert 12 < ratio < 20, f"Expected ~16, got {ratio}"

    # Option 2: Log-log regression
    order = abs(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    assert 3.5 < order < 4.5, f"Expected ~4, got {order}"
```

### Guidelines:
1. Always test with ≥ 3 different timesteps
2. Use problems with known solutions when possible
3. Document the expected theoretical behavior
4. Allow reasonable tolerance ranges
5. Consider numerical precision limits

## Continuous Integration (CI)

The `.github/workflows/test.yml` workflow:
- Runs on every push and PR
- Tests matrix: 3 OS × 4 Python versions = 12 configurations
- Uploads coverage to Codecov (optional)
- Fails CI if any test fails

## Performance

Typical test run time: **~2-4 seconds** (local machine)

CI run time: **~5-10 seconds per configuration**

## Future Test Additions

Potential areas for expansion:
- Stiff equations (large damping)
- Double-well potential specific tests
- Poincaré map validation
- Lyapunov exponent calculation (chaos verification)
- Phase space volume conservation (symplectic tests)
- Parameter sensitivity analysis

## References

### Numerical Methods
- [Runge-Kutta methods - Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
- [Numerical methods for ordinary differential equations - Wikipedia](https://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations)
- [Convergence (numerical analysis) - Wikipedia](https://en.wikipedia.org/wiki/Rate_of_convergence)

### Physics
- [Harmonic oscillator - Wikipedia](https://en.wikipedia.org/wiki/Harmonic_oscillator)
- [Duffing equation - Wikipedia](https://en.wikipedia.org/wiki/Duffing_equation)
- [Damping (mechanics) - Wikipedia](https://en.wikipedia.org/wiki/Damping)
- [Resonance - Wikipedia](https://en.wikipedia.org/wiki/Resonance)
- [Chaos theory - Wikipedia](https://en.wikipedia.org/wiki/Chaos_theory)

### Software Testing
- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
