#!/usr/bin/env python3
"""
Comprehensive test suite for oscillator.py

Tests cover:
- Numerical integration accuracy with convergence rate verification
- Energy conservation scaling with timestep
- Comparison with analytical solutions
- Physical correctness of different modes
- Visualization component creation

Uses theoretical convergence rates: RK4 is O(dt^4)

Designed for pytest and GitHub Actions CI
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import pytest
import sys
import os

# Import the oscillator module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oscillator import (
    oscillator_rhs, rk4_step, integrate,
    create_animation, setup_visualization,
    PRESETS
)


class TestNumericalIntegration:
    """Test the RK4 integration method with convergence analysis."""

    def test_rk4_fourth_order_convergence(self):
        """Verify RK4 exhibits 4th order convergence: error ~ O(dt^4)."""
        # Test with simple harmonic oscillator: x'' = -x
        # Analytical: x(t) = cos(t), v(t) = -sin(t)

        def sho_rhs(state, t):
            return np.array([state[1], -state[0]])

        y0 = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
        t_final = 2.0  # Integration time

        # Test multiple timesteps
        dts = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            result = integrate(sho_rhs, y0, t_array, dt)

            # Compare to analytical at t_final
            x_numerical = result[-1, 0]
            x_analytical = np.cos(t_final)
            error = abs(x_numerical - x_analytical)
            errors.append(error)

        errors = np.array(errors)

        # Compute convergence rates between consecutive timesteps
        # For 4th order: error(dt) / error(dt/2) should be ~16
        convergence_rates = errors[:-1] / errors[1:]

        # Log-log fit to verify order
        log_dt = np.log(dts)
        log_error = np.log(errors)
        order = np.polyfit(log_dt, log_error, 1)[0]

        # Should be 4th order (order ≈ 4.0, allowing some numerical tolerance)
        assert 3.5 < order < 4.5, \
            f"Expected 4th order convergence, got order {order:.2f}"

        # Individual convergence ratios should be close to 16 = 2^4
        for i, rate in enumerate(convergence_rates):
            assert 12 < rate < 20, \
                f"Timestep {dts[i]}: convergence rate {rate:.1f}, expected ~16"

    def test_rk4_long_time_accuracy(self):
        """Verify RK4 maintains accuracy over long integration times."""
        # Simple harmonic oscillator should remain accurate over many periods
        def sho_rhs(state, t):
            return np.array([state[1], -state[0]])

        y0 = np.array([1.0, 0.0])
        dt = 0.01
        t_final = 100.0  # Many periods (T = 2π ≈ 6.28)
        t_array = np.arange(0, t_final + dt, dt)

        result = integrate(sho_rhs, y0, t_array, dt)

        # Check multiple points throughout trajectory
        test_times = [10.0, 30.0, 50.0, 70.0, 100.0]
        for t_test in test_times:
            idx = np.argmin(np.abs(t_array - t_test))
            x_numerical = result[idx, 0]
            x_analytical = np.cos(t_array[idx])

            # Error should still be O(dt^4) even at long times
            error = abs(x_numerical - x_analytical)
            expected_error = 1e-6  # Rough estimate for dt=0.01
            assert error < 10 * expected_error, \
                f"Accuracy degraded at t={t_test}: error={error:.2e}"


class TestEnergyConservation:
    """Test energy conservation with convergence analysis."""

    def test_harmonic_simple_energy_drift_scaling(self):
        """Energy drift in conservative system should scale as O(dt^4)."""
        # Simple harmonic oscillator
        delta = 0.0
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        omega = 1.0

        y0 = np.array([1.0, 0.0])
        t_final = 50.0

        dts = [0.04, 0.02, 0.01]
        energy_drifts = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x = traj[:, 0]
            v = traj[:, 1]

            # Total energy
            energy = 0.5 * v**2 + 0.5 * alpha * x**2
            initial_energy = energy[0]

            # Maximum relative drift
            relative_drift = np.max(np.abs(energy - initial_energy)) / initial_energy
            energy_drifts.append(relative_drift)

        energy_drifts = np.array(energy_drifts)

        # Energy drift should scale as O(dt^4) for RK4
        # drift(dt) / drift(dt/2) should be ~16
        convergence_rates = energy_drifts[:-1] / energy_drifts[1:]

        for i, rate in enumerate(convergence_rates):
            # Allow wider range since energy is accumulated quantity
            # RK4 can actually perform better than 4th order for some problems
            assert 8 < rate < 40, \
                f"Energy drift convergence rate {rate:.1f} not ~16 (dt={dts[i]})"

    def test_duffing_undriven_energy_conservation(self):
        """Undamped, undriven Duffing oscillator energy drift scales correctly."""
        delta = 0.0
        alpha = 1.0
        beta = 5.0
        gamma = 0.0
        omega = 1.0

        y0 = np.array([0.5, 0.5])
        t_final = 30.0

        dts = [0.02, 0.01, 0.005]
        energy_drifts = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x = traj[:, 0]
            v = traj[:, 1]

            energy = 0.5 * v**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4
            initial_energy = energy[0]
            relative_drift = np.max(np.abs(energy - initial_energy)) / initial_energy
            energy_drifts.append(relative_drift)

        energy_drifts = np.array(energy_drifts)

        # Check 4th order scaling
        convergence_rates = energy_drifts[:-1] / energy_drifts[1:]

        for i, rate in enumerate(convergence_rates):
            assert 8 < rate < 40, \
                f"Duffing energy drift rate {rate:.1f} not ~16 (dt={dts[i]})"


class TestAnalyticalComparison:
    """Compare numerical solutions to analytical solutions with convergence."""

    def test_simple_harmonic_oscillator_convergence(self):
        """SHO solution should converge at 4th order to analytical."""
        delta = 0.0
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        omega = 1.0

        y0 = np.array([1.0, 0.0])
        t_final = 10.0

        dts = [0.04, 0.02, 0.01, 0.005]
        errors = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x_numerical = traj[-1, 0]
            omega_0 = np.sqrt(alpha)
            x_analytical = np.cos(omega_0 * t_final)

            error = abs(x_numerical - x_analytical)
            errors.append(error)

        errors = np.array(errors)

        # Verify 4th order convergence
        # error ~ C * dt^p, so log(error) ~ log(C) + p*log(dt)
        # slope of log(error) vs log(dt) gives the order p
        log_dt = np.log(dts)
        log_error = np.log(errors)
        order = np.polyfit(log_dt, log_error, 1)[0]

        # Should be 4th order (|order| ≈ 4.0)
        # Order is positive since error increases with dt
        assert 3.5 < abs(order) < 4.5, \
            f"SHO convergence order {abs(order):.2f}, expected ~4.0"

    def test_damped_oscillator_convergence(self):
        """Damped oscillator convergence to analytical solution."""
        # Analytical solution exists for underdamped case
        delta = 0.2
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        omega_val = 1.0

        y0 = np.array([1.0, 0.0])
        t_final = 5.0  # Shorter time to avoid excessive decay

        # Analytical solution for underdamped oscillator:
        # x'' + δx' + ω₀²x = 0
        # x(t) = exp(-δt/2) * [C₁*cos(ω_d*t) + C₂*sin(ω_d*t)]
        # where ω_d = sqrt(ω_0^2 - δ^2/4)
        omega_0 = np.sqrt(alpha)
        omega_d = np.sqrt(omega_0**2 - delta**2/4)

        # For x(0)=1, v(0)=0:
        # x(0) = C₁ = 1
        # v(0) = -δ/2 + C₂*ω_d = 0  =>  C₂ = δ/(2*ω_d)
        C1 = 1.0
        C2 = delta / (2 * omega_d)

        def analytical_solution(t):
            return np.exp(-delta * t / 2) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))

        dts = [0.08, 0.04, 0.02]
        errors = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega_val)

            # Use RMS error over multiple time points, not just endpoint
            # This is more robust to phase errors
            test_indices = np.linspace(0, len(t_array)-1, 20, dtype=int)
            rms_error = 0.0
            for idx in test_indices:
                x_numerical = traj[idx, 0]
                x_analytical = analytical_solution(t_array[idx])
                rms_error += (x_numerical - x_analytical)**2
            rms_error = np.sqrt(rms_error / len(test_indices))
            errors.append(rms_error)

        errors = np.array(errors)

        # Check 4th order convergence using log-log fit
        log_dt = np.log(dts)
        log_error = np.log(errors)
        order = abs(np.polyfit(log_dt, log_error, 1)[0])

        assert 3.0 < order < 5.0, \
            f"Damped oscillator convergence order {order:.2f}, expected ~4.0"

    def test_driven_resonance_steady_state_convergence(self):
        """Driven oscillator steady-state amplitude convergence."""
        alpha = 1.0
        omega_0 = np.sqrt(alpha)
        delta = 0.1
        beta = 0.0
        gamma = 0.1
        omega = omega_0  # at resonance

        y0 = np.array([0.0, 0.0])
        t_final = 100.0  # Long enough to reach steady state

        # Theoretical steady-state amplitude: A = γ / (δω)
        expected_amplitude = gamma / (delta * omega)

        dts = [0.04, 0.02, 0.01]
        amplitude_errors = []

        for dt in dts:
            t_array = np.arange(0, t_final + dt, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x = traj[:, 0]

            # Measure amplitude in last few periods
            measured_amplitude = np.max(np.abs(x[-2000:]))
            error = abs(measured_amplitude - expected_amplitude)
            amplitude_errors.append(error)

        amplitude_errors = np.array(amplitude_errors)

        # Should show convergence (may not be exactly 4th order due to transients)
        # Steady-state measurement may have limited precision
        # Check that we're close to expected value with finest dt
        finest_error_fraction = amplitude_errors[-1] / expected_amplitude

        assert finest_error_fraction < 0.05, \
            f"Steady-state amplitude error {finest_error_fraction:.1%} should be < 5%"

        # At least verify errors don't increase
        assert not (amplitude_errors[1] > 1.5 * amplitude_errors[0]), \
            "Amplitude error should not increase significantly"


class TestPhysicalBehavior:
    """Test that physical behavior matches expectations."""

    def test_damping_monotonic_energy_decrease(self):
        """Damped system should have monotonically decreasing energy."""
        delta = 0.2
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        omega = 1.0

        y0 = np.array([1.0, 0.0])
        dt = 0.01  # Fine enough to resolve dynamics
        t_array = np.arange(0, 50.0, dt)

        traj = integrate(oscillator_rhs, y0, t_array, dt,
                        delta, alpha, beta, gamma, omega)

        x = traj[:, 0]
        v = traj[:, 1]
        energy = 0.5 * v**2 + 0.5 * alpha * x**2

        # Energy should decrease (with numerical tolerance)
        energy_increases = np.sum(np.diff(energy) > 1e-10)
        fraction_increasing = energy_increases / len(energy)

        # Allow < 0.1% of points to show small increases due to numerical noise
        assert fraction_increasing < 0.001, \
            f"Energy increased in {fraction_increasing:.1%} of steps (damped system)"

        # Final energy should be much less than initial
        assert energy[-1] < 0.01 * energy[0], \
            "Damping should significantly reduce energy"

    def test_period_measurement_convergence(self):
        """Period measurement should converge as dt decreases."""
        delta = 0.0
        alpha = 4.0  # ω₀ = 2
        beta = 0.0
        gamma = 0.0
        omega = 1.0

        y0 = np.array([1.0, 0.0])
        omega_0 = np.sqrt(alpha)
        expected_period = 2 * np.pi / omega_0

        dts = [0.01, 0.005, 0.0025]
        measured_periods = []

        for dt in dts:
            t_array = np.arange(0, 30.0, dt)
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x = traj[:, 0]

            # Find zero crossings
            zero_crossings = []
            for i in range(len(x)-1):
                if x[i] > 0 and x[i+1] <= 0:
                    t_cross = t_array[i] - x[i] * dt / (x[i+1] - x[i])
                    zero_crossings.append(t_cross)

            if len(zero_crossings) < 3:
                continue

            periods = np.diff(zero_crossings)
            measured_period = np.mean(periods)
            measured_periods.append(measured_period)

        # Period measurements should converge to expected value
        errors = np.abs(np.array(measured_periods) - expected_period)

        # Should converge (approximately 4th order, but period measurement
        # is not the primary integrated quantity)
        for i in range(len(errors)-1):
            ratio = errors[i] / errors[i+1]
            assert ratio > 2, \
                f"Period measurement should converge (got ratio {ratio:.1f})"

    def test_duffing_hardening_frequency_shift(self):
        """Duffing oscillator with β>0 shows amplitude-dependent frequency."""
        alpha = 1.0
        beta = 5.0
        delta = 0.02  # light damping to allow measurement
        gamma = 0.0
        omega = 1.0

        dt = 0.005
        t_array = np.arange(0, 100.0, dt)

        # Different amplitudes
        amplitudes = [0.2, 0.5, 0.8]
        frequencies = []

        for amp in amplitudes:
            y0 = np.array([amp, 0.0])
            traj = integrate(oscillator_rhs, y0, t_array, dt,
                           delta, alpha, beta, gamma, omega)

            x = traj[:, 0]

            # Measure frequency from later part (after transients)
            x_stable = x[5000:]
            t_stable = t_array[5000:]

            # Find peaks
            peaks = []
            for i in range(1, len(x_stable)-1):
                if x_stable[i] > x_stable[i-1] and x_stable[i] > x_stable[i+1]:
                    peaks.append(t_stable[i])

            if len(peaks) < 3:
                continue

            # Average period
            period = np.mean(np.diff(peaks))
            freq = 2 * np.pi / period
            frequencies.append(freq)

        if len(frequencies) < 2:
            pytest.skip("Could not measure frequencies reliably")

        # For hardening spring (β>0), frequency increases with amplitude
        for i in range(len(frequencies)-1):
            assert frequencies[i+1] > frequencies[i], \
                "Hardening Duffing should show increasing frequency with amplitude"


class TestVisualization:
    """Test that visualization components are created correctly."""

    def test_setup_visualization_structure(self):
        """Verify visualization setup returns complete structure."""
        t = np.linspace(0, 10, 100)
        x = np.sin(t)
        v = np.cos(t)

        elements = setup_visualization(x, v, alpha=1.0, beta=0.0,
                                      gamma=0.0, omega=1.0, dt=0.1)

        required_keys = ['fig', 'ax_potential', 'ax_anim', 'ax_phase',
                        'particle_dot', 'velocity_arrow', 'time_text',
                        'mass_circle', 'spring_line', 'trajectory_line',
                        'current_point', 'recent_line', 'X_MIN', 'X_MAX']

        for key in required_keys:
            assert key in elements, f"Missing key: {key}"

        # Verify axis limits are set
        xlim = elements['ax_phase'].get_xlim()
        assert xlim[0] < xlim[1], "Phase plot x-axis not set correctly"

        plt.close(elements['fig'])

    def test_animation_creation_no_errors(self):
        """Animation creation should complete without errors."""
        delta, alpha, beta = 0.1, 1.0, 0.0
        gamma, omega = 0.5, 1.0

        y0 = np.array([0.2, 0.0])
        dt = 0.05
        t = np.arange(0, 5.0, dt)

        traj = integrate(oscillator_rhs, y0, t, dt,
                        delta, alpha, beta, gamma, omega)
        x, v = traj[:, 0], traj[:, 1]

        anim, fig = create_animation(x, v, t, dt, alpha, beta, gamma, omega,
                                     frame_stride=5)

        assert anim is not None
        assert hasattr(anim, 'save'), "Animation should have save method"

        plt.close(fig)

    def test_all_presets_integrate_successfully(self):
        """All preset modes should integrate without errors."""
        for mode_name, params in PRESETS.items():
            param_dict = params.copy()
            param_dict.pop('description', None)

            delta = param_dict['delta']
            alpha = param_dict['alpha']
            beta = param_dict['beta']
            gamma = param_dict['gamma']
            omega = param_dict['omega']

            y0 = np.array([0.2, 0.0])
            dt = 0.02
            t = np.arange(0, 10.0, dt)

            traj = integrate(oscillator_rhs, y0, t, dt,
                           delta, alpha, beta, gamma, omega)

            assert traj.shape == (len(t), 2), \
                f"Mode {mode_name}: wrong shape"
            assert np.all(np.isfinite(traj)), \
                f"Mode {mode_name}: non-finite values"

            # Check energy makes sense
            x, v = traj[:, 0], traj[:, 1]
            energy = 0.5 * v**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4
            assert np.all(energy >= 0), \
                f"Mode {mode_name}: negative kinetic/potential energy"


class TestNumericalStability:
    """Test numerical stability across parameter ranges."""

    def test_chaotic_regime_stability(self):
        """Chaotic Duffing should remain stable over long times."""
        delta, alpha, beta = 0.02, 1.0, 5.0
        gamma, omega = 8.0, 0.5

        y0 = np.array([0.2, 0.0])
        dt = 0.02
        t = np.arange(0, 200.0, dt)  # Long time

        traj = integrate(oscillator_rhs, y0, t, dt,
                        delta, alpha, beta, gamma, omega)

        assert np.all(np.isfinite(traj)), \
            "Chaotic integration produced NaN/Inf"

        # Check bounded behavior (strange attractor should be bounded)
        x_max = np.max(np.abs(traj[:, 0]))
        v_max = np.max(np.abs(traj[:, 1]))

        assert x_max < 100, "Position unbounded in chaotic regime"
        assert v_max < 100, "Velocity unbounded in chaotic regime"

    def test_timestep_convergence_consistency(self):
        """Solution should converge consistently for various parameters."""
        test_cases = [
            {'delta': 0.0, 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},  # Conservative
            {'delta': 0.2, 'alpha': 1.0, 'beta': 0.0, 'gamma': 1.0},  # Driven
            {'delta': 0.1, 'alpha': 1.0, 'beta': 5.0, 'gamma': 0.0},  # Nonlinear
        ]

        for params in test_cases:
            delta = params['delta']
            alpha = params['alpha']
            beta = params['beta']
            gamma = params['gamma']
            omega = 1.0

            y0 = np.array([0.5, 0.0])
            t_final = 10.0

            # Compare two timesteps
            dt_coarse = 0.04
            dt_fine = 0.02

            t_coarse = np.arange(0, t_final + dt_coarse, dt_coarse)
            t_fine = np.arange(0, t_final + dt_fine, dt_fine)

            traj_coarse = integrate(oscillator_rhs, y0, t_coarse, dt_coarse,
                                   delta, alpha, beta, gamma, omega)
            traj_fine = integrate(oscillator_rhs, y0, t_fine, dt_fine,
                                 delta, alpha, beta, gamma, omega)

            # Compare final states
            x_coarse = traj_coarse[-1, 0]
            x_fine = traj_fine[-1, 0]

            # With dt halved, error should decrease by factor ~16 (4th order)
            # So solutions should agree well
            relative_diff = abs(x_coarse - x_fine) / (abs(x_fine) + 1e-10)

            assert relative_diff < 0.02, \
                f"Timestep convergence failed for params {params}: diff={relative_diff:.2%}"

    def test_energy_bounds_physical(self):
        """Energy should remain physically reasonable."""
        # High-energy initial condition
        delta, alpha, beta = 0.05, 1.0, 1.0
        gamma, omega = 0.1, 1.0

        y0 = np.array([2.0, 2.0])
        dt = 0.01
        t = np.arange(0, 50.0, dt)

        traj = integrate(oscillator_rhs, y0, t, dt,
                        delta, alpha, beta, gamma, omega)

        x, v = traj[:, 0], traj[:, 1]
        energy = 0.5 * v**2 + 0.5 * alpha * x**2 + 0.25 * beta * x**4

        # Energy should not grow unboundedly for damped system
        assert np.max(energy) < 100 * energy[0], \
            "Energy grew unreasonably in damped driven system"


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v', '--tb=short'])
