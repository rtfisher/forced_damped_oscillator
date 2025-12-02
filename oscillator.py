#!/usr/bin/env python3
"""
Unified Oscillator Simulator
Simulates driven, damped oscillator systems with configurable parameters.

General equation: x'' + δx' + αx + βx³ = γcos(ωt)
- δ: damping coefficient
- α: linear restoring force coefficient
- β: cubic (nonlinear) restoring force coefficient
- γ: driving force amplitude
- ω: driving force frequency

Usage:
    python oscillator.py --mode harmonic-simple
    python oscillator.py --mode duffing-chaotic --time 100
    python oscillator.py --custom --delta 0.2 --alpha 1.0 --gamma 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
import argparse
import sys

# ============================
# Preset Configurations
# ============================
PRESETS = {
    'harmonic-simple': {
        'delta': 0.0,    # no damping
        'alpha': 1.0,    # linear restoring force
        'beta': 0.0,     # no nonlinearity
        'gamma': 0.0,    # no driving force
        'omega': 0.5,    # (unused without driving)
        'description': 'Simple harmonic oscillator (undamped, undriven)'
    },
    'harmonic-damped': {
        'delta': 0.1,    # light damping
        'alpha': 1.0,
        'beta': 0.0,
        'gamma': 0.0,
        'omega': 0.5,
        'description': 'Damped harmonic oscillator (no driving force)'
    },
    'harmonic-driven': {
        'delta': 0.2,    # damping
        'alpha': 1.0,
        'beta': 0.0,
        'gamma': 1.0,    # driving force
        'omega': 1.0,    # driving frequency
        'description': 'Driven damped harmonic oscillator'
    },
    'duffing-damped': {
        'delta': 0.1,
        'alpha': 1.0,
        'beta': 5.0,     # cubic nonlinearity
        'gamma': 0.0,
        'omega': 0.5,
        'description': 'Damped Duffing oscillator (no driving force)'
    },
    'duffing-driven': {
        'delta': 0.02,
        'alpha': 1.0,
        'beta': 5.0,
        'gamma': 8.0,
        'omega': 0.5,
        'description': 'Driven damped Duffing oscillator'
    },
    'duffing-chaotic': {
        'delta': 0.02,
        'alpha': 1.0,
        'beta': 5.0,
        'gamma': 8.0,
        'omega': 0.5,
        'description': 'Duffing oscillator in chaotic regime'
    },
}

# ============================
# Integration Engine
# ============================

def oscillator_rhs(state, time, delta, alpha, beta, gamma, omega):
    """
    Right-hand side of the oscillator ODE.

    Equation: x'' + δx' + αx + βx³ = γcos(ωt)

    State vector: [x, v] where v = dx/dt
    Returns: [dx/dt, dv/dt]
    """
    x, v = state
    dxdt = v
    dvdt = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*time)
    return np.array([dxdt, dvdt], dtype=float)

def rk4_step(f, y, time, h, *args):
    """4th-order Runge-Kutta integration step."""
    k1 = f(y, time, *args)
    k2 = f(y + 0.5*h*k1, time + 0.5*h, *args)
    k3 = f(y + 0.5*h*k2, time + 0.5*h, *args)
    k4 = f(y + h*k3, time + h, *args)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(f, y0, t_array, h, *args):
    """Integrate ODE using RK4."""
    y = np.zeros((len(t_array), len(y0)), dtype=float)
    y[0] = y0
    for i in range(1, len(t_array)):
        y[i] = rk4_step(f, y[i-1], t_array[i-1], h, *args)
    return y

# ============================
# Visualization Setup
# ============================

def make_spring(x0, x1, n_coils=15, amp=0.15, npts=200):
    """Generate spring coordinates for visualization."""
    if abs(x1 - x0) < 0.01:
        return [x0, x1], [0, 0]
    xs = np.linspace(x0, x1, npts)
    ys = np.sin(np.linspace(0, n_coils*2*np.pi, xs.size)) * amp
    return xs, ys

def setup_visualization(x, v, alpha, beta, gamma, omega, dt):
    """Set up the 3-panel animation figure."""

    # Define symmetric x-range based on actual motion with margin
    x_amplitude = max(abs(x.min()), abs(x.max()))
    margin_factor = 1.1
    X_MIN, X_MAX = -x_amplitude * margin_factor, x_amplitude * margin_factor

    # Create figure with 3 panels
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.4, 1.4], hspace=0.25)

    ax_potential = fig.add_subplot(gs[0, 0])
    ax_anim = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[2, 0])

    # ---- Top panel: Potential energy ----
    x_pot = np.linspace(X_MIN, X_MAX, 1000)
    V_pot = (alpha/2) * x_pot**2 + (beta/4) * x_pot**4

    ax_potential.plot(x_pot, V_pot, 'k-', linewidth=2, label='Potential V(x)')
    ax_potential.set_xlabel('Position x')
    ax_potential.set_ylabel('Potential Energy V(x)')

    # Set title based on potential type
    if beta == 0:
        ax_potential.set_title('Potential Energy: V(x) = (α/2)x²')
    else:
        ax_potential.set_title('Potential Energy: V(x) = (α/2)x² + (β/4)x⁴')
    ax_potential.grid(True, alpha=0.3)

    # Mark equilibrium points
    if alpha < 0 and beta > 0:
        # Double-well potential
        x_stable = np.sqrt(-alpha/beta)
        stable_positions = [-x_stable, x_stable]
        for eq_x in stable_positions:
            eq_energy = (alpha/2) * eq_x**2 + (beta/4) * eq_x**4
            ax_potential.plot(eq_x, eq_energy, 'go', markersize=8,
                            label='Stable' if eq_x == stable_positions[0] else "", zorder=10)
        eq_energy = 0
        ax_potential.plot(0, eq_energy, 'ro', markersize=8,
                        label='Unstable', zorder=10)
    else:
        # Single-well potential
        eq_energy = 0
        ax_potential.plot(0, eq_energy, 'go', markersize=8,
                        label='Stable equilibrium', zorder=10)

    # Current particle position marker
    particle_dot = ax_potential.scatter([], [], s=80, c='red', zorder=15,
                                       label='Current position')
    velocity_arrow = ax_potential.annotate('', xy=(0, 0), xytext=(0, 0),
                                           arrowprops=dict(arrowstyle='->',
                                                         color='blue', lw=2),
                                           zorder=12)

    ax_potential.set_xlim(X_MIN, X_MAX)
    V_min, V_max = np.min(V_pot), np.max(V_pot)
    V_margin = 0.1 * (V_max - V_min) if V_max > V_min else 0.1
    ax_potential.set_ylim(V_min - V_margin, V_max + V_margin)
    ax_potential.legend(loc='upper right')

    # ---- Middle panel: Physical system animation ----
    ax_anim.set_xlim(X_MIN, X_MAX)
    y_range = X_MAX - X_MIN
    Y_MIN, Y_MAX = -y_range/2, y_range/2
    ax_anim.set_ylim(Y_MIN, Y_MAX)
    ax_anim.set_xlabel("Displacement x")
    ax_anim.set_yticks([])
    ax_anim.set_title("Physical System: Mass-Spring Oscillator")

    # Time display
    time_text = ax_anim.text(0.05, 0.92, '', transform=ax_anim.transAxes,
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                    edgecolor="black", alpha=0.9))

    # Wall at x=0
    wall_x = 0.0
    wall_height = 0.6 * (Y_MAX - Y_MIN)
    ax_anim.plot([wall_x, wall_x], [-wall_height/2, wall_height/2], 'k-',
                linewidth=3, label='Equilibrium')

    # Forcing function arrow (if driven)
    force_arrow = None
    force_scale = 0.4 * (X_MAX - X_MIN)
    force_y_position = 0.7 * Y_MAX
    if gamma > 0:
        force_arrow = ax_anim.annotate('', xy=(0, force_y_position),
                                       xytext=(0, force_y_position),
                                       arrowprops=dict(arrowstyle='->',
                                                     color='green', lw=3),
                                       zorder=15)
        ax_anim.plot([], [], 'g-', linewidth=3, label='Forcing γcos(ωt)')

    # Mass as ellipse
    mass_radius = 0.08 * (X_MAX - X_MIN) / 3.0
    ellipse_width = 2 * mass_radius * 0.3
    ellipse_height = 2 * mass_radius
    mass_circle = Ellipse((x[0], 0), ellipse_width, ellipse_height,
                         fill=True, color='red', zorder=10)
    ax_anim.add_patch(mass_circle)

    # Spring
    spring_line, = ax_anim.plot([], [], 'b-', linewidth=2, alpha=0.8)

    ax_anim.legend(loc='upper right')

    # ---- Bottom panel: Phase portrait ----
    ax_phase.set_title("Phase Portrait: Velocity vs. Position")
    ax_phase.set_xlabel("Position x")
    ax_phase.set_ylabel("Velocity dx/dt")

    ax_phase.set_xlim(X_MIN, X_MAX)
    v_range = v.max() - v.min()
    v_margin = 0.1 * v_range if v_range > 0 else 0.1
    ax_phase.set_ylim(v.min() - v_margin, v.max() + v_margin)
    ax_phase.grid(True, alpha=0.3)

    # Equilibrium points in phase space
    if alpha < 0 and beta > 0:
        x_stable = np.sqrt(-alpha/beta)
        ax_phase.plot([-x_stable, x_stable], [0, 0], 'go', markersize=8,
                     alpha=0.7, label='Stable equilibria')
        ax_phase.plot(0, 0, 'ro', markersize=8, alpha=0.7, label='Unstable equilibrium')
    else:
        ax_phase.plot(0, 0, 'go', markersize=8, alpha=0.7, label='Stable equilibrium')

    # Trajectory lines (blue=history, red=recent)
    trajectory_line, = ax_phase.plot([], [], 'blue', linewidth=0.8, alpha=0.8,
                                     label='History')
    current_point = ax_phase.scatter([], [], s=50, c='red', zorder=10,
                                    label='Current state')
    recent_length = int(1.0 / dt)
    recent_line, = ax_phase.plot([], [], 'red', linewidth=2, alpha=0.7,
                                 label='Recent (1 time unit)')

    ax_phase.legend(loc='upper right')

    # Package all elements
    elements = {
        'fig': fig,
        'ax_potential': ax_potential,
        'ax_anim': ax_anim,
        'ax_phase': ax_phase,
        'particle_dot': particle_dot,
        'velocity_arrow': velocity_arrow,
        'time_text': time_text,
        'mass_circle': mass_circle,
        'spring_line': spring_line,
        'force_arrow': force_arrow,
        'trajectory_line': trajectory_line,
        'current_point': current_point,
        'recent_line': recent_line,
        'X_MIN': X_MIN,
        'X_MAX': X_MAX,
        'Y_MIN': Y_MIN,
        'Y_MAX': Y_MAX,
        'wall_x': wall_x,
        'force_scale': force_scale,
        'force_y_position': force_y_position,
        'recent_length': recent_length
    }

    return elements

# ============================
# Animation Functions
# ============================

def create_animation(x, v, t, dt, alpha, beta, gamma, omega, frame_stride=2):
    """Create and return the animation."""

    # Set up visualization
    elements = setup_visualization(x, v, alpha, beta, gamma, omega, dt)

    # Animation parameters
    frame_indices = np.arange(0, len(t), frame_stride)
    N = len(t)

    # Trajectory data storage
    traj_x_data = []
    traj_v_data = []

    def init():
        """Initialize animation."""
        # Initialize spring
        xs, ys = make_spring(elements['wall_x'], x[0])
        elements['spring_line'].set_data(xs, ys)

        # Initialize potential panel
        V_current = (alpha/2) * x[0]**2 + (beta/4) * x[0]**4
        elements['particle_dot'].set_offsets([[x[0], V_current]])
        elements['velocity_arrow'].set_position((x[0], V_current))
        elements['velocity_arrow'].xy = (x[0], V_current)

        # Initialize empty plots
        elements['trajectory_line'].set_data([], [])
        elements['recent_line'].set_data([], [])
        elements['current_point'].set_offsets(np.empty((0, 2)))

        # Initialize time display
        elements['time_text'].set_text('t = 0.00')

        returns = [elements['mass_circle'], elements['spring_line'],
                  elements['trajectory_line'], elements['recent_line'],
                  elements['current_point'], elements['time_text'],
                  elements['particle_dot'], elements['velocity_arrow']]

        if elements['force_arrow']:
            returns.append(elements['force_arrow'])

        return tuple(returns)

    def animate(frame_k):
        """Update animation frame."""
        nonlocal traj_x_data, traj_v_data

        i = frame_indices[frame_k]
        xi, vi = x[i], v[i]
        current_time = t[i]

        # Clear trajectory data when animation loops back to start
        if i == 0:
            traj_x_data = []
            traj_v_data = []

        # Update time display
        elements['time_text'].set_text(f't = {current_time:6.2f}')

        # ---- Update potential energy panel ----
        V_current = (alpha/2) * xi**2 + (beta/4) * xi**4
        elements['particle_dot'].set_offsets([[xi, V_current]])

        # Velocity arrow
        velocity_scale = 0.3
        arrow_end_x = xi + velocity_scale * vi
        arrow_end_y = V_current
        elements['velocity_arrow'].set_position((xi, V_current))
        elements['velocity_arrow'].xy = (arrow_end_x, arrow_end_y)

        # ---- Update physical animation ----
        elements['mass_circle'].center = (xi, 0)

        xs, ys = make_spring(elements['wall_x'], xi)
        elements['spring_line'].set_data(xs, ys)

        # Update forcing arrow if present
        if elements['force_arrow']:
            current_force = gamma * np.cos(omega * current_time)
            arrow_length = elements['force_scale'] * current_force / gamma
            arrow_start_x = 0
            arrow_end_x = arrow_start_x + arrow_length
            elements['force_arrow'].set_position((arrow_start_x, elements['force_y_position']))
            elements['force_arrow'].xy = (arrow_end_x, elements['force_y_position'])

        # ---- Update phase portrait ----
        # Add current point to history
        traj_x_data.append(xi)
        traj_v_data.append(vi)

        # Update full history trajectory (blue line)
        elements['trajectory_line'].set_data(traj_x_data, traj_v_data)

        # Update current position marker
        elements['current_point'].set_offsets([[xi, vi]])

        # Update recent trajectory (red line)
        recent_start = max(0, len(traj_x_data) - elements['recent_length'])
        if len(traj_x_data) > 1:
            recent_x = traj_x_data[recent_start:]
            recent_v = traj_v_data[recent_start:]
            elements['recent_line'].set_data(recent_x, recent_v)

        returns = [elements['mass_circle'], elements['spring_line'],
                  elements['trajectory_line'], elements['recent_line'],
                  elements['current_point'], elements['time_text'],
                  elements['particle_dot'], elements['velocity_arrow']]

        if elements['force_arrow']:
            returns.append(elements['force_arrow'])

        return tuple(returns)

    # Create animation
    anim = animation.FuncAnimation(
        elements['fig'], animate, init_func=init,
        frames=len(frame_indices), interval=1000*dt*frame_stride,
        blit=True, repeat=True
    )

    return anim, elements['fig']

# ============================
# Main Program
# ============================

def main():
    parser = argparse.ArgumentParser(
        description='Unified oscillator simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available preset modes:
  harmonic-simple    : Simple harmonic oscillator (no damping, no driving)
  harmonic-damped    : Damped harmonic oscillator
  harmonic-driven    : Driven damped harmonic oscillator
  duffing-damped     : Damped Duffing oscillator (nonlinear)
  duffing-driven     : Driven damped Duffing oscillator
  duffing-chaotic    : Duffing oscillator in chaotic regime

Examples:
  python oscillator.py --mode harmonic-simple
  python oscillator.py --mode duffing-chaotic --time 100
  python oscillator.py --custom --delta 0.2 --alpha 1.0 --gamma 1.0 --omega 1.0
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--mode', choices=PRESETS.keys(),
                           help='Use a preset configuration')
    mode_group.add_argument('--custom', action='store_true',
                           help='Use custom parameters')

    # Parameter overrides
    parser.add_argument('--delta', type=float, help='Damping coefficient')
    parser.add_argument('--alpha', type=float, help='Linear restoring force coefficient')
    parser.add_argument('--beta', type=float, help='Cubic restoring force coefficient')
    parser.add_argument('--gamma', type=float, help='Driving force amplitude')
    parser.add_argument('--omega', type=float, help='Driving force frequency')

    # Initial conditions
    parser.add_argument('--x0', type=float, default=0.2, help='Initial position (default: 0.2)')
    parser.add_argument('--v0', type=float, default=0.0, help='Initial velocity (default: 0.0)')

    # Simulation parameters
    parser.add_argument('--time', type=float, default=60.0,
                       help='Simulation time (default: 60.0)')
    parser.add_argument('--dt', type=float, default=0.02,
                       help='Time step (default: 0.02)')
    parser.add_argument('--stride', type=int, default=2,
                       help='Frame stride for animation (default: 2)')

    # Output options
    parser.add_argument('--save-gif', action='store_true', help='Save animation as GIF')
    parser.add_argument('--save-mp4', action='store_true', help='Save animation as MP4')
    parser.add_argument('--output', type=str, default='oscillator_animation',
                       help='Output filename (without extension)')

    args = parser.parse_args()

    # Load parameters
    if args.custom:
        # Custom mode requires all parameters
        required = ['delta', 'alpha', 'beta', 'gamma', 'omega']
        missing = [p for p in required if getattr(args, p) is None]
        if missing:
            parser.error(f"--custom mode requires: {', '.join('--'+p for p in missing)}")

        params = {
            'delta': args.delta,
            'alpha': args.alpha,
            'beta': args.beta,
            'gamma': args.gamma,
            'omega': args.omega
        }
        description = "Custom configuration"
    else:
        # Load preset and apply overrides
        params = PRESETS[args.mode].copy()
        description = params.pop('description')

        # Apply command-line overrides
        for param in ['delta', 'alpha', 'beta', 'gamma', 'omega']:
            if getattr(args, param) is not None:
                params[param] = getattr(args, param)

    # Extract parameters
    delta = params['delta']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    omega = params['omega']

    # Print configuration
    print("="*60)
    print(f"Configuration: {args.mode if args.mode else 'custom'}")
    print(f"Description: {description}")
    print("="*60)
    print(f"Parameters:")
    print(f"  δ (delta)  = {delta:8.4f}  (damping)")
    print(f"  α (alpha)  = {alpha:8.4f}  (linear restoring force)")
    print(f"  β (beta)   = {beta:8.4f}  (cubic restoring force)")
    print(f"  γ (gamma)  = {gamma:8.4f}  (driving amplitude)")
    print(f"  ω (omega)  = {omega:8.4f}  (driving frequency)")
    print(f"\nInitial conditions:")
    print(f"  x₀ = {args.x0:.3f}")
    print(f"  v₀ = {args.v0:.3f}")
    print(f"\nSimulation settings:")
    print(f"  Time: {args.time:.1f} s")
    print(f"  dt: {args.dt:.4f} s")
    print("="*60)

    # Time array
    t = np.arange(0.0, args.time, args.dt)

    # Initial conditions
    y0 = np.array([args.x0, args.v0], dtype=float)

    # Integrate
    print("\nIntegrating...")
    traj = integrate(oscillator_rhs, y0, t, args.dt, delta, alpha, beta, gamma, omega)
    x = traj[:, 0]
    v = traj[:, 1]

    print(f"Position range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Velocity range: [{v.min():.3f}, {v.max():.3f}]")

    # Create animation
    print("\nCreating animation...")
    anim, fig = create_animation(x, v, t, args.dt, alpha, beta, gamma, omega,
                                 frame_stride=args.stride)

    # Save if requested
    if args.save_gif:
        filename = f"{args.output}.gif"
        print(f"Saving animation as {filename}...")
        anim.save(filename, writer="pillow", fps=20, dpi=80)
        print(f"Saved: {filename}")

    if args.save_mp4:
        filename = f"{args.output}.mp4"
        print(f"Saving animation as {filename}...")
        anim.save(filename, writer="ffmpeg", fps=60, dpi=200)
        print(f"Saved: {filename}")

    # Display
    print("\nDisplaying animation...")
    print("Close window to exit.")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
