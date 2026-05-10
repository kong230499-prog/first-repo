"""
Project:
Delay-Coordinate State Reconstruction for Data-Driven Control
under Noisy Partial Measurements

Author:
Kenneth O Tze Jin

Description:
This script simulates a nonlinear Van der Pol oscillator, measures only noisy
position, builds delay-coordinate vectors, and trains data-driven models to
reconstruct the full state.

The main goal is to test whether delay-coordinate features improve state
reconstruction when only noisy partial measurements are available.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Simulate nonlinear system
# ============================================================

def van_der_pol(t, state, mu=1.0, u=0.0):
    """
    Van der Pol oscillator.

    State:
        x1 = position
        x2 = velocity

    Dynamics:
        x1_dot = x2
        x2_dot = mu * (1 - x1^2) * x2 - x1 + u

    Parameters:
        t: time
        state: [x1, x2]
        mu: nonlinearity parameter
        u: control input, set to zero for open-loop simulation

    Returns:
        state derivative
    """
    x1, x2 = state

    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1 + u

    return [dx1, dx2]


def simulate_vdp(
    t_final=30.0,
    dt=0.01,
    initial_state=(2.0, 0.0),
    mu=1.0
):
    """
    Simulate the Van der Pol oscillator.

    Returns:
        t: time vector
        X: true state matrix with shape [num_samples, 2]
    """
    t_eval = np.arange(0.0, t_final, dt)

    sol = solve_ivp(
        fun=lambda t, y: van_der_pol(t, y, mu=mu),
        t_span=(0.0, t_final),
        y0=initial_state,
        t_eval=t_eval,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError("Simulation failed.")

    t = sol.t
    X = sol.y.T

    return t, X


# ============================================================
# 2. Add noisy partial measurement
# ============================================================

def create_noisy_measurement(X, noise_std=0.1, random_seed=42):
    """
    Measure only position x1 and add Gaussian noise.

    Parameters:
        X: true state matrix [x1, x2]
        noise_std: standard deviation of measurement noise

    Returns:
        y_clean: clean measured position
        y_noisy: noisy measured position
    """
    rng = np.random.default_rng(random_seed)

    y_clean = X[:, 0]
    noise = rng.normal(loc=0.0, scale=noise_std, size=y_clean.shape)

    y_noisy = y_clean + noise

    return y_clean, y_noisy


# ============================================================
# 3. Build delay-coordinate features
# ============================================================

def build_delay_coordinates(y, X_true, delay_steps=5, delay_gap=5):
    """
    Build delay-coordinate vectors from one measured signal.

    Example:
        z_k = [y_k, y_{k-gap}, y_{k-2gap}, ..., y_{k-(m-1)gap}]

    Parameters:
        y: measured output signal
        X_true: true state matrix
        delay_steps: number of delayed measurements
        delay_gap: number of time steps between delays

    Returns:
        Z: delay-coordinate feature matrix
        X_target: matching true states
    """
    max_lag = (delay_steps - 1) * delay_gap

    Z = []
    X_target = []

    for k in range(max_lag, len(y)):
        delay_vector = []

        for j in range(delay_steps):
            index = k - j * delay_gap
            delay_vector.append(y[index])

        Z.append(delay_vector)
        X_target.append(X_true[k])

    Z = np.array(Z)
    X_target = np.array(X_target)

    return Z, X_target


# ============================================================
# 4. Train reconstruction models
# ============================================================

def train_ridge_model(Z_train, X_train):
    """
    Train a Ridge regression model.

    This is the simple baseline model.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(Z_train, X_train)

    return model


def train_mlp_model(Z_train, X_train):
    """
    Train a neural-network regression model.

    This is the machine-learning model.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="tanh",
            solver="adam",
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(Z_train, X_train)

    return model


# ============================================================
# 5. Evaluate models
# ============================================================

def evaluate_model(model, Z_test, X_test, model_name="model"):
    """
    Evaluate reconstruction performance.

    Returns:
        X_pred: predicted full state
        metrics: dictionary with RMSE and R2 values
    """
    X_pred = model.predict(Z_test)

    rmse_total = np.sqrt(mean_squared_error(X_test, X_pred))
    rmse_position = np.sqrt(mean_squared_error(X_test[:, 0], X_pred[:, 0]))
    rmse_velocity = np.sqrt(mean_squared_error(X_test[:, 1], X_pred[:, 1]))

    r2_position = r2_score(X_test[:, 0], X_pred[:, 0])
    r2_velocity = r2_score(X_test[:, 1], X_pred[:, 1])

    metrics = {
        "model": model_name,
        "rmse_total": rmse_total,
        "rmse_position": rmse_position,
        "rmse_velocity": rmse_velocity,
        "r2_position": r2_position,
        "r2_velocity": r2_velocity
    }

    return X_pred, metrics


def print_metrics(metrics):
    """
    Print model performance clearly.
    """
    print("\n======================================")
    print(f"Model: {metrics['model']}")
    print("======================================")
    print(f"Total RMSE:      {metrics['rmse_total']:.4f}")
    print(f"Position RMSE:   {metrics['rmse_position']:.4f}")
    print(f"Velocity RMSE:   {metrics['rmse_velocity']:.4f}")
    print(f"Position R2:     {metrics['r2_position']:.4f}")
    print(f"Velocity R2:     {metrics['r2_velocity']:.4f}")


# ============================================================
# 6. Plot results
# ============================================================

def plot_measurement(t, y_clean, y_noisy):
    """
    Plot clean and noisy measured position.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_clean, label="Clean position")
    plt.plot(t, y_noisy, label="Noisy measured position", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.title("Partial Noisy Measurement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_reconstruction(X_test, X_pred, model_name="model", num_points=1000):
    """
    Plot true and reconstructed states.
    """
    n = min(num_points, len(X_test))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(X_test[:n, 0], label="True position")
    plt.plot(X_pred[:n, 0], label="Reconstructed position", linestyle="--")
    plt.ylabel("Position")
    plt.title(f"State Reconstruction Using {model_name}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(X_test[:n, 1], label="True velocity")
    plt.plot(X_pred[:n, 1], label="Reconstructed velocity", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_phase_portrait(X_test, X_pred, model_name="model"):
    """
    Compare true and reconstructed phase portrait.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(X_test[:, 0], X_test[:, 1], label="True state", alpha=0.8)
    plt.plot(X_pred[:, 0], X_pred[:, 1], label="Reconstructed state", alpha=0.8)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"Phase Portrait: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Optional simple control test
# ============================================================

def simple_pd_control(x_hat, reference=0.0, kp=2.0, kd=0.5):
    """
    Simple PD controller using reconstructed state.

    Control law:
        u = -kp * (position - reference) - kd * velocity

    This is only a draft for the stretch goal.
    """
    position_hat = x_hat[0]
    velocity_hat = x_hat[1]

    u = -kp * (position_hat - reference) - kd * velocity_hat

    return u


def simulate_controlled_vdp(
    model,
    delay_steps=5,
    delay_gap=5,
    t_final=20.0,
    dt=0.01,
    initial_state=(2.0, 0.0),
    mu=1.0,
    noise_std=0.05,
    reference=0.0
):
    """
    Simulate closed-loop control using reconstructed state.

    This is an early draft. It uses the trained reconstruction model
    to estimate the current full state from delayed noisy measurements.

    Important:
        This is a simple stretch-goal controller.
        It should be treated as experimental.
    """
    rng = np.random.default_rng(123)

    num_steps = int(t_final / dt)
    t = np.arange(num_steps) * dt

    X = np.zeros((num_steps, 2))
    U = np.zeros(num_steps)
    Y = np.zeros(num_steps)

    X[0, :] = initial_state
    Y[0] = X[0, 0] + rng.normal(0.0, noise_std)

    # Store measurement history.
    y_history = [Y[0]]

    for k in range(num_steps - 1):

        # If we do not have enough history yet, use a simple fallback estimate.
        max_lag = (delay_steps - 1) * delay_gap

        if len(y_history) <= max_lag:
            x_hat = np.array([Y[k], 0.0])
        else:
            delay_vector = []

            current_index = len(y_history) - 1

            for j in range(delay_steps):
                index = current_index - j * delay_gap
                delay_vector.append(y_history[index])

            delay_vector = np.array(delay_vector).reshape(1, -1)
            x_hat = model.predict(delay_vector).flatten()

        # Compute control input.
        u = simple_pd_control(
            x_hat=x_hat,
            reference=reference,
            kp=2.0,
            kd=0.5
        )

        U[k] = u

        # One-step integration using solve_ivp.
        sol = solve_ivp(
            fun=lambda time, state: van_der_pol(time, state, mu=mu, u=u),
            t_span=(t[k], t[k] + dt),
            y0=X[k, :],
            t_eval=[t[k] + dt],
            method="RK45"
        )

        X[k + 1, :] = sol.y[:, -1]

        # New noisy measurement.
        Y[k + 1] = X[k + 1, 0] + rng.normal(0.0, noise_std)
        y_history.append(Y[k + 1])

    return t, X, U, Y


def plot_control_results(t, X, U, reference=0.0):
    """
    Plot closed-loop response.
    """
    plt.figure(figsize=(12, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t, X[:, 0], label="Position")
    plt.axhline(reference, color="k", linestyle="--", label="Reference")
    plt.ylabel("Position")
    plt.title("Closed-Loop Control Using Reconstructed State")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, X[:, 1], label="Velocity")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, U, label="Control input")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. Main experiment
# ============================================================

def main():
    """
    Run the full experiment.
    """

    # -----------------------------
    # Experiment settings
    # -----------------------------
    t_final = 40.0
    dt = 0.01
    mu = 1.0
    noise_std = 0.1

    delay_steps = 8
    delay_gap = 5

    test_size = 0.3
    random_seed = 42

    # -----------------------------
    # Simulate system
    # -----------------------------
    print("Simulating Van der Pol oscillator...")
    t, X_true = simulate_vdp(
        t_final=t_final,
        dt=dt,
        initial_state=(2.0, 0.0),
        mu=mu
    )

    # -----------------------------
    # Create noisy partial measurement
    # -----------------------------
    y_clean, y_noisy = create_noisy_measurement(
        X=X_true,
        noise_std=noise_std,
        random_seed=random_seed
    )

    plot_measurement(t, y_clean, y_noisy)

    # -----------------------------
    # Build delay-coordinate dataset
    # -----------------------------
    print("Building delay-coordinate features...")
    Z, X_target = build_delay_coordinates(
        y=y_noisy,
        X_true=X_true,
        delay_steps=delay_steps,
        delay_gap=delay_gap
    )

    print(f"Delay feature shape: {Z.shape}")
    print(f"Target state shape: {X_target.shape}")

    # -----------------------------
    # Train-test split
    # -----------------------------
    Z_train, Z_test, X_train, X_test = train_test_split(
        Z,
        X_target,
        test_size=test_size,
        random_state=random_seed,
        shuffle=False
    )

    # -----------------------------
    # Train Ridge model
    # -----------------------------
    print("Training Ridge model...")
    ridge_model = train_ridge_model(Z_train, X_train)

    X_pred_ridge, metrics_ridge = evaluate_model(
        ridge_model,
        Z_test,
        X_test,
        model_name="Ridge Regression with Delay Coordinates"
    )

    print_metrics(metrics_ridge)
    plot_reconstruction(X_test, X_pred_ridge, model_name="Ridge Regression")
    plot_phase_portrait(X_test, X_pred_ridge, model_name="Ridge Regression")

    # -----------------------------
    # Train Neural Network model
    # -----------------------------
    print("Training neural-network model...")
    mlp_model = train_mlp_model(Z_train, X_train)

    X_pred_mlp, metrics_mlp = evaluate_model(
        mlp_model,
        Z_test,
        X_test,
        model_name="MLP Neural Network with Delay Coordinates"
    )

    print_metrics(metrics_mlp)
    plot_reconstruction(X_test, X_pred_mlp, model_name="MLP Neural Network")
    plot_phase_portrait(X_test, X_pred_mlp, model_name="MLP Neural Network")

    # -----------------------------
    # Optional closed-loop control test
    # -----------------------------
    run_control_test = True

    if run_control_test:
        print("Running simple closed-loop control test...")

        t_ctrl, X_ctrl, U_ctrl, Y_ctrl = simulate_controlled_vdp(
            model=mlp_model,
            delay_steps=delay_steps,
            delay_gap=delay_gap,
            t_final=20.0,
            dt=dt,
            initial_state=(2.0, 0.0),
            mu=mu,
            noise_std=0.05,
            reference=0.0
        )

        plot_control_results(t_ctrl, X_ctrl, U_ctrl, reference=0.0)


if __name__ == "__main__":
    main()