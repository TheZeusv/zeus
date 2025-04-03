import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from qpsolvers import solve_qp
import support_files_car as sfc_g
import matplotlib.animation as animation
import time


np.set_printoptions(suppress=True)


def init_simulation():
    support = sfc_g.SupportFilesCar()
    constants = support.constants
    Ts, hz, outputs, inputs = constants['Ts'], constants['hz'], constants['outputs'], constants['inputs']
    x_lim, y_lim = constants['x_lim'], constants['y_lim']

    # Reference Trajectory
    x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, t, lap_times = support.trajectory_generator()
    sim_length = len(t)


    refSignals = np.ravel(np.column_stack((x_dot_ref, psi_ref, X_ref, Y_ref)))
    states = np.array([x_dot_ref[0], y_dot_ref[0], psi_ref[0], 0., X_ref[0], Y_ref[0]])
    statesTotal = np.zeros((sim_length, len(states)))
    statesTotal[0] = states

    # Initial control inputs
    U1, U2 = 0, 0
    UTotal = np.zeros((sim_length, 2))
    UTotal[0] = [U1, U2]
    du = np.zeros((inputs * hz, 1))
    accelerations_total = np.zeros((sim_length, 3))

    return support, constants, Ts, hz, outputs, inputs, x_lim, y_lim, t, sim_length, refSignals, \
       states, statesTotal, U1, U2, UTotal, du, accelerations_total, \
       x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, lap_times

def run_mpc_loop(support, constants, t, sim_length, refSignals,
                 states, statesTotal, U1, U2, UTotal, du,
                 accelerations_total):
    start_time = time.time()  

    hz = constants['hz']
    outputs = constants['outputs']

    t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani = [], [], [], [], [], []
    k = 0
    qp_fail_count = 0
    MAX_FAILS = 50

    for i in range(sim_length - 1):
        # Prevent division by zero
        if abs(states[0]) < 0.01:
            states[0] = 0.01  

        Ad, Bd, Cd, Dd = support.state_space(states, U1, U2)
        x_aug = np.concatenate((states, [U1, U2]))[:, None]

        # Extract reference for the prediction horizon
        k += outputs
        r = np.zeros(outputs * hz)
        if k + outputs * hz <= len(refSignals):
            r[:] = refSignals[k:k + outputs * hz]
        else:
            remaining = len(refSignals) - k
            r[:remaining] = refSignals[k:]

        # Build MPC QP matrices
        Hdb, Fdbt, Cdb, Adc, G, ht = support.mpc_simplification(Ad, Bd, Cd, Dd, hz, x_aug, du, i)
        ft = np.dot(np.concatenate((x_aug.ravel(), r)), Fdbt)

        # Solve the QP problem to find optimal control increments
        try:
            du_raw = solve_qp(Hdb, ft, G, ht, solver="cvxopt")
            if du_raw is None:
                qp_fail_count += 1  
                print(f"[WARNING] QP failed at step {i} ({qp_fail_count}/{MAX_FAILS}), setting du to zeros")
                du = np.zeros((2 * hz, 1))
            else:
                du = np.array(du_raw).reshape(-1, 1)

        except ValueError:
            qp_fail_count += 1
            print(f"[ERROR] QP solver raised an exception at step {i} ({qp_fail_count}/{MAX_FAILS})")
            du = np.zeros((2 * hz, 1))

        if qp_fail_count > MAX_FAILS:
            print("[FATAL] Too many QP failures. Stopping simulation.")
            break


        U1 += du[0, 0]
        U2 += du[1, 0]
        UTotal[i + 1] = [U1, U2]

        states, x_dd, y_dd, psi_dd = support.open_loop_new_states(states, U1, U2)

        statesTotal[i + 1] = states
        accelerations_total[i + 1] = [x_dd, y_dd, psi_dd]

        # Store values for animation
        if i % 5 == 1:
            t_ani.append(t[i])
            x_dot_ani.append(states[0])
            psi_ani.append(states[2])
            X_ani.append(states[4])
            Y_ani.append(states[5])
            delta_ani.append(U1)

        if i % 500 == 0:
            print(f"Progress: {100 * i / sim_length:.2f}%")

    end_time = time.time()  
    print(f"Execution time (MPC): {end_time - start_time:.2f} seconds")

    return statesTotal, UTotal, accelerations_total, t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani

# Animation function
def animate_vehicle(t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani, constants):
    lf, lr = constants['lf'], constants['lr']
    frame_amount = len(X_ani)

    fig = plt.figure(figsize=(14, 7))
    ax_main = fig.add_subplot(1, 2, 1)
    ax_zoom = fig.add_subplot(1, 2, 2)

    # Track
    ax_main.set_xlim(-50, constants['x_lim'])
    ax_main.set_ylim(-30, constants['y_lim'])
    ax_main.set_xlabel("X [m]")
    ax_main.set_ylabel("Y [m]")
    ax_main.set_title("Global View")
    ax_main.grid(True)
    ref_traj, = ax_main.plot(X_ani, Y_ani, '--', label='Reference trajectory')
    car_path, = ax_main.plot([], [], 'r-', label='Vehicle path')
    car_body_main, = ax_main.plot([], [], 'k-', linewidth=2, label='Car body')
    ax_main.legend()

    # Car View
    ax_zoom.set_xlim(-5, 5)
    ax_zoom.set_ylim(-4, 4)
    ax_zoom.set_aspect('equal')
    ax_zoom.set_title("Vehicle State (Zoom)")
    ax_zoom.grid(True)
    car_body_zoom, = ax_zoom.plot([], [], 'k-', linewidth=3)
    front_arrow, = ax_zoom.plot([], [], 'r--', linewidth=1.5)
    back_wheel, = ax_zoom.plot([], [], 'g-', linewidth=4)
    front_wheel, = ax_zoom.plot([], [], 'b-', linewidth=4)

    # Text
    text_speed = ax_zoom.text(3, 3.0, '', fontsize=10, color='blue')
    text_delta = ax_zoom.text(3, 2.2, '', fontsize=10, color='red')
    text_yaw = ax_zoom.text(3, 1.4, '', fontsize=10, color='black')

    def update(num):
        x = X_ani[num]
        y = Y_ani[num]
        psi = psi_ani[num]
        delta = delta_ani[num]
        v = x_dot_ani[num]

        rear_x = -lr * np.cos(psi)
        rear_y = -lr * np.sin(psi)
        front_x = lf * np.cos(psi)
        front_y = lf * np.sin(psi)

        car_body_zoom.set_data([rear_x, front_x], [rear_y, front_y])
        car_body_main.set_data([x - lr * np.cos(psi), x + lf * np.cos(psi)],
                               [y - lr * np.sin(psi), y + lf * np.sin(psi)])
        front_arrow.set_data([0, (lf + 1.5) * np.cos(psi)], [0, (lf + 1.5) * np.sin(psi)])

        dx = 0.25 * np.cos(psi)
        dy = 0.25 * np.sin(psi)
        back_wheel.set_data([rear_x - dx, rear_x + dx], [rear_y - dy, rear_y + dy])

        wheel_angle = psi + delta
        dx_fw = 0.25 * np.cos(wheel_angle)
        dy_fw = 0.25 * np.sin(wheel_angle)
        front_wheel.set_data([front_x - dx_fw, front_x + dx_fw], [front_y - dy_fw, front_y + dy_fw])

        car_path.set_data(X_ani[:num], Y_ani[:num])

        text_speed.set_text(f"Speed: {v:.2f} m/s")
        text_delta.set_text(f"Steering: {delta:.2f} rad")
        text_yaw.set_text(f"Yaw: {psi:.2f} rad")

        return car_path, car_body_main, car_body_zoom, front_arrow, back_wheel, front_wheel, text_speed, text_delta, text_yaw

    global ani
    ani = animation.FuncAnimation(fig, update, frames=frame_amount, interval=50, blit=False)

    plt.tight_layout()
    plt.show()




def plot_results(t, x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, statesTotal, UTotal, accelerations_total):
    # Trajectory plot
    plt.figure()
    plt.plot(X_ref, Y_ref, '--b', label='Reference')
    plt.plot(statesTotal[:, 4], statesTotal[:, 5], 'r', label='Actual')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True)
    plt.legend()
    plt.title("Vehicle Trajectory")
    plt.show()

    # Inputs
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, UTotal[:, 0], label='Steering [rad]')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t, UTotal[:, 1], label='Acceleration [m/s²]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Yaw, X, Y
    labels = ['Yaw [rad]', 'X [m]', 'Y [m]']
    refs = [psi_ref, X_ref, Y_ref]
    idx = [2, 4, 5]
    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, refs[i], '--b', label=f'{labels[i]} ref')
        plt.plot(t, statesTotal[:, idx[i]], 'r', label='Actual')
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Velocities and accelerations
    plt.figure()
    for i, label in enumerate(['x_dot [m/s]', 'y_dot [m/s]', 'psi_dot [rad/s]']):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, statesTotal[:, i], 'r', label=label)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for i, label in enumerate(['x_dd [m/s²]', 'y_dd [m/s²]', 'psi_dd [rad/s²]']):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, accelerations_total[:, i], label=label)
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    support, constants, Ts, hz, outputs, inputs, x_lim, y_lim, t, sim_length, refSignals, \
    states, statesTotal, U1, U2, UTotal, du, accelerations_total, \
    x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, lap_times = init_simulation() 

    print("Lap Times:", lap_times)

    # Run the MPC simulation loop
    statesTotal, UTotal, accelerations_total, t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani = \
        run_mpc_loop(support, constants, t, sim_length, refSignals,
                     states, statesTotal, U1, U2, UTotal, du, accelerations_total)

    # Visualize and plot results
    animate_vehicle(t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani, constants)
    plot_results(t, x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, statesTotal, UTotal, accelerations_total)