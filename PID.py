import numpy as np
import matplotlib.pyplot as plt
from support_files_car import SupportFilesCar
import matplotlib.animation as animation

# PID controller class
class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_output, self.max_output = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measurement):
        # Calculates de PID control base of the error
        error = setpoint - measurement
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Limit output
        if self.max_output is not None:
            output = min(output, self.max_output)
        if self.min_output is not None:
            output = max(output, self.min_output)

        self.prev_error = error
        return output

#Visualization
def animate_vehicle(t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani, constants, X_ref, Y_ref):
    lf, lr = constants['lf'], constants['lr']
    frame_amount = len(X_ani)

    fig = plt.figure(figsize=(14, 7))
    ax_main = fig.add_subplot(1, 2, 1)
    ref_traj, = ax_main.plot(X_ref, Y_ref, '--b', label='Reference trajectory')
    ax_zoom = fig.add_subplot(1, 2, 2)

    ax_main.set_xlim(-50, constants['x_lim'])
    ax_main.set_ylim(-30, constants['y_lim'])
    ax_main.set_xlabel("X [m]")
    ax_main.set_ylabel("Y [m]")
    ax_main.set_title("Global View")
    ax_main.grid(True)



    car_path, = ax_main.plot([], [], 'r-', label='Vehicle path')
    car_body_main, = ax_main.plot([], [], 'k-', linewidth=2, label='Car body')
    ax_main.legend()

    ax_zoom.set_xlim(-5, 5)
    ax_zoom.set_ylim(-4, 4)
    ax_zoom.set_aspect('equal')
    ax_zoom.set_title("Vehicle State (Zoom)")
    ax_zoom.grid(True)
    car_body_zoom, = ax_zoom.plot([], [], 'k-', linewidth=3)
    front_arrow, = ax_zoom.plot([], [], 'r--', linewidth=1.5)
    back_wheel, = ax_zoom.plot([], [], 'g-', linewidth=4)
    front_wheel, = ax_zoom.plot([], [], 'b-', linewidth=4)

    text_speed = ax_zoom.text(3, 3.0, '', fontsize=10, color='blue')
    text_delta = ax_zoom.text(3, 2.2, '', fontsize=10, color='red')
    text_yaw = ax_zoom.text(3, 1.4, '', fontsize=10, color='black')

    # Updates the car position in every frame
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

    ani = animation.FuncAnimation(fig, update, frames=frame_amount, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

# Iniciate the Simulation
def init_simulation():
    support = SupportFilesCar()
    constants = support.constants
    Ts = support.constants['Ts']

    x_dot_ref, y_dot_ref, psi_ref, X_ref, Y_ref, t, lap_times = support.trajectory_generator()
    sim_length = len(t)

    states = np.array([x_dot_ref[0], y_dot_ref[0], psi_ref[0], 0., X_ref[0], Y_ref[0]])
    statesTotal = np.zeros((sim_length, len(states)))
    statesTotal[0] = states

    UTotal = np.zeros((sim_length, 2))
    accelerations_total = np.zeros((sim_length, 3))

    return support, constants, Ts, t, sim_length, x_dot_ref, psi_ref, X_ref, Y_ref, states, statesTotal, UTotal, accelerations_total, lap_times

# Main Loop
def run_pid_loop(support, constants, Ts, t, sim_length, x_dot_ref, psi_ref, X_ref, Y_ref,
                 states, statesTotal, UTotal, accelerations_total):

    # PID parameters (tune if needed)
    delta = 0.0  
    speed_pid = PIDController(kp=2.0, ki=0.1, kd=0.5, dt=Ts, output_limits=(-5, 5))   # Acceleration PID
    yaw_pid = PIDController(kp=3.0, ki=0.2, kd=0.8, dt=Ts, output_limits=(-np.pi/4, np.pi/4))  # Steering PID
    t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani = [], [], [], [], [], []

    USE_PID_CONSTANT_SPEED = False       
    PID_CONSTANT_SPEED_VALUE = 30.0 

    for i in range(sim_length - 1):
        ref_speed = PID_CONSTANT_SPEED_VALUE if USE_PID_CONSTANT_SPEED else x_dot_ref[i]
        ref_yaw = psi_ref[i]

        current_speed = states[0]
        current_yaw = states[2]

        if i % 5 == 1:
            t_ani.append(t[i])
            x_dot_ani.append(states[0])
            psi_ani.append(states[2])
            X_ani.append(states[4])
            Y_ani.append(states[5])
            delta_ani.append(delta)

        # PID control
        a = speed_pid.compute(ref_speed, current_speed)
        delta = yaw_pid.compute(ref_yaw, current_yaw)

        # Save control inputs
        UTotal[i + 1] = [delta, a]

        # Simulate next state
        states, x_dd, y_dd, psi_dd = support.open_loop_new_states(states, delta, a)
        statesTotal[i + 1] = states
        accelerations_total[i + 1] = [x_dd, y_dd, psi_dd]

        if i % 500 == 0:
            print(f"Progress: {100 * i / sim_length:.2f}%")

    return statesTotal, UTotal, accelerations_total, t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani

def plot_results(t, x_dot_ref, psi_ref, X_ref, Y_ref, statesTotal, UTotal, accelerations_total):
    plt.figure()
    plt.plot(X_ref, Y_ref, '--b', label='Reference')
    plt.plot(statesTotal[:, 4], statesTotal[:, 5], 'r', label='Actual')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Vehicle Trajectory vs Reference')
    plt.legend()
    plt.grid(True)
    plt.show()

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

    labels = ['Yaw angle (rad)', 'X [m]', 'Y [m]']
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
    support, constants, Ts, t, sim_length, x_dot_ref, psi_ref, X_ref, Y_ref, \
    states, statesTotal, UTotal, accelerations_total, lap_times = init_simulation()

    print("Lap Times:", lap_times)

    statesTotal, UTotal, accelerations_total, t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani = \
        run_pid_loop(support, constants, Ts, t, sim_length,
                     x_dot_ref, psi_ref, X_ref, Y_ref,
                     states, statesTotal, UTotal, accelerations_total)

    animate_vehicle(t_ani, x_dot_ani, psi_ani, X_ani, Y_ani, delta_ani, constants, X_ref, Y_ref)

    plot_results(t, x_dot_ref, psi_ref, X_ref, Y_ref, statesTotal, UTotal, accelerations_total)
