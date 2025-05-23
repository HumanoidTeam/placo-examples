#!/usr/bin/env python3
"""
Keep the robot standing with both feet fixed using PlaCo and Pinocchio.
"""
import pinocchio as pin
import placo
import argparse
import time
import numpy as np
from placo_utils.visualization import robot_viz, frame_viz,contacts_viz
from placo_utils.tf import tf  # Correct import for transformation utilities

# Parse arguments for visualization
parser = argparse.ArgumentParser(description="Humanoid standing motion generator.")
parser.add_argument('-p', '--pybullet', action='store_true', help='Enable PyBullet simulation')
parser.add_argument('-m', '--meshcat', action='store_true', help='Enable MeshCat visualization')
args = parser.parse_args()

# Simulation parameters
DT = 0.005  # Timestep [s]

# Load robot
model_filename = "../models/sigmaban/robot.urdf"
robot = placo.HumanoidRobot(model_filename)
# solver = placo.KinematicsSolver(robot)
# robot.update_kinematics()
solver = placo.DynamicsSolver(robot)
# Setting the solver delta time to 1ms
solver.dt = 0.005

# Visualization setup
viz = robot_viz(robot) if args.meshcat else None

# ---- FRAME TASKS FOR FEET ----
# Freeze both feet: add frame tasks for detected frames
T_left = placo.flatten_on_floor(robot.get_T_world_frame("left_foot"))
left_frame = solver.add_frame_task("left_foot", T_left)
left_frame.configure("left_foot", "hard", 1e3, 1e3)

T_right = placo.flatten_on_floor(robot.get_T_world_frame("right_foot"))
right_frame = solver.add_frame_task("right_foot", T_right)
right_frame.configure("right_foot", "hard", 1e3, 1e3)

# ---- CONTACTS FOR FEET ----
# Add planar contacts for both feet
left_contact = solver.add_planar_contact(left_frame)
left_contact.length = 0.15
left_contact.width = 0.09
left_contact.weight_moments = 1e-3

right_contact = solver.add_planar_contact(right_frame)
right_contact.length = 0.15
right_contact.width = 0.09
right_contact.weight_moments = 1e-3

# ---- CoM TASK ----
# Initialize the CoM task to keep the CoM at its current position
com_init = robot.com_world().copy()  # Get the initial CoM position
com_task = solver.add_com_task(com_init)
com_task.configure("com", "soft", 1.0)

# ---- TORSO ORIENTATION TASK ----
# Add a task to keep the torso upright by maintaining its orientation
torso_orientation = solver.add_orientation_task("torso_2023", np.eye(3))  # Goal is identity rotation (upright)
torso_orientation.configure("torso_orientation", "soft", 1.0)

# ---- REGULARIZATION TASK ----
# Add a regularization task for posture
posture_regularization_task = solver.add_joints_task()
posture_regularization_task.set_joints({joint: 0.0 for joint in robot.joint_names()})
posture_regularization_task.configure("posture", "soft", 1e-6)

external_wrench_left  = solver.add_external_wrench_contact("radius_v2", "world")
external_wrench_right = solver.add_external_wrench_contact("radius_v2_2", "world")
external_wrench_left.w_ext  = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
external_wrench_right.w_ext = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])

# ---- FRAME TASKS FOR HANDS ----
# Use the correct hand end-effector links for the hands
# Add orientation task for the left hand to maintain its current orientation
# T_left_hand = robot.get_T_world_frame("radius_v2")  # Get initial transform of the left hand end-effector
# left_hand_orientation = solver.add_orientation_task("radius_v2", T_left_hand[:3, :3])  # Keep current orientation
# left_hand_orientation.configure("left_hand_orientation", "soft", 1.0)

# Add a frame task for the right hand to keep it in front of the robot
# T_right_hand = robot.get_T_world_frame("radius_v2_2")  # Get initial transform of the right hand end-effector
# right_hand_frame = solver.add_frame_task("radius_v2_2", T_right_hand)
# right_hand_frame.configure("right_hand", "soft", 1.0)

# # Set the target position for the right hand to stay in front of the robot
# right_hand_target = T_right_hand.copy()
# right_hand_target[:3, 3] += np.array([0.3, 0.0, 0.0])  # Move 30 cm forward

# Enable joint, velocity, and torque limits
solver.enable_joint_limits(True)
solver.enable_velocity_limits(True)
solver.enable_torque_limits(True)

# Initializing the robot with a puppet contact
puppet = solver.add_puppet_contact()
for k in range(1000):
    robot.add_q_noise(1e-3)  # adding noise to exit singularities
    solver.solve(True)
    robot.update_kinematics()
solver.remove_contact(puppet)

# Initialize torque and time logs
torque_log = []
time_log = []

# Main loop: Keep the robot standing with sinusoidal CoM motion
print("Robot is standing with feet fixed and CoM moving sinusoidally. Press Ctrl+C to exit.")
try:
    t = 0  # Time variable for sinusoidal motion
    while True:
        # Update CoM task target with sinusoidal motion
        z_offset = -0.10 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2  # 10 cm downward amplitude, 0.5 Hz frequency
        com_task.target_world = com_init + np.array([0.0, 0.0, z_offset])  # Update CoM position

        # Update torso orientation to remain upright
        torso_orientation.R_world_frame = np.eye(3)
        external_wrench_left.w_ext = np.array([0.0, 0.0, -40.0, 0.0, 0.0, 0.0])
        external_wrench_right.w_ext = np.array([0.0, 0.0, -40.0, 0.0, 0.0, 0.0])
        # Solve dynamics
        result = solver.solve(True)

        # Log time and torques
        time_log.append(t)
        torque_log.append(result.tau.copy())

        # Update kinematics
        robot.update_kinematics()

        # Visualization
        if viz:
            viz.display(robot.state.q)
            frame_viz("left_foot_frame", left_frame.T_world_frame)
            frame_viz("right_foot_frame", right_frame.T_world_frame)
            frame_viz("com_frame", tf.translation_matrix(com_task.target_world))  # Convert CoM to 4x4 matrix

            # Visualize forces under the feet using contact forces from the solver
            contacts_viz(solver, ratio=3e-3, radius=0.01)

        # Maintain real time
        time.sleep(DT)
        t += DT  # Increment time
except KeyboardInterrupt:
    print("Exiting.")

# Plot joint torques after the loop exits
import matplotlib.pyplot as plt
import numpy as np

T = np.array(time_log)
Tau = np.stack(torque_log, axis=1)  # shape: (n_joints, len(T))

plt.figure()
for i, name in enumerate(robot.joint_names()):
    plt.plot(T, Tau[i], label=name)
plt.xlabel('time [s]')
plt.ylabel('torque [Nm]')
plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.title('Joint torques during squatting motion')
plt.show()
