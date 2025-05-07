#!/usr/bin/env python3
"""
Keep the robot standing with both feet fixed using PlaCo and Pinocchio.
"""
import pinocchio as pin
import placo
import argparse
import time
import numpy as np
from placo_utils.visualization import robot_viz, frame_viz
from placo_utils.tf import tf  # Correct import for transformation utilities

# Parse arguments for visualization
parser = argparse.ArgumentParser(description="Humanoid standing motion generator.")
parser.add_argument('-p', '--pybullet', action='store_true', help='Enable PyBullet simulation')
parser.add_argument('-m', '--meshcat', action='store_true', help='Enable MeshCat visualization')
args = parser.parse_args()

# Simulation parameters
DT = 0.005  # Timestep [s]

# Load robot
model_filename = "../models/a1_alpha_biped_concept_urdf/urdf/alpha.urdf"
# model_filename = "../models/sigmaban/robot.urdf"
# model_filename = "../models/dummyBipedURDF/urdf/250226_DummyBipedURDF.urdf"
# robot = placo.RobotWrapper(model_filename, placo.Flags.ignore_collisions)
robot = placo.HumanoidRobot(model_filename, placo.Flags.ignore_collisions)

solver = placo.DynamicsSolver(robot)
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

# # ---- CoM TASK ----
# # Initialize the CoM task to keep the CoM at its current position
com_init = robot.com_world().copy()  # Get the initial CoM position
com_task = solver.add_com_task(com_init)
com_task.configure("com", "soft", 1.0)

# ---- TORSO ORIENTATION TASK ----
# Add a task to keep the torso upright by maintaining its orientation
# torso_orientation = solver.add_orientation_task("trunk", np.eye(3))  # Goal is identity rotation (upright)
# torso_orientation.configure("trunk", "soft", 1.0)

# Define a target rotation matrix that allows free rotation around x and y axes
target_rotation = np.eye(3)  # Identity matrix for upright orientation
target_rotation[0, 0] = 0  # Free rotation around x-axis
target_rotation[1, 1] = 0  # Free rotation around y-axis

# base_orientation = solver.add_orientation_task("Base_link", target_rotation)  # Custom target rotation
# base_orientation.configure("Base_link", "soft", 1.0)  # Constrain only z-axis rotation

# ---- REGULARIZATION TASK ----
# Add a regularization task for posture
# posture_regularization_task = solver.add_joints_task()
# posture_regularization_task.set_joints({joint: 0.0 for joint in robot.joint_names()})
# posture_regularization_task.configure("posture", "soft", 1e-6)

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

# Visualization setup
viz = robot_viz(robot) if args.meshcat else None
# Initialize torque and time logs
torque_log = []
time_log = []
# Main loop: Visualize the robot
print("Visualizing the robot. Press Ctrl+C to exit.")
try:
    t = 0  # Time variable for sinusoidal motion
    while True:
        # Update CoM task target with sinusoidal motion
        # z_offset = -0.05 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2  # 5 cm downward amplitude, 0.5 Hz frequency
        # com_task.target_world = com_init + np.array([0.0, 0.0, z_offset])  # Update CoM position
        # Solve QP
        # Solve dynamics

        z_offset = -0.20 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2  # 10 cm downward amplitude, 0.5 Hz frequency
        com_task.target_world = com_init + np.array([0.0, 0.0, z_offset])  # Update CoM position

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
