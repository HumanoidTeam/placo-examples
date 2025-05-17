#!/usr/bin/env python3
"""
Keep the robot standing with both feet fixed using PlaCo and Pinocchio.
"""
import pinocchio as pin
import placo
import argparse
import time
import numpy as np
from placo_utils.visualization import robot_viz, frame_viz, contacts_viz
from placo_utils.tf import tf  # Correct import for transformation utilities

# Add PyBullet imports
import pybullet as p
from onshape_to_robot.simulation import Simulation

# Parse arguments for visualization
parser = argparse.ArgumentParser(description="Humanoid standing motion generator.")
parser.add_argument('-p', '--pybullet', action='store_true', help='Enable PyBullet simulation')
parser.add_argument('-m', '--meshcat', action='store_true', help='Enable MeshCat visualization')
args = parser.parse_args()

# Simulation parameters
DT = 0.005  # Timestep [s]

# Load robot
# model_filename = "../models/a1_alpha_biped_concept_urdf/urdf/alpha.urdf"
model_filename = "../models/a1_alpha_biped_corrected_mass/urdf/A1_alpha_biped_concept_urdf.urdf"
# model_filename = "../models/sigmaban/robot.urdf"
# model_filename = "../models/dummyBipedURDF/urdf/250226_DummyBipedURDF.urdf"
robot = placo.HumanoidRobot(model_filename, placo.Flags.ignore_collisions)

# Set gravity explicitly
robot.set_gravity(np.array([0.0, 0.0, -9.81]))

# Kinematics solver for initial posture
# Add these lines to grab the Pinocchio model and create a fresh Data container
model = robot.model  # Pinocchio Model (no parentheses)
data = pin.Data(model)  # Pinocchio Data container

# Walk parameters - if double_support_ratio is not set to 0, should be greater than replan_frequency
parameters = placo.HumanoidParameters()

# Timing parameters
parameters.single_support_duration = 0.6  # Duration of single support phase [s]
parameters.single_support_timesteps = 10  # Number of planning timesteps per single support phase
parameters.double_support_ratio = 0.3  # Ratio of double support (0.0 to 1.0)
parameters.startend_double_support_ratio = 1.5  # Ratio duration of supports for starting and stopping walk
parameters.planned_timesteps = 60  # Number of timesteps planned ahead
parameters.replan_timesteps = 12  # Replanning each n timesteps

# Posture parameters

parameters.walk_com_height = 0.95 #robot.com_world().copy()[2]  # Constant height for the CoM [m]
parameters.walk_foot_height = 0.04  # Height of foot rising while walking [m]
parameters.walk_trunk_pitch = 0.05  # Trunk pitch angle [rad]
parameters.walk_foot_rise_ratio = 0.2  # Time ratio for the foot swing plateau (0.0 to 1.0)

# Feet parameters
parameters.foot_length = 0.1576  # Foot length [m]
parameters.foot_width = 0.092  # Foot width [m]
parameters.feet_spacing = 0.35  # Lateral feet spacing [m]
parameters.zmp_margin = 0.02  # ZMP margin [m]
parameters.foot_zmp_target_x = 0.0  # Reference target ZMP position in the foot [m]
parameters.foot_zmp_target_y = 0.0  # Reference target ZMP position in the foot [m]

# Limit parameters
parameters.walk_max_dtheta = 1  # Maximum dtheta per step [rad]
parameters.walk_max_dy = 0.04  # Maximum dy per step [m]
parameters.walk_max_dx_forward = 0.08  # Maximum dx per step forward [m]
parameters.walk_max_dx_backward = 0.03  # Maximum dx per step backward [m]

# Creating the kinematics solver
solver_kin = placo.KinematicsSolver(robot)
solver_kin.enable_velocity_limits(True)
solver_kin.dt = DT

# Creating the walk QP tasks
tasks = placo.WalkTasks()
tasks.initialize_tasks(solver_kin, robot)

tasks.reach_initial_pose(
    np.eye(4),
    parameters.feet_spacing,
    parameters.walk_com_height,
    parameters.walk_trunk_pitch,
)

robot.update_kinematics()



# Dynamic Solver after bended-knee posture
solver = placo.DynamicsSolver(robot)
solver.dt = 0.005

# Visualization setup   
if args.pybullet:
    # Loading the PyBullet simulation
    import pybullet as p
    from onshape_to_robot.simulation import Simulation
    
    sim = Simulation(model_filename, realTime=True, dt=DT)
    # p.getNumJoints()
    actual_torque_log = []  # Log for PyBullet-applied torques
    # joint_indices = sim.getJoints()
    # print("Number of joints:", p.getNumJoints(1))
    # print("Joint indices:", joint_indices, "Size of joint_indices:", len(joint_indices))
elif args.meshcat:
    # Starting Meshcat viewer
    viz = robot_viz(robot)
else:
    print("No visualization selected, use either -p or -m")
    exit()

# ---- FRAME TASKS FOR FEET ----
# Freeze both feet: add frame tasks for detected frames
# T_left = placo.flatten_on_floor(robot.get_T_world_frame("left_foot"))
# left_frame = solver.add_frame_task("left_foot", T_left)
# left_frame.configure("left_foot", "hard", 1e3, 1e3)

# T_right = placo.flatten_on_floor(robot.get_T_world_frame("right_foot"))
# right_frame = solver.add_frame_task("right_foot", T_right)
# right_frame.configure("right_foot", "hard", 1e3, 1e3)

T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
left_frame = solver.add_frame_task("left_foot", T_world_left)
left_frame.configure("left_foot", "hard", 1e3, 1e3)
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
right_frame = solver.add_frame_task("right_foot", T_world_right)
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
# target_rotation[0, 0] = 0  # Free rotation around x-axis
# target_rotation[1, 1] = 0  # Free rotation around y-axis

base_orientation = solver.add_orientation_task("Base_link", target_rotation)  # Custom target rotation
base_orientation.configure("Base_link", "soft", 1.0)  # Constrain only z-axis rotation

# ---- REGULARIZATION TASK ----
# Add a regularization task for posture
# posture_regularization_task = solver.add_joints_task()
# posture_regularization_task.set_joints({joint: 0.0 for joint in robot.joint_names()})
# posture_regularization_task.configure("posture", "soft", 1e-6)

external_wrench_trunk  = solver.add_external_wrench_contact("trunk", "world")
external_wrench_trunk.w_ext  = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])

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

# Initialize force logs
left_foot_force_log = []
right_foot_force_log = []

# # Main loop: First reach the initial posture for 5 seconds
# print("Reaching initial posture for 5 seconds...")
# try:
#     t = 0  # Time variable
#     while t < 5.0:  # Run for 0.25 seconds
#         # Solve kinematics to reach the initial pose
#         solver_kin.solve(True)
#         robot.update_kinematics()

#         if args.pybullet:
#             if t < 2:
#                 T_left_origin = sim.transformation("origin", "left_foot_frame")
#                 T_world_left = sim.poseToMatrix(([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]))
#                 T_world_origin = T_world_left @ T_left_origin

#                 sim.setRobotPose(*sim.matrixToPose(T_world_origin))

#             joints = {joint: robot.get_joint(joint) for joint in sim.getJoints()}
#             applied = sim.setJoints(joints)
#             sim.tick()
#         # Visualization
#         elif viz:
#             viz.display(robot.state.q)
#             frame_viz("left_foot_frame", left_frame.T_world_frame)
#             frame_viz("right_foot_frame", right_frame.T_world_frame)

#         # Maintain real time
#         time.sleep(DT)
#         t += DT  # Increment time
# except KeyboardInterrupt:
#     print("Exiting.")

# Main loop: Start squatting motion
print("Starting squatting motion. Press Ctrl+C to exit.")
try:
    t = 0  # Reset time variable for squatting motion
    while True:
        # Update CoM task target with sinusoidal motion
        z_offset = -0.10 * (1 - np.cos(2 * np.pi * 1.0 * t)) / 2
        y_offset = -0.0 * (1 - np.cos(2 * np.pi * 1.0 * t)) / 2
        com_task.target_world = com_init + np.array([-0.01, y_offset, z_offset])  # Update CoM position
        external_wrench_trunk.w_ext = np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
        # Solve dynamics
        result = solver.solve(True)

        # Log time and torques
        time_log.append(t)
        torque_log.append(result.tau.copy())

        # Log external forces on the feet
        left_foot_force_log.append(left_contact.wrench[:3].copy())  # Extract force (first 3 components of wrench)
        right_foot_force_log.append(right_contact.wrench[:3].copy())  # Extract force (first 3 components of wrench)

        # Update kinematics
        robot.update_kinematics()

        # Update PyBullet simulation if enabled
        if args.pybullet:
            if t < 2:
                T_left_origin = sim.transformation("origin", "left_foot_frame")
                T_world_left = sim.poseToMatrix(([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]))
                T_world_origin = T_world_left @ T_left_origin

                sim.setRobotPose(*sim.matrixToPose(T_world_origin))

            joints = {joint: robot.get_joint(joint) for joint in sim.getJoints()}
            applied = sim.setJoints(joints)
            sim.tick()

            # Get all movable joint indices
            num_joints = p.getNumJoints(1)  # Assuming robot_id is 1
            joint_indices = [
                i for i in range(num_joints) if p.getJointInfo(1, i)[2] != p.JOINT_FIXED
            ]

            # Get joint states and extract torques
            joint_states = p.getJointStates(1, joint_indices)
            applied_torques = [state[3] for state in joint_states]
            actual_torque_log.append(applied_torques)

        # Visualization
        elif viz:
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


if args.pybullet:
    import csv
    pybullet_output_file = "/home/sasa/Software/hmnd-robot/pybullet_squat_torque_data.csv"
    min_length = min(len(time_log), len(actual_torque_log))  # Ensure synchronized lengths
    with open(pybullet_output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time"] + [f"joint_{i}" for i in range(len(actual_torque_log[0]))])  # Header row
        for i in range(min_length):
            writer.writerow([time_log[i]] + actual_torque_log[i])
    print(f"PyBullet torque data saved to {pybullet_output_file}")

     # Plot PyBullet joint torques after the loop exits
    import matplotlib.pyplot as plt
    import numpy as np

    T = np.array(time_log[:min_length])  # Use synchronized time_log
    PyBullet_Tau = np.array(actual_torque_log[:min_length]).T  # shape: (n_joints, len(T))

    # Plot all PyBullet joint torques
    plt.figure()
    joint_names = list(robot.joint_names())  # Get joint names from the robot model
    for i, name in enumerate(joint_names[:PyBullet_Tau.shape[0]]):  # Use joint names for legend
        plt.plot(T, PyBullet_Tau[i], label=name)
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.legend(loc="upper right", ncol=2, fontsize="small")
    plt.title("PyBullet Joint Torques during Motion")
    plt.show()

if args.meshcat:
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

    # Plot external forces after the loop exits
    left_foot_forces = np.array(left_foot_force_log)  # Convert to numpy array
    right_foot_forces = np.array(right_foot_force_log)  # Convert to numpy array

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(T, left_foot_forces[:, 2], label="Left Foot Z-Force")
    plt.plot(T, right_foot_forces[:, 2], label="Right Foot Z-Force")
    plt.xlabel('time [s]')
    plt.ylabel('Force [N]')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Vertical Forces on Feet')

    plt.subplot(2, 1, 2)
    plt.plot(T, left_foot_forces[:, 0], label="Left Foot X-Force")
    plt.plot(T, right_foot_forces[:, 0], label="Right Foot X-Force")
    plt.plot(T, left_foot_forces[:, 1], label="Left Foot Y-Force")
    plt.plot(T, right_foot_forces[:, 1], label="Right Foot Y-Force")
    plt.xlabel('time [s]')
    plt.ylabel('Force [N]')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Horizontal Forces on Feet')

    plt.tight_layout()
    plt.show()
