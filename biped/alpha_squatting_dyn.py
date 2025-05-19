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
parser.add_argument('--external_force', type=float, default=0.0, help='External force applied to the trunk in N')
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
parameters.walk_trunk_pitch = 0.0  # Trunk pitch angle [rad]
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
    # grab the body (robot) unique id  
    robot_id = sim.robot

    # now disable every self‐collision pair, using the default client  
    num_joints = p.getNumJoints(robot_id)
    for i in range(-1, num_joints):
        for j in range(i+1, num_joints):
            p.setCollisionFilterPair(
                bodyUniqueIdA=robot_id,
                bodyUniqueIdB=robot_id,
                linkIndexA=i,
                linkIndexB=j,
                enableCollision=False
            )
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
torso_orientation = solver.add_orientation_task("trunk", np.eye(3))  # Goal is identity rotation (upright)
torso_orientation.configure("trunk", "soft", 1.0)

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
external_wrench_trunk.w_ext  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
placo_left_foot_force_log = []  # For PlaCo 3D forces
placo_right_foot_force_log = []  # For PlaCo 3D forces

# Main loop: Start squatting motion
print("Starting squatting motion. Press Ctrl+C to exit.")

landed = False
squat_start_time = None
squat_delay = 5.0         # how long to wait after touchdown
force_threshold = 20.0    # N, tune this to your contact model
t = 0.0

external_force = args.external_force  # Get the external force from the command-line argument

try:
    while True:
        # --- 1) Detect landing ---
        if not landed:
            if args.meshcat:
                # Skip force sensing and proceed immediately for MeshCat
                landed = True
                squat_start_time = t
                print(f"[Squat] proceeding immediately with MeshCat at t = {squat_start_time:.2f}s")
            else:
                # Use actual contact forces for PyBullet
                fL = left_contact.wrench[2]
                fR = right_contact.wrench[2]
                if fL > force_threshold and fR > force_threshold:
                    landed = True
                    squat_start_time = t
                    print(f"[Squat] touchdown detected at t = {squat_start_time:.2f}s")

        # --- 2) Compute CoM target ---
        if (not landed) or (t - squat_start_time < squat_delay):
            # still hanging or in post‐touchdown hang
            com_task.target_world = com_init
        else:
            # now we really start squatting
            t_squat = t - squat_start_time - squat_delay
            z_offset = -0.10 * (1 - np.cos(2 * np.pi * 1.0 * t_squat)) / 2
            com_task.target_world = com_init + np.array([-0.01, 0.0, z_offset])
        if args.meshcat:
            t_squat = t - squat_start_time - squat_delay
            z_offset = -0.10 * (1 - np.cos(2 * np.pi * 1.0 * t_squat)) / 2
            com_task.target_world = com_init + np.array([-0.01, 0.0, z_offset])
        external_wrench_trunk.w_ext = np.array([0.0, 0.0, external_force, 0.0, 0.0, 0.0])

        # Solve dynamics
        result = solver.solve(True)

        # Log time and torques
        time_log.append(t)
        torque_log.append(result.tau.copy())

        # Log external forces on the feet (PlaCo)
        placo_left_foot_force_log.append(left_contact.wrench[:3].copy())  # Extract force (first 3 components of wrench)
        placo_right_foot_force_log.append(right_contact.wrench[:3].copy())  # Extract force (first 3 components of wrench)

        # Update kinematics
        robot.update_kinematics()

        # Update PyBullet simulation if enabled
        if args.pybullet:
            # Get the trunk link index using PyBullet's API
            trunk_link_index = None

            for i in range(p.getNumJoints(robot_id)):
                joint_info = p.getJointInfo(robot_id, i)
                link_name = joint_info[12].decode("utf-8")
                if link_name == "trunk":
                    trunk_link_index = i

            if trunk_link_index is None:
                raise ValueError("Trunk link not found in the robot model.")

            # Apply external force to the trunk in PyBullet
            p.applyExternalForce(
                objectUniqueId=robot_id,
                linkIndex=trunk_link_index,
                forceObj=[0.0, 0.0, external_force],  # Use the parameterized external force
                posObj=[0.0, 0.0, 0.0],  # Apply at the center of mass
                flags=p.WORLD_FRAME
            )

            if t < 2:
                T_left_origin = sim.transformation("origin", "left_foot_frame")
                T_world_left = sim.poseToMatrix(([0.0, 0.0, 0.005], [0.0, 0.0, 0.0, 1.0]))
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

    # Bar plot for average joint torques
    plt.figure(figsize=(12, 6))
    joint_names = list(robot.joint_names())  # Get joint names from the robot model
    avg_torques = [np.mean(PyBullet_Tau[i]) for i in range(PyBullet_Tau.shape[0])]  # Compute average torques
    plt.bar(joint_names, avg_torques, alpha=0.8)
    plt.xlabel("Joint Names")
    plt.ylabel("Average Torque [Nm]")
    plt.title("Average Joint Torques (PyBullet)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Plot all PyBullet joint torques
    plt.figure()
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

    # Plot PlaCo forces after the loop exits
    T = np.array(time_log)
    placo_left_foot_forces = np.array(placo_left_foot_force_log)  # Convert to numpy array
    placo_right_foot_forces = np.array(placo_right_foot_force_log)  # Convert to numpy array

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(T, placo_left_foot_forces[:, 2], label="Left Foot Z-Force (PlaCo)")
    plt.plot(T, placo_right_foot_forces[:, 2], label="Right Foot Z-Force (PlaCo)")
    plt.xlabel('time [s]')
    plt.ylabel('Force [N]')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Vertical Forces on Feet (PlaCo)')

    plt.subplot(2, 1, 2)
    plt.plot(T, placo_left_foot_forces[:, 0], label="Left Foot X-Force (PlaCo)")
    plt.plot(T, placo_right_foot_forces[:, 0], label="Right Foot X-Force (PlaCo)")
    plt.plot(T, placo_left_foot_forces[:, 1], label="Left Foot Y-Force (PlaCo)")
    plt.plot(T, placo_right_foot_forces[:, 1], label="Right Foot Y-Force (PlaCo)")
    plt.xlabel('time [s]')
    plt.ylabel('Force [N]')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Horizontal Forces on Feet (PlaCo)')

    plt.tight_layout()
    plt.show()

if args.meshcat or args.pybullet:
    # Plot contact forces from PlaCo
    T = np.array(time_log)
    left_contact_forces = [force[2] for force in placo_left_foot_force_log]  # Z-force for left foot
    right_contact_forces = [force[2] for force in placo_right_foot_force_log]  # Z-force for right foot

    plt.figure()
    plt.plot(T, left_contact_forces, label="Left Foot Z-Force (PlaCo)")
    plt.plot(T, right_contact_forces, label="Right Foot Z-Force (PlaCo)")
    plt.xlabel("Time [s]")
    plt.ylabel("Z-Force [N]")
    plt.legend(loc="upper right", fontsize="small")
    plt.title("Vertical Contact Forces on Feet (PlaCo)")
    plt.show()
