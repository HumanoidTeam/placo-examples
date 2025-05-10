import pinocchio as pin  # Import Pinocchio
import time
import placo
import argparse
import numpy as np
import warnings
from placo_utils.visualization import (
    robot_viz,
    frame_viz,
    line_viz,
    footsteps_viz,
)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
parser.add_argument("-m", "--meshcat", action="store_true", help="MeshCat visualization")
args = parser.parse_args()

DT = 0.005
# model_filename = "../models/sigmaban/robot.urdf"
model_filename = "../models/a1_alpha_biped_concept_urdf/urdf/alpha.urdf"
# model_filename = "../models/250226_DummyBipedURDF/urdf/250226_DummyBipedURDF.urdf"

# Loading the robot
robot = placo.HumanoidRobot(model_filename)

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

parameters.walk_com_height = 1.0 #robot.com_world().copy()[2]  # Constant height for the CoM [m]
parameters.walk_foot_height = 0.04  # Height of foot rising while walking [m]
parameters.walk_trunk_pitch = 0.15  # Trunk pitch angle [rad]
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
solver = placo.KinematicsSolver(robot)
solver.enable_velocity_limits(True)
solver.dt = DT

# Creating the walk QP tasks
tasks = placo.WalkTasks()
tasks.initialize_tasks(solver, robot)

# Creating a joint task to assign DoF values for upper body
# elbow = -50 * np.pi / 180
# shoulder_roll = 0 * np.pi / 180
# shoulder_pitch = 20 * np.pi / 180
# joints_task = solver.add_joints_task()
# joints_task.set_joints(
#     {
#         "left_shoulder_roll": shoulder_roll,
#         "left_shoulder_pitch": shoulder_pitch,
#         "left_elbow": elbow,
#         "right_shoulder_roll": -shoulder_roll,
#         "right_shoulder_pitch": shoulder_pitch,
#         "right_elbow": elbow,
#         "head_pitch": 0.0,
#         "head_yaw": 0.0,
#     }
# )
# joints_task.configure("joints", "soft", 1.0)

# Placing the robot in the initial position
# print("Placing the robot in the initial position...")
tasks.reach_initial_pose(
    np.eye(4),
    parameters.feet_spacing,
    parameters.walk_com_height,
    parameters.walk_trunk_pitch,
)
# print("Initial position reached")

print(f"CoM position: {robot.com_world()}")
# Creating the FootstepsPlanner
repetitive_footsteps_planner = placo.FootstepsPlannerRepetitive(parameters)
d_x = 0.1
d_y = 0.0
d_theta = 0.2
nb_steps = 10
repetitive_footsteps_planner.configure(d_x, d_y, d_theta, nb_steps)

# Planning footsteps
T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
footsteps = repetitive_footsteps_planner.plan(placo.HumanoidRobot_Side.left, T_world_left, T_world_right)

supports = placo.FootstepsPlanner.make_supports(footsteps, 0.0, True, parameters.has_double_support(), True)

# Creating the pattern generator and making an initial plan
walk = placo.WalkPatternGenerator(robot, parameters)
trajectory = walk.plan(supports, robot.com_world(), 0.0)

if args.pybullet:
    # Loading the PyBullet simulation
    import pybullet as p
    from onshape_to_robot.simulation import Simulation

    sim = Simulation(model_filename, realTime=True, dt=DT)
elif args.meshcat:
    # Starting Meshcat viewer
    viz = robot_viz(robot)
    footsteps_viz(trajectory.get_supports())
else:
    print("No visualization selected, use either -p or -m")
    exit()

# Initialize logs for joint kinematics and torques
q_log = []
qdot_log = []
qddot_log = []
torque_log = []
time_log = []

# Timestamps
start_t = time.time()
initial_delay = -2.0 if args.pybullet or args.meshcat else 0.0
t = initial_delay
last_display = time.time()
last_replan = 0

# Debugging: Log solver and task configuration
print("Solver and task configuration:")
print(f"Single support duration: {parameters.single_support_duration}")
print(f"Double support ratio: {parameters.double_support_ratio}")
print(f"ZMP margin: {parameters.zmp_margin}")
print(f"Replan timesteps: {parameters.replan_timesteps}")
print(f"Planned timesteps: {parameters.planned_timesteps}")

# Adjust parameters to ensure feasibility
parameters.double_support_ratio = 0.1  # Increase double support ratio slightly
parameters.zmp_margin = 0.03  # Increase ZMP margin for stability

# Debugging: Log initial robot state
print("Initial robot state:")
print(f"q: {robot.state.q}")
print(f"CoM position: {robot.com_world()}")

# Run the loop for a fixed duration (e.g., 10 seconds)
start_time = time.time()
while time.time() - start_time < 10:  # Run for 10 seconds
    # Updating the QP tasks from planned trajectory
    tasks.update_tasks_from_trajectory(trajectory, t)

    try:
        # 1) Solve the kinematic QP and get joint velocities
        qd_sol = solver.solve(True)
    except RuntimeError as e:
        print("QP Solver Error:", e)
        print("Check constraints and initial conditions.")
        exit(1)

    # 2) Advance PlaCo’s internal state
    robot.update_kinematics()

    # 3) Log position & velocity
    q_log.append(robot.state.q.copy())
    qdot_log.append(qd_sol.copy())

    # 4) Once you have two velocity samples, finite‐difference to compute acceleration
    if len(qdot_log) >= 2:
        qdd = (qdot_log[-1] - qdot_log[-2]) / DT
        qddot_log.append(qdd)

        # 5) Compute torques using inverse dynamics via Pinocchio RNEA
        tau = pin.rnea(model, data, q_log[-1], qdot_log[-1], qdd)
        torque_log.append(tau.copy())
        time_log.append(t)

    # Ensuring the robot is kinematically placed on the floor on the proper foot
    if not trajectory.support_is_both(t):
        robot.update_support_side(str(trajectory.support_side(t)))
        robot.ensure_on_floor()

    # If enough time elapsed and we can replan, do the replanning
    if (t - last_replan > parameters.replan_timesteps * parameters.dt() and walk.can_replan_supports(trajectory, t)):
        # Replanning footsteps from current trajectory
        supports = walk.replan_supports(repetitive_footsteps_planner, trajectory, t, last_replan)

        # Replanning CoM trajectory, yielding a new trajectory we can switch to
        new_trajectory = walk.replan(supports, trajectory, t)
        if new_trajectory.t_start < t:
            print("Warning: Invalid replanning time. Skipping this replan.")
        else:
            print("Valid replanning time.")
            print("new_trajectory.t_start: {}, t: {}", new_trajectory.t_start, t)
            trajectory = new_trajectory
            last_replan = t
        # Update trajectory and last_replan **only after replanning succeeds**
        # trajectory = new_trajectory
        # last_replan = t

        if args.meshcat:
            # Drawing footsteps
            footsteps_viz(supports)

            # Drawing planned CoM trajectory on the ground
            coms = [[*trajectory.get_p_world_CoM(t)[:2], 0.0]
                for t in np.linspace(trajectory.t_start, trajectory.t_end, 100)]
            line_viz("CoM_trajectory", np.array(coms), 0xFFAA00)

    # During the warmup phase, the robot is enforced to stay in the initial position
    if args.pybullet:
        if t < -2:
            T_left_origin = sim.transformation("origin", "left_foot_frame")
            T_world_left = sim.poseToMatrix(([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]))
            T_world_origin = T_world_left @ T_left_origin

            sim.setRobotPose(*sim.matrixToPose(T_world_origin))

        joints = {joint: robot.get_joint(joint) for joint in sim.getJoints()}
        applied = sim.setJoints(joints)
        sim.tick()

    # Updating meshcat display periodically
    elif args.meshcat:
        if time.time() - last_display > 0.03:
            last_display = time.time()
            viz.display(robot.state.q)

            frame_viz("left_foot_target", trajectory.get_T_world_left(t))
            frame_viz("right_foot_target", trajectory.get_T_world_right(t))

            T_world_trunk = np.eye(4)
            T_world_trunk[:3, :3] = trajectory.get_R_world_trunk(t)
            T_world_trunk[:3, 3] = trajectory.get_p_world_CoM(t)
            frame_viz("trunk_target", T_world_trunk)

    # Spin-lock until the next tick
    t += DT
    while time.time() + initial_delay < start_t + t:
        time.sleep(1e-3)

# Save torque data to a CSV file after exiting the loop
import csv
output_file = "/home/sasa/Software/hmnd-robot/walk_torque_data.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time"] + list(robot.joint_names()))  # Header row
    for i, tau in enumerate(torque_log):
        writer.writerow([time_log[i]] + tau.tolist())
print(f"Torque data saved to {output_file}")

# Plot joint torques after the loop exits
import matplotlib.pyplot as plt
import numpy as np

T = np.array(time_log)
Tau = np.stack(torque_log, axis=1)  # shape: (n_joints, len(T))

# Plot all joint torques
plt.figure()
for i, name in enumerate(robot.joint_names()):
    plt.plot(T, Tau[i], label=name)
plt.xlabel("time [s]")
plt.ylabel("torque [Nm]")
plt.legend(loc="upper right", ncol=2, fontsize="small")
plt.title("Joint torques during motion")
plt.show()

# Plot only hip torques
joint_names = list(robot.joint_names())  # Convert to Python list
hip_joints = [name for name in joint_names if "hip" in name.lower()]
hip_indices = [joint_names.index(name) for name in hip_joints]

plt.figure()
for i in hip_indices:
    plt.plot(T, Tau[i], label=joint_names[i])
plt.xlabel("time [s]")
plt.ylabel("torque [Nm]")
plt.legend(loc="upper right", fontsize="small")
plt.title("Hip joint torques during motion")
plt.show()
