import pinocchio
import numpy as np
from ischedule import schedule, run_loop
import placo
from placo_utils.visualization import (
    robot_viz,
    point_viz,
    tf,
    contacts_viz,
)

"""
In this example, the Sigmaban humanoid robot is on the ground, moving its center of mass above one foot,
and rising the opposing foot.

This demostrates the use of CoM task, orientation task, frame tasks, planar contacts, and puppet contacts
for initialization. 
"""

robot = placo.RobotWrapper("../models/sigmaban/")

robot.update_kinematics()

solver = placo.DynamicsSolver(robot)
legs_spacing = 0.1
t = 0.0
solver.dt = 0.001
view_fps = 50  # FPS for viewer
steps_per_view = int((1 / view_fps) / solver.dt)

# Adding a task for the trunk
com_world = np.array([0.0, 0.0, 0.33])
com_task = solver.add_com_task(com_world)
com_task.configure("com", "soft", 1.0)

trunk_orientation_task = solver.add_orientation_task("trunk", np.eye(3))
trunk_orientation_task.configure("trunk", "soft", 1.0)

# Adding task for feet
T_world_leftFoot = tf.translation_matrix([0.0, legs_spacing / 2, 0.0])
leftFoot_task = solver.add_frame_task("left_foot", T_world_leftFoot)
leftFoot_task.configure("leftFoot", "hard", 1.0, 1.0)
left_contact = solver.add_planar_contact(leftFoot_task)
left_contact.length = 0.15
left_contact.width = 0.09
left_contact.weight_moments = 1e-3

T_world_rightFoot = tf.translation_matrix([0.0, -legs_spacing / 2, 0.0])
rightFoot_task = solver.add_frame_task("right_foot", T_world_rightFoot)
rightFoot_task.configure("rightFoot", "hard", 1.0, 1.0)
right_contact = solver.add_planar_contact(rightFoot_task)
right_contact.length = 0.15
right_contact.width = 0.09
right_contact.weight_moments = 1e-3

# Adding a regularisation task for the posture
posture_regularization_task = solver.add_joints_task()
posture_regularization_task.set_joints({joint: 0.0 for joint in robot.joint_names()})
posture_regularization_task.configure("posture", "soft", 1e-6)

# Enabling all limits
solver.enable_joint_limits(True)
solver.enable_velocity_limits(True)
solver.enable_torque_limits(True)

# Building a trajectory for the CoM, and to rise the right foot
com_trajectory = placo.CubicSpline3D()
com_trajectory.add_point(0.0, com_world, np.zeros(3))
com_trajectory.add_point(
    2.0, com_world + np.array([0.0, legs_spacing / 2, 0.0]), np.zeros(3)
)
com_trajectory.add_point(
    6.0, com_world + np.array([0.0, legs_spacing / 2, 0.0]), np.zeros(3)
)
com_trajectory.add_point(10.0, com_world, np.zeros(3))

rightFoot_trajectory = placo.CubicSpline3D()
rightFoot = T_world_rightFoot[:3, 3]
rightFoot_trajectory.add_point(0.0, rightFoot, np.zeros(3))
rightFoot_trajectory.add_point(2.0, rightFoot, np.zeros(3))
rightFoot_trajectory.add_point(4.0, rightFoot + np.array([0.0, 0.0, 0.1]), np.zeros(3))
rightFoot_trajectory.add_point(6.0, rightFoot, np.zeros(3))
rightFoot_trajectory.add_point(10.0, rightFoot, np.zeros(3))

# Initializing the robot with a puppet contact
puppet = solver.add_puppet_contact()
for k in range(1000):
    robot.add_q_noise(1e-3)  # adding noise to exit singularities
    solver.solve(True)
    robot.update_kinematics()
solver.remove_contact(puppet)

viz = robot_viz(robot)

# Initialize torque and time logs
torque_log = []
time_log = []

@schedule(interval=(1 / view_fps))
def loop():
    global t

    for _ in range(steps_per_view):
        t += solver.dt
        t = t % 10

        # Update targets
        com_task.target_world = com_trajectory.pos(t)
        right_foot_target = rightFoot_trajectory.pos(t)
        rightFoot_task.position().target_world = right_foot_target
        right_contact.active = bool(right_foot_target[2] < 1e-3)

        robot.update_kinematics()
        result = solver.solve(True)  # Capture solver result

    # Log time and torques
    time_log.append(t)
    torque_log.append(result.tau.copy())

    # Display robot and contacts
    viz.display(robot.state.q)
    contacts_viz(solver, ratio=3e-3, radius=0.01)

    # Visualize CoM on the ground
    com = robot.com_world()
    com[2] = 0.0
    point_viz("com", com)

# Run the loop for a fixed duration (e.g., 10 seconds)
import time
start_time = time.time()
while time.time() - start_time < 10:  # Run for 10 seconds
    loop()  # Directly call the loop function

# Save torque data to a CSV file
# import csv
# output_file = "/home/sasa/Software/hmnd-robot/torque_data.csv"
# with open(output_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(["time"] + list(robot.joint_names()))  # Convert to Python list
#     for time, torque in zip(time_log, torque_log):
#         writer.writerow([time] + torque.tolist())

# print(f"Torque data saved to {output_file}")

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
plt.title('Joint torques during motion')
plt.show()
