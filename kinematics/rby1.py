import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz
from placo_utils.tf import tf

# Loading the robot
robot = placo.RobotWrapper("../../../../ros_nodes/core/prealpha_description/urdf/model.urdf", placo.Flags.ignore_collisions)
viz = robot_viz(robot)

# Creating the solver
solver = placo.KinematicsSolver(robot)

# Control the base
base_task = solver.add_frame_task("base", np.eye(4))  # Task to keep the base in its current position

# Control the left-hand end effector
left_hand_task = solver.add_frame_task("FT_sensor_L", np.eye(4))  # Task for the left-hand end effector

## Omniwheel
#wheel1_task = solver.add_wheel_task("wheel1", 0.04, True)
#wheel2_task = solver.add_wheel_task("wheel2", 0.04, True)
#wheel3_task = solver.add_wheel_task("wheel3", 0.04, True)

t = 0
dt = 0.01

@schedule(interval=dt)
def loop():
    global base_task, left_hand_task, t, dt
    t += dt

    # Move the base in a circular trajectory
    x = 0.25 * np.sin(t)  # X-coordinate
    y = 0.25 * np.cos(t)  # Y-coordinate
    z = 0.0              # Z-coordinate (constant height)
    T_world_base = tf.translation_matrix([x, y, z])  # Create transformation matrix
    # base_task.T_world_frame = T_world_base

    # Move the left-hand end effector in a sinusoidal motion
    lh_x = 0.0  # Constant X-coordinate
    lh_y = 0.4  # Constant Y-coordinate
    lh_z = 0.9 + 0.1 * np.sin(2 * t)  # Z-coordinate oscillates sinusoidally
    T_world_left_hand = tf.translation_matrix([lh_x, lh_y, lh_z])  # Create transformation matrix
    left_hand_task.T_world_frame = T_world_left_hand

    solver.solve(True)
    robot.update_kinematics()
    viz.display(robot.state.q)
    robot_frame_viz(robot, "base")  # Visualize the base frame
    robot_frame_viz(robot, "FT_sensor_L")  # Visualize the left-hand end effector frame

run_loop()
