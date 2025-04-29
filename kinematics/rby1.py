import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, robot_frame_viz
from placo_utils.tf import tf
from pinocchio.robot_wrapper import RobotWrapper as PinocchioRobotWrapper

# Loading the robot
urdf_path = "../../../../ros_nodes/core/prealpha_description/urdf/model.urdf"
model = pinocchio.buildModelFromUrdf(urdf_path)
visual_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.VISUAL)
collision_model = pinocchio.buildGeomFromUrdf(model, urdf_path, pinocchio.GeometryType.COLLISION)
robot = placo.RobotWrapper(model, collision_model, visual_model)
viz = robot_viz(robot)

# Creating the solver
solver = placo.KinematicsSolver(robot)

# Control the base
base_task = solver.add_frame_task("base", np.eye(4))  # Task to keep the base in its current position

# Control the left-hand end effector
left_hand_task = solver.add_frame_task("FT_sensor_L", np.eye(4))  # Task for the left-hand end effector

# Control the right-hand end effector
right_hand_task = solver.add_frame_task("FT_sensor_R", np.eye(4))  # Task for the right-hand end effector
regularization_task = solver.add_regularization_task(1e-5)
# Adds the constraint to the solver
solver.add_avoid_self_collisions_constraint()

## Omniwheel
#wheel1_task = solver.add_wheel_task("wheel1", 0.04, True)
#wheel2_task = solver.add_wheel_task("wheel2", 0.04, True)
#wheel3_task = solver.add_wheel_task("wheel3", 0.04, True)

t = 0
dt = 0.01

@schedule(interval=dt)
def loop():
    global base_task, left_hand_task, right_hand_task, t, dt
    t += dt

    # Move the base in a circular trajectory
    x = 0.25 * np.sin(t)  # X-coordinate
    y = 0.25 * np.cos(t)  # Y-coordinate
    z = 0.0              # Z-coordinate (constant height)
    T_world_base = tf.translation_matrix([x, y, z])  # Create transformation matrix
    base_task.T_world_frame = T_world_base  # Update the base position to follow the circular trajectory

    # Move the left-hand end effector in a sinusoidal motion
    lh_x = 0.2  # Constant X-coordinate
    lh_y = 0.25  # Constant Y-coordinate
    lh_z = 0.8 + 0.1 * np.sin(2 * t)  # Z-coordinate oscillates sinusoidally
    T_world_left_hand = tf.translation_matrix([lh_x, lh_y, lh_z])  # Create transformation matrix
    left_hand_task.T_world_frame = T_world_left_hand

    # Move the right-hand end effector in a sinusoidal motion
    rh_x = 0.2  # Constant X-coordinate
    rh_y = -0.25  # Constant Y-coordinate
    rh_z = 0.8 - 0.1 * np.sin(2 * t)  # Z-coordinate oscillates sinusoidally
    T_world_right_hand = tf.translation_matrix([rh_x, rh_y, rh_z])  # Create transformation matrix
    right_hand_task.T_world_frame = T_world_right_hand

    solver.solve(True)
    robot.update_kinematics()
    viz.display(robot.state.q)
    robot_frame_viz(robot, "base")  # Visualize the base frame
    robot_frame_viz(robot, "FT_sensor_L")  # Visualize the left-hand end effector frame
    robot_frame_viz(robot, "FT_sensor_R")  # Visualize the right-hand end effector frame

run_loop()
