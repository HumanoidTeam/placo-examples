"""Launch the controllers and the robot state publisher."""

from ament_index_python.packages import (
    get_package_share_path,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from prealpha_launch_utils import (
    launch_configurations,
    load_robot_description,
    process_file,
)


@launch_configurations
def make_nodes(args):
    """Create a Robot State Publisher node given a hardware type."""
    if len(args.robot_ip) == 0 and args.hardware_type == "real":
        msg = f"Robot IP is required for hardware type '{args.hardware_type}' - IP: '{args.robot_ip}'"
        raise ValueError(msg)
    mappings = {
        "ros2_control_hardware_type": args.hardware_type,
        "robot_ip": args.robot_ip,
        "use_prealpha_head": "true",
        "gripper": args.gripper,
    }
    robot_description = load_robot_description(mappings)
    ros2_controllers_file = process_file(
        get_package_share_path("prealpha_description")
        / "control"
        / "ros2_controllers.yaml",
        mappings=mappings,
        save=True,
    )

    startup_controller_nodes = [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller],
        )
        for controller in [
            "joint_state_broadcaster",
            "torso_forward_position_controller",
            "gripper_forward_position_controller",
            "head_forward_position_controller",
            "left_ft_sensor",
            "right_ft_sensor",
        ]
    ] + [
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["--inactive", controller],
        )
        for controller in [
            "joint_trajectory_controller",
            "left_gripper_controller",
            "right_gripper_controller",
            "joint_admittance_controller",
            "forward_position_controller",
            "joint_admittance_forward_position_controller",
        ]
    ]

    if args.hardware_type == "real":
        startup_controller_nodes.extend(
            [
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["odometry_broadcaster"],
                ),
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["cmd_vel_forward_controller"],
                ),
            ],
        )

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description},
            ],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[ros2_controllers_file],
            remappings=[("~/robot_description", "/robot_description")],
            # To get logs from spdlog
            output="screen",
            # Colorful output
            emulate_tty=True,
        ),
        *startup_controller_nodes,
    ]


def generate_launch_description():
    """Generate the launch descriptions."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hardware_type",
                default_value="mock_components",
            ),
            DeclareLaunchArgument("robot_ip", default_value=""),
            DeclareLaunchArgument("gripper", default_value="robotiq"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["0", "0", "0", "0", "0", "0", "world", "base"],
                output="screen",
            ),
            *make_nodes(),
        ],
    )
