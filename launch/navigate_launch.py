from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Define the parameter file path
    parameter_file_path = LaunchConfiguration('parameter_file')

    return LaunchDescription([
        # Declare the launch argument for the parameter file
        DeclareLaunchArgument(
            'parameter_file',
            default_value=os.path.join(
                get_package_share_directory('mclsplat'),
                'cfg',
                'nerfstudio.yaml'  # Default file name
            ),
            description='Path to the parameter file'
        ),
        
        # Load parameters from the specified YAML file
        Node(
            package='mclsplat',
            executable='nav_node',  # Use the executable name without .py
            name='nav_node',
            output='screen',
            parameters=[parameter_file_path],
        ),
    ])
