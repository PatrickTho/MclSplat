from setuptools import find_packages, setup

package_name = 'mclsplat'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/cfg', [
            'cfg/jackal.yaml',
            'cfg/llff.yaml',
            'cfg/llff_global.yaml',
             ]),
        ('share/' + package_name + '/launch', [
            'launch/navigate_launch.py',
             ]),
        ('share/' + package_name + '/rviz', [
            'rviz/rviz.rviz',
             ]),         
                 
                 
    ],
    install_requires=['setuptools','rclpy',
        'gtsam','nerfstudio'],
    zip_safe=True,
    maintainer='Patrick Thomasius',
    maintainer_email='patrick_christian.thomasius@smail.th-koeln.de',
    description='Run particle filter with Gaussian Splats',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nav_node = mclsplat.nav_node:main'
        ],
    },
)
