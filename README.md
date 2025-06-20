# SG VLN Project 
## Architecture
The SG VLN project separates the navigation agent (client) and interactions with the robot (server) through a websocket based network interface, allowing the same agent to interface with both isaac sim and physical go1 robot. 

The robot server encapsulates interaction with onboard sensors and issuing action commands, including both direct velocity control and waypoint following via a pure pursuit based local planner. 

The navigation agent then acts as a client that connects to the robot server, requesting sensor data and then issuing action commands. 

## Operation
To use the navigation agent, one of the two robot servers must be first started as following:

### Isaac Lab:
python3 isaac_server.py --episode_index <episode_index>

### Real World:
python3 [text](go1_server_rs2_mp.py)

### Navigation Agent:
python3 remote_main.py --host <server host name>

Once the agent is started, you would be prompted to enter the object goal for navigation. To start a new episode without restarting the agent, type reset and press enter. 

### Debugging Tool:
To validate that the server is working correctly, a gui client is provided which offers visualization of sensor data and teleoperation. It can be started with the following command:

python3 mapping_client.py --host <server host name>