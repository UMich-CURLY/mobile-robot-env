class VelMessage():
    type = "VEL"
    def __init__(self, x=0.0,y=0.0,omega=0.0):
        self.x=x
        self.y=y
        self.omega = omega

class WaypointMessage():
    type = "WAYPOINT"
    def __init__(self):
        self.x = [] #list of
        self.y = []

message_types = [VelMessage,WaypointMessage]