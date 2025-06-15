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


class HabitatMessage():
    type = "HABITAT"
    action_names = ["STOP","FORWARD","LEFT","RIGHT","LOOPUP","LOOKDOWN"]
    def __init__self(self,action):
        self.number = action
    def __repr__(self):
        return HabitatMessage.action_names[self.action_names]

message_types = [VelMessage,WaypointMessage,HabitatMessage]