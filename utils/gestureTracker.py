import enum

class Gesturer:
    @enum.unique
    class Gesturez(enum.IntEnum):
        NO_GESTURE = 0
        NO_GESTURE2 = 1
        CLICK1 = 2
        CLICK2 =3
        SWITCH_DESKTOP_LEFT = 4
        SWITCH_DESKTOP_RIGHT = 5
        NEXT_SONG = 6
        ON_PAUSE_SONG = 7
        SCREENSHOT = 8
        ALT_TAB = 9
    
    def __init__(self,gesture=None):
        if gesture is None:
            self.Gesturez = self.Gesturez.NO_GESTURE
        else:
            if isinstance(gesture, self.Pose):
                self.pose = gesture
            if isinstance(gesture, int):
                self.pose = self.Gesturez(gesture)
            if isinstance(gesture, str):
                self.pose = self.Gesturez[gesture]