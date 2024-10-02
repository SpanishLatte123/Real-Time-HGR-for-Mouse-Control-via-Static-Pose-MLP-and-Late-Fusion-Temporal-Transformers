import enum
import pyautogui
from .gestureTracker import Gesturer
class MacroGestureControl:
    
    @enum.unique
    class GestureState(enum.IntEnum):
        STANDARD = 0
        OFF = 1

    @enum.unique
    class MacroGestures(enum.IntEnum):
        NO_GESTURE = Gesturer.Gesturez.NO_GESTURE
        NO_GESTURE2 = Gesturer.Gesturez.NO_GESTURE2
        CLICK1 = Gesturer.Gesturez.CLICK1
        CLICK2 =Gesturer.Gesturez.CLICK2 
        SWITCH_DESKTOP_RIGHT = Gesturer.Gesturez.SWITCH_DESKTOP_RIGHT
        SWITCH_DESKTOP_LEFT = Gesturer.Gesturez.SWITCH_DESKTOP_LEFT
        NEXT_SONG = Gesturer.Gesturez.NEXT_SONG
        ON_PAUSE_SONG = Gesturer.Gesturez.ON_PAUSE_SONG
        SCREENSHOT = Gesturer.Gesturez.SCREENSHOT
        ALT_TAB = Gesturer.Gesturez.ALT_TAB

    def __init__(self):
        #gesture states
        self.current_state = self.GestureState.OFF
        self.current_gesture = self.MacroGestures.NO_GESTURE
        self.previous_gesture = self.MacroGestures.NO_GESTURE
    

    def _on_gesture_change(self,new_gesture):
        if self.current_state == MacroGestureControl.GestureState.STANDARD:
            self._on_gesture_change_STANDARD(new_gesture)
        if self.current_state == MacroGestureControl.GestureState.OFF:
            self._on_gesture_change_STANDARD(new_gesture)

    def _on_gesture_change_STANDARD(self,new_gesture):
        if new_gesture == self.MacroGestures.NO_GESTURE:
            pass
        elif new_gesture == self.MacroGestures.NO_GESTURE2:
            pass
        elif new_gesture == self.MacroGestures.CLICK1:
            pass
        elif new_gesture == self.MacroGestures.CLICK2:
            pass
        elif new_gesture == self.MacroGestures.SWITCH_DESKTOP_RIGHT:
            self.Copy()
        elif new_gesture == self.MacroGestures.SWITCH_DESKTOP_LEFT:
            self.SDL()
        elif new_gesture == self.MacroGestures.NEXT_SONG:
            self.ON()
        elif new_gesture == self.MacroGestures.ON_PAUSE_SONG:
            self.OFF()
        elif new_gesture == self.MacroGestures.SCREENSHOT:
            self.Screenshot()
        elif new_gesture == self.MacroGestures.ALT_TAB:
            self.SDR()
    def enter_state(self,state):
        self.current_state = state

    def _on_pause_change_OTHERS(self,new_gesture,reference_gesture,target_gesture):
         if new_gesture != reference_gesture:
             self.enter_state(target_gesture)

    #macros commands
    def Copy():
        pyautogui.hotkey('ctrl','c')
    def ON(self):
        print("GESTURING IS ON :> ")
        self.current_state = MacroGestureControl.GestureState.STANDARD
    def OFF(self):
        print("GESTURING IS OFF :< ")
        self.current_state = MacroGestureControl.GestureState.OFF

    def SDR(self):
        pyautogui.hotkey('win','tab')

    def SDL(self):
        pyautogui.hotkey('alt', 'esc')

    def Screenshot(self):
        pyautogui.hotkey('win','shift','s')


    def perform_dynamic_gesture(self,gestureTracked):
        if self.current_state == MacroGestureControl.GestureState.STANDARD:
            
            if gestureTracked.Gesturez != self.previous_gesture:
                print(gestureTracked.Gesturez)
                self._on_gesture_change(gestureTracked.Gesturez)
                self.previous_gesture = gestureTracked.Gesturez

            else:
                self.previous_gesture = gestureTracked.Gesturez
        elif self.current_state == MacroGestureControl.GestureState.OFF:
            if gestureTracked.Gesturez != self.previous_gesture:
                print(gestureTracked.Gesturez)
                if(gestureTracked.Gesturez == MacroGestureControl.MacroGestures.NEXT_SONG):
                    self._on_gesture_change(gestureTracked.Gesturez)
                    self.previous_gesture = gestureTracked.Gesturez
                else:
                    self.previous_gesture = gestureTracked.Gesturez
                    
            else:
                self.previous_gesture = gestureTracked.Gesturez


            