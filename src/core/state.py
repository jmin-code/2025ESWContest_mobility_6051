from enum import Enum, auto
class Mode(Enum): IDLE=auto(); NAV=auto(); SPEAK=auto(); SOS=auto()

class Store:
    def __init__(self): self.mode=Mode.IDLE; self.speed_kmh=0.0; self.subs=[]
    def subscribe(self, fn): self.subs.append(fn)
    def set_mode(self, mode): 
        if mode!=self.mode: self.mode=mode; [fn(self) for fn in self.subs]
    def set_speed(self, v): self.speed_kmh=v; [fn(self) for fn in self.subs]