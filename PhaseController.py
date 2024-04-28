from Controller import Controller

class PhaseController(Controller):

    def __init__(self,env):
        Controller.__init__(self,env)
        self.cycle = 40
        self.greentime_straight = 10
        self.greentime_left = self.cycle/2 - self.greentime_straight
        self.numjunctions = len(self.env._traffic_lights)
        self.TrafficLight_states = ['G'*12 for i in range(self.numjunctions)]
   

    def get_actions(self,time):
        Phases = ['GGrrrrGGrrrr','rrGrrrrrGrrr','rrrGGrrrrGGr','rrrrrGrrrrrG']
        # need a function here
        for i in range(self.numjunctions):
            timeincycle = time % self.cycle
            if timeincycle < self.greentime_straight:
                phase = Phases[0]
            elif timeincycle < self.greentime_straight + self.greentime_left:
                phase = Phases[1]
            elif timeincycle < 2*self.greentime_straight + self.greentime_left:
                phase = Phases[2]
            else:
                phase = Phases[3]
            self.TrafficLight_states[i] = phase

        return self.TrafficLight_states