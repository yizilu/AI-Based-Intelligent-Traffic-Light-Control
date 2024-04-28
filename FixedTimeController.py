from Controller import Controller

class FixedTimeController(Controller):

    def __init__(self,env):
        Controller.__init__(self,env)
        self.cycle = 20
        self.greentime = 10
        self.numjunctions = len(self.env._traffic_lights)
        self.TrafficLight_states = ['g'*12 for i in range(self.numjunctions)]
   
    def get_actions(self,time):

        # need a function here
        for i in range(self.numjunctions):
            timeincycle = time % self.cycle
            color = 'g' if timeincycle<self.greentime else 'r'
            TrafficLight_state = color*12
            self.TrafficLight_states[i] = TrafficLight_state

        return self.TrafficLight_states


