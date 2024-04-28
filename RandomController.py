from Controller import Controller
import random

class RandomController(Controller):

    def __init__(self,env):
        Controller.__init__(self,env)
        self.action_step = 10
        self.TrafficLight_states = ['G'*12 for i in range(len(self.env._traffic_lights))]
   

    def get_actions(self,time):
        if time%self.action_step==0:
            self.TrafficLight_states = []
            for i in range(len(self.env._traffic_lights)):
                TrafficLight_state = ''
                for j in range(12): 
                    lane_TrafficLight = random.choice(self.TrafficLight_Signal)
                    TrafficLight_state+=lane_TrafficLight
                self.TrafficLight_states.append(TrafficLight_state)
        return self.TrafficLight_states


