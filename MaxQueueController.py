from Controller import Controller

class MaxQueueController(Controller):

    def __init__(self,env):
        Controller.__init__(self,env)
        self.action_step = 10
        self.traffic_lights = self.env._traffic_lights
        self.num_traffic_lights = len(self.traffic_lights)
        self.Traffic_Light_states_space = ['GGrrrrGGrrrr','rrGrrrrrGrrr','rrrGGrrrrGGr','rrrrrGrrrrrG']
        self.Lane_Group = [[0,1,6,7],[2,8],[3,4,9,10],[5,11]]
        self.TrafficLight_states = ['G'*12 for i in range(self.num_traffic_lights)]
        
   
    def get_actions(self,time):
        if time%self.action_step==0:
            for i, traffic_light in enumerate(self.traffic_lights):
                Traffic_Light_Controlled_Lanes = self.env.get_Traffic_Light_Controlled_Lanes(traffic_light)
                Traffic_Light_Controlled_Lanes_Groups = []
                Queues = []
                for Group_Number in range(len(self.Lane_Group)):
                    Traffic_Light_Controlled_Lanes_Groups.append(
                        [Traffic_Light_Controlled_Lanes[j] for j in self.Lane_Group[Group_Number]]
                        )
                for Group_Number in range(len(self.Lane_Group)):
                    Queues.append(
                        sum(self.env.get_Vehicle_Numbers(Traffic_Light_Controlled_Lanes_Groups[Group_Number]))
                    )
                Max_Queue_index = Queues.index(max(Queues))
                self.TrafficLight_states[i] = self.Traffic_Light_states_space[Max_Queue_index]
            return self.TrafficLight_states
        else:
            return self.env.get_Traffic_Lights_States(self.env._traffic_lights)

  
