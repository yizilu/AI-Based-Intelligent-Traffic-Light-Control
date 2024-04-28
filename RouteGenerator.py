import math
import os
import sys
import traci
import random
import pandas as pd
import numpy as np
from sumolib import checkBinary  
import optparse
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns

class RouteGenerator:
    def __init__(self, max_steps, n_cars_generated, seed, route_file, start_end_points):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self._seed = seed
        self._route_file = route_file
        self._start_end_points = start_end_points

    def generate_routefile(self):
        np.random.seed(self._seed)  # make tests reproducible

        # the generation of cars is distributed according to a sum of two normal distribution

        # The first normal distribution represents morning rush hour
        mu_1, sigma_1 = self._max_steps/3, self._max_steps/12 # mean (8:00) and standard deviation (2 hr)
        timings_1 = np.rint(np.random.normal(mu_1, sigma_1, round(self._n_cars_generated*0.4)))

        # The second normal distribution represents evening rush hour
        mu_2, sigma_2 = self._max_steps*3/4, self._max_steps/12 # mean (18:00) and standard deviation (2 hr)
        timings_2 = np.rint(np.random.normal(mu_2, sigma_2, round(self._n_cars_generated*0.6)))

        # Get the sum of two normal distributions
        self.timings = np.hstack((timings_1,timings_2))

        # Fit the outliners to the upper and lower bounds
        # just throw away
        self.timings = self.timings[self.timings<self._max_steps] 
        self.timings = self.timings[self.timings>0]
        
        # sort the timings
        self.timings = np.sort(self.timings)
        

        # produce the file for cars generation, one car per line
        with open(self._route_file, "w") as routes:
            print("""<routes>
            <vType id="CarA" length="5" maxSpeed="22" carFollowModel="IDM" actionStepLength="1" tau="1.4" speedDev="0.0" accel="1" decel="2" speedFactor="1.0" minGap="2" delta="4" stepping="1.5"/>
            """, file=routes)

            for car_counter, step in enumerate(self.timings):
                random_start_end_points = sample(self._start_end_points, 2)
                print("""    <trip id="%i" type="CarA" depart="%i" fromJunction="%s" toJunction="%s" />""" \
                    % (car_counter, step, random_start_end_points[0], random_start_end_points[1]), file=routes)
                
            print("</routes>", file=routes)
    

#




