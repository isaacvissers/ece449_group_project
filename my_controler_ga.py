# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
import json
from pickle import FALSE
# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
# detailed discussion of this source code.
from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt
class MyControllerGA(KesslerController):
     def __init__(self, chromosome=None):
         if chromosome is None:
             try:
                 with open("best_results", "r") as file:
                     chromosome = json.load(file)["solution"]
             # Load from file here, otherwise use default values
             except:
                chromosome = [400, 300, 90]
         self.eval_frames = 0 #What is this?
         # self.targeting_control is the targeting rulebase, which is static in this controller.
         # Declare variables
         bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
         theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
         ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
         ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
         ship_thrust = ctrl.Consequent(np.arange(-480, 480, 1), 'ship_thrust')
         ship_position_x = ctrl.Antecedent(np.arange(-500, 500, 1), 'ship_pos_x')
         ship_position_y = ctrl.Antecedent(np.arange(-400, 400, 1), 'ship_pos_y')
         ship_heading = ctrl.Antecedent(np.arange(0, 360, 1), 'ship_heading')
         ship_speed = ctrl.Antecedent(np.arange(-240, 240, 1), 'ship_speed')

        # We only care if the position is near the edges
         ship_position_x['NL'] = fuzz.trimf(ship_position_x.universe, [-500, -500, -chromosome[0]])
         ship_position_x['N'] = fuzz.trimf(ship_position_x.universe, [-500, -0, 500])
         ship_position_x['PL'] = fuzz.trimf(ship_position_x.universe, [chromosome[0], 500, 500])

         ship_position_y['NL'] = fuzz.trimf(ship_position_y.universe, [-400, -400, -chromosome[1]])
         ship_position_y['N'] = fuzz.trimf(ship_position_y.universe, [-400, -0, 400])
         ship_position_y['PL'] = fuzz.trimf(ship_position_y.universe, [chromosome[1], 400, 400])

         h = chromosome[2]
         ship_heading['N'] = fuzz.trimf(ship_turn.universe, [90-h,90,90+h])
         ship_heading['S'] = fuzz.trimf(ship_turn.universe, [270-h,270,270+h])
         ship_heading['E1'] = fuzz.trimf(ship_heading.universe, [360-h, 360, 360])
         ship_heading['E2'] = fuzz.trimf(ship_heading.universe, [0, 0, h])
         ship_heading['E'] = np.fmax(ship_heading['E1'].mf, ship_heading['E2'].mf)
         ship_heading['W'] = fuzz.trimf(ship_turn.universe, [180-h,180,180+h])
         # TODO keep doing stuff here
         ship_speed['NL'] = fuzz.trimf(ship_speed.universe, [-240, -240, -chromosome[7]])
         ship_speed['NM'] = fuzz.trimf(ship_speed.universe, [-240, -chromosome[6], -chromosome[4]])
         ship_speed['NS'] = fuzz.trimf(ship_speed.universe, [-chromosome[5], -chromosome[3], 0])
         # ship_speed['Z'] = fuzz.trimf(ship_speed.universe, [-60,0,60])
         ship_speed['PS'] = fuzz.trimf(ship_speed.universe, [0, chromosome[3], chromosome[5]])
         ship_speed['PM'] = fuzz.trimf(ship_speed.universe, [chromosome[4], chromosome[6], 240])
         ship_speed['PL'] = fuzz.trimf(ship_speed.universe, [chromosome[7], 240, 240])

         #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
         bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.025])
         bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.025,0.1])
         bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

         # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
         # Hard-coded for a game step of 1/30 seconds
         theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
         theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
         theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
         # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
         theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
         theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
         theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

         # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
         # Hard-coded for a game step of 1/30 seconds
         ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
         ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
         ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
         # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
         ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
         ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
         ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

         ship_thrust['NL'] = fuzz.trimf(ship_thrust.universe, [-480, -480, -320])
         ship_thrust['NM'] = fuzz.trimf(ship_thrust.universe, [-480, -320, -160])
         ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [-320, -160, 0])
         ship_thrust['Z'] = fuzz.trimf(ship_thrust.universe, [-60,0,60])
         ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [-0, 160, 320])
         ship_thrust['PM'] = fuzz.trimf(ship_thrust.universe, [160, 320, 480])
         ship_thrust['PL'] = fuzz.trimf(ship_thrust.universe, [320, 480, 480])

         #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be thresholded
         # and returned as the boolean 'fire'
         ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
         ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

         #Declare each fuzzy rule
         # rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['PS']))
         # rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'], ship_thrust['PM']))
         # rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['PL']))
         # # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         # rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['PL']))
         # rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'], ship_thrust['PM']))
         # rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['PS']))
         # rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
         # rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
         # rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['PM']))
         # # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         # rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['PM']))
         # rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
         # rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
         # rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['NS']))
         # rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['NM']))
         # rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['NM']))
         # # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         # rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['NM']))
         # rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['NM']))
         # rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['NS']))

         rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['Z']))
         rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'], ship_thrust['Z']))
         rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['PL']))
         # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['PL']))
         rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'], ship_thrust['Z']))
         rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['Z']))
         rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'], ship_thrust['Z']))
         rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'], ship_thrust['Z']))
         rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['PM']))
         # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['PM']))
         rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'], ship_thrust['Z']))
         rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'], ship_thrust['Z']))
         rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'], ship_thrust['Z']))
         rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'], ship_thrust['Z']))
         rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'], ship_thrust['NM']))
         # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
         rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'], ship_thrust['NM']))
         rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'], ship_thrust['Z']))
         rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'], ship_thrust['Z']))

        # TODO I want to avoid the edges of the field of play
         rule22 = ctrl.Rule(ship_position_x['PL'] & ship_heading['E'], (ship_thrust['NL']))
         rule23 = ctrl.Rule(ship_position_x['NL'] & ship_heading['W'], (ship_thrust['NL']))
         rule24 = ctrl.Rule(ship_position_y['PL'] & ship_heading['N'], (ship_thrust['NL']))
         rule25 = ctrl.Rule(ship_position_y['NL'] & ship_heading['S'], (ship_thrust['NL']))
         rule26 = ctrl.Rule(ship_position_x['PL'] & ship_heading['W'], (ship_thrust['PL']))
         rule27 = ctrl.Rule(ship_position_x['NL'] & ship_heading['E'], (ship_thrust['PL']))
         rule28 = ctrl.Rule(ship_position_y['PL'] & ship_heading['S'], (ship_thrust['PL']))
         rule29 = ctrl.Rule(ship_position_y['NL'] & ship_heading['N'], (ship_thrust['PL']))

         #DEBUG
         #bullet_time.view()
         #theta_delta.view()
         #ship_turn.view()
         #ship_fire.view()



         # Declare the fuzzy controller, add the rules
         # This is an instance variable, and thus available for other methods in the same object. See notes.
         # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

         self.targeting_control = ctrl.ControlSystem()
         self.targeting_control.addrule(rule1)
         self.targeting_control.addrule(rule2)
         self.targeting_control.addrule(rule3)
         # self.targeting_control.addrule(rule4)
         self.targeting_control.addrule(rule5)
         self.targeting_control.addrule(rule6)
         self.targeting_control.addrule(rule7)
         self.targeting_control.addrule(rule8)
         self.targeting_control.addrule(rule9)
         self.targeting_control.addrule(rule10)
         # self.targeting_control.addrule(rule11)
         self.targeting_control.addrule(rule12)
         self.targeting_control.addrule(rule13)
         self.targeting_control.addrule(rule14)
         self.targeting_control.addrule(rule15)
         self.targeting_control.addrule(rule16)
         self.targeting_control.addrule(rule17)
         # self.targeting_control.addrule(rule18)
         self.targeting_control.addrule(rule19)
         self.targeting_control.addrule(rule20)
         self.targeting_control.addrule(rule21)
         self.targeting_control.addrule(rule22)
         self.targeting_control.addrule(rule23)
         self.targeting_control.addrule(rule24)
         self.targeting_control.addrule(rule25)
         self.targeting_control.addrule(rule26)
         self.targeting_control.addrule(rule27)
         self.targeting_control.addrule(rule28)
         self.targeting_control.addrule(rule29)


     def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
         """
         Method processed each time step by this controller.
         """
         # These were the constant actions in the basic demo, just spinning and shooting.
         #thrust = 0 <- How do the values scale with asteroid velocity vector?
         #turn_rate = 90 <- How do the values scale with asteroid velocity vector?

         # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
         # So are the ship position and velocity, and bullet position and velocity.
         # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
         # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
         # So, position is updated by multiplying velocity by delta_time, and adding that to position.
         # Ship velocity is updated by multiplying thrust by delta time.
         # Ship position for this time increment is updated after the the thrust was applied.

         # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
         # Goal: demonstrate processing of game state, fuzzy controller, intercept computation
         # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.
         # Find the closest asteroid (disregards asteroid velocity)
         ship_pos_x = ship_state["position"][0] # See src/kesslergame/ship.py in the KesslerGame Github
         ship_pos_y = ship_state["position"][1]
         ship_velocity = ship_state["velocity"]
         closest_asteroid = None

         for a in game_state["asteroids"]:
            # Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)

            else:
                # closest_asteroid exists, and is thus initialized.
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist
         # closest_asteroid is now the nearest asteroid object.
         # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
         # Based on Law of Cosines calculation, see notes.

         # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
         # and the angle of the asteroid's current movement.
         # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!


         asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
         asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
         # print(ship_state['heading'])

         asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

         asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
         my_theta2 = asteroid_ship_theta - asteroid_direction
         cos_my_theta2 = math.cos(my_theta2)
         # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
         asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
         bullet_speed = 800 # Hard-coded bullet speed from bullet.py

         # Determinant of the quadratic formula b^2-4ac
         targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])

         # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
         intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
         intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

         # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
         if intrcpt1 > intrcpt2:
             if intrcpt2 >= 0:
                bullet_t = intrcpt2
             else:
                bullet_t = intrcpt1
         else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

         # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
         # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
         intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
         intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

         my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

         # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
         shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])

         # Wrap all angles to (-pi, pi)
         shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

         # Pass the inputs to the rulebase and fire it
         shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

         shooting.input['bullet_time'] = bullet_t
         shooting.input['theta_delta'] = shooting_theta
         shooting.input['ship_heading'] = ship_state['heading']
         shooting.input['ship_pos_x'] = ship_pos_x
         shooting.input['ship_pos_y'] = ship_pos_y

         shooting.compute()

         # Get the defuzzified outputs
         turn_rate = shooting.output['ship_turn']
         thrust = shooting.output['ship_thrust']

         if shooting.output['ship_fire'] >= 0:
            fire = True
         else:
            fire = False

         # And return your four outputs to the game simulation. Controller algorithm complete.
         # thrust = 0.0
         drop_mine = False

         self.eval_frames +=1

         #DEBUG
         # print(thrust, bullet_t, shooting_theta, turn_rate, fire)

         return thrust, turn_rate, fire, drop_mine
     @property
     def name(self) -> str:
        return "My Fuzzy Controller"
