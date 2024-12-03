# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


def create_team(first_index, second_index, is_red,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
   """
   This function should return a list of two agents that will form the
   team, initialized using firstIndex and secondIndex as their agent
   index numbers.  isRed is True if the red team is being created, and
   will be False if the blue team is being created.


   As a potentially helpful development aid, this function can take
   additional string-valued keyword arguments ("first" and "second" are
   such arguments in the case of this function), which will come from
   the --redOpts and --blueOpts command-line arguments to capture.py.
   For the nightly contest, however, your team will be created without
   any extra arguments, so you should make sure that the default
   behavior is what you want for the nightly contest.
   """
   return [eval(first)(first_index), eval(second)(second_index)]


class ReflexCaptureAgent(CaptureAgent):
   """
   A base class for reflex agents that choose actions based on a reflexive strategy.
   """
   def __init__(self, index, time_for_computing=.1):
       super().__init__(index, time_for_computing)
       self.start = None


   def register_initial_state(self, game_state):
       self.start = game_state.get_agent_position(self.index)
       CaptureAgent.register_initial_state(self, game_state)


   def get_successor(self, game_state, action):
       """
       Finds the next successor which is a grid position (location tuple).
       """
       successor = game_state.generate_successor(self.index, action)
       pos = successor.get_agent_state(self.index).get_position()
       if pos != nearest_point(pos):
           return successor.generate_successor(self.index, action)
       else:
           return successor

class OffensiveReflexAgent(ReflexCaptureAgent):
   """
   An offensive reflex agent that aggressively pursues food, power dots, and enemies.
   The strategy involves aggressive food collection and using power dots to chase down enemies/food.
   """


   def __init__(self, index):
       super().__init__(index)
       self.food_carried_threshold = 6  # The threshold for returning food to base (adjustable)
       self.power_dot_collected = False  # Flag to check if power dot is collected
       self.stuck_counter = 0           # Track if agent is stuck for too long


   def choose_action(self, game_state):
       actions = game_state.get_legal_actions(self.index)
       my_pos = game_state.get_agent_state(self.index).get_position()
       food_list = self.get_food(game_state).as_list()
       capsules = self.get_capsules(game_state)
       enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
       visible_enemies = [e for e in enemies if e.get_position() is not None and e.is_pacman]
       food_carried = game_state.get_agent_state(self.index).num_carrying


       # If power dot is collected, focus on aggressively pursuing the next food or chasing enemies
       if self.power_dot_collected:
           return self.aggressive_behavior(actions, game_state, food_list, visible_enemies)


       # If we're carrying a lot of food, we need to return it to base
       if food_carried >= self.food_carried_threshold:
           return self.return_to_start(actions, game_state)


       # No food carried, not on power dot yet, so collect food
       if food_list:
           closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
           return self.move_towards(actions, game_state, closest_food)


       # If we are stuck, try to break out
       return self.break_out_of_stuck(actions, game_state)


   def aggressive_behavior(self, actions, game_state, food_list, visible_enemies):
       """
       This behavior prioritizes aggression once the power dot is collected.
       """
       my_pos = game_state.get_agent_state(self.index).get_position()


       # Chase after food aggressively
       if food_list:
           closest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
           return self.move_towards(actions, game_state, closest_food)


       # If no food left, check if we can chase an enemy
       if visible_enemies:
           closest_enemy = min(visible_enemies, key=lambda e: self.get_maze_distance(my_pos, e.get_position()))
           return self.chase_enemy(actions, game_state, closest_enemy)


       # If no food and no enemies, go back to base (the fallback plan)
       return self.return_to_start(actions, game_state)


   def move_towards(self, actions, game_state, target_pos):
       """
       Chooses the best action to move towards a target position.
       """
       best_action = None
       best_dist = float('inf')


       for action in actions:
           successor = self.get_successor(game_state, action)
           successor_pos = successor.get_agent_position(self.index)
           dist = self.get_maze_distance(successor_pos, target_pos)
           if dist < best_dist:
               best_dist = dist
               best_action = action


       return best_action


   def chase_enemy(self, actions, game_state, enemy):
       """
       This method is called once the agent has collected the power dot.
       The agent will now chase after any visible enemy pacman.
       """
       my_pos = game_state.get_agent_state(self.index).get_position()
       enemy_pos = enemy.get_position()


       return self.move_towards(actions, game_state, enemy_pos)


   def return_to_start(self, actions, game_state):
       """
       Returns to the base if carrying enough food.
       """
       my_pos = game_state.get_agent_state(self.index).get_position()
       best_action = None
       best_dist = float('inf')


       for action in actions:
           successor = self.get_successor(game_state, action)
           successor_pos = successor.get_agent_position(self.index)
           dist = self.get_maze_distance(successor_pos, self.start)  # Start position
           if dist < best_dist:
               best_dist = dist
               best_action = action


       return best_action


   def break_out_of_stuck(self, actions, game_state):
       """
       Attempts to break out of stuck situations by trying to move in any direction
        if the agent has been stuck for too long (based on position history).
       """
       my_pos = game_state.get_agent_state(self.index).get_position()


       # If the agent is stuck, it tries to break free by taking a random action
       # to avoid staying in a loop. If it hasn’t moved for several steps, force it to move.
       if self.stuck_counter > 5:
           self.stuck_counter = 0  # Reset stuck counter once we break free
           return random.choice(actions)  # Force random movement if stuck


       self.stuck_counter += 1  # Increment stuck counter
       return random.choice(actions)  # Try a random action as a fallback


   def get_successor(self, game_state, action):
       """
       Generates a successor state by performing the given action.
       """
       successor = game_state.generate_successor(self.index, action)
       return successor

class DefensiveReflexAgent(ReflexCaptureAgent):
   """
   Defensive agent that sticks closer to power dots and guards them effectively.
   The strategy here is to patrol around capsules and intercept invaders (enemy pacmen) attempting to capture food.
   """

   def choose_action(self, game_state):
       actions = game_state.get_legal_actions(self.index)
       my_pos = game_state.get_agent_state(self.index).get_position()
       capsules = self.get_capsules_you_are_defending(game_state)
       enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
       invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]


       # Default: Patrol around capsules if no visible threats
       if capsules:
           closest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
           if not invaders:  # No invaders visible, patrol near capsules
               return self.move_towards(actions, game_state, closest_capsule)


       # If invaders are visible, prioritize chasing them
       if invaders:
           closest_invader = min(invaders, key=lambda inv: self.get_maze_distance(my_pos, inv.get_position()))
           return self.move_towards(actions, game_state, closest_invader.get_position())


       # Default fallback: Move randomly or stay near starting point
       return random.choice(actions)

   def move_towards(self, actions, game_state, target_pos):
       """
       Move towards a specific target position.
       The agent calculates the distance to the target and selects the action that minimizes the distance.
       """
       best_action = None
       best_distance = float('inf')
       for action in actions:
           successor = self.get_successor(game_state, action)
           successor_pos = successor.get_agent_position(self.index)
           distance = self.get_maze_distance(successor_pos, target_pos)
           if distance < best_distance:
               best_distance = distance
               best_action = action
       return best_action