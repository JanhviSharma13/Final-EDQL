import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

def start_sumo(net_file, route_file, use_gui=False, port=8813):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--step-length", "1",
        "--duration-log.disable", "true",
        "--no-warnings", "true",
        "--log", "NUL"
    ]
    traci.start(sumo_cmd)
    return traci

class EDQLBaseEnv(gym.Env):
    """
    Dynamic-spawning EDQL environment with fixed start/goal edges for reproducibility.
    Spawns rl_agent0 if missing and calculates rewards using paper's MWT bins.
    """
    def __init__(self, net_file, route_file, use_gui=False, max_steps=1000,
                 start_edge=None, goal_edge=None):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.vehicle_id = "rl_agent0"
        self.step_count = 0
        self.start_edge = start_edge
        self.goal_edge = goal_edge

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=200, shape=(4,), dtype=np.float32)

        self.traci = None
        self.initial_route_length = None
        self.total_distance_travelled = 0.0
        self.travel_time = 0.0
        self.success = False

    def reset(self, seed=None, options=None):
        if self.traci:
            self.traci.close()

        self.step_count = 0
        self.total_distance_travelled = 0.0
        self.travel_time = 0.0
        self.success = False
        self.traci = start_sumo(
            net_file=self.net_file,
            route_file=self.route_file,
            use_gui=self.use_gui
        )

        available_edges = self.traci.edge.getIDList()
        if not self.start_edge:
            self.start_edge = next((e for e in available_edges if not e.startswith(":")), available_edges[0])
        if not self.goal_edge:
            self.goal_edge = next((e for e in available_edges if e != self.start_edge and not e.startswith(":")),
                                  available_edges[1])

        self.traci.route.add("r0", [self.start_edge, self.goal_edge])
        #if self.vehicle_id not in self.traci.vehicle.getIDList():
            #self.traci.vehicle.add(self.vehicle_id, "r0")

        self.traci.simulationStep()

        try:
            self.initial_route_length = sum(
                self.traci.lane.getLength(lane_id) for lane_id in self.traci.vehicle.getRoute(self.vehicle_id)
                if not lane_id.startswith(":")
            )
        except Exception:
            self.initial_route_length = None

        obs = self._get_obs()
        info = {
            "route_changed": False,
            "travel_time": 0.0,
            "route_length": self.initial_route_length,
            "avg_speed": 0.0,
            "success": False
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        route_changed = False

        if self.vehicle_id in self.traci.vehicle.getIDList():
            speed = 5 + (action % 4) * 5
            lane = (action // 4) % max(1, self.traci.edge.getLaneNumber(self.traci.vehicle.getRoadID(self.vehicle_id)))
            self.traci.vehicle.setSpeed(self.vehicle_id, speed)
            self.traci.vehicle.changeLane(self.vehicle_id, lane, 25)

        prev_edge = None
        if self.vehicle_id in self.traci.vehicle.getIDList():
            prev_edge = self.traci.vehicle.getRoadID(self.vehicle_id)

        self.traci.simulationStep()

        if self.vehicle_id in self.traci.vehicle.getIDList():
            current_edge = self.traci.vehicle.getRoadID(self.vehicle_id)
            if prev_edge is not None and current_edge != prev_edge:
                route_changed = True
            self.total_distance_travelled += self.traci.vehicle.getSpeed(self.vehicle_id)
            self.travel_time += 1

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()

        if done and self.vehicle_id not in self.traci.vehicle.getIDList():
            self.success = True

        avg_speed = (self.total_distance_travelled / self.travel_time) if self.travel_time > 0 else 0.0
        info = {
            "route_changed": route_changed,
            "travel_time": self.travel_time,
            "route_length": self.initial_route_length,
            "avg_speed": avg_speed,
            "success": self.success
        }

        return obs, reward, done, info

    def _get_obs(self):
        if self.vehicle_id not in self.traci.vehicle.getIDList():
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        speed = self.traci.vehicle.getSpeed(self.vehicle_id)
        lane_pos = self.traci.vehicle.getLanePosition(self.vehicle_id)
        lane_idx = self.traci.vehicle.getLaneIndex(self.vehicle_id)
        mwt = self.traci.vehicle.getWaitingTime(self.vehicle_id)
        return np.array([speed, lane_pos, lane_idx, mwt], dtype=np.float32)

    def _get_reward(self):
        if self.vehicle_id not in self.traci.vehicle.getIDList():
            return 100.0
        mwt = self.traci.vehicle.getWaitingTime(self.vehicle_id)
        if mwt <= 50:
            return -5
        elif 50 < mwt <= 100:
            return -10
        elif 100 < mwt <= 150:
            return -20
        elif 150 < mwt <= 200:
            return -30
        else:
            return -100

    def _is_done(self):
        if self.step_count >= self.max_steps:
            return True
        if self.vehicle_id not in self.traci.vehicle.getIDList():
            return True
        return False

    def close(self):
        if self.traci:
            self.traci.close()
            self.traci = None
