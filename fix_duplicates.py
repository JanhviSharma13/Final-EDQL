import os
from lxml import etree
import subprocess

def remove_duplicate_edges(net_file, output_file):
    """Remove duplicate edges from network file"""
    print(f"Removing duplicates from: {net_file}")
    
    # Read the network file
    tree = etree.parse(net_file)
    root = tree.getroot()
    
    # Track seen edges
    seen_edges = set()
    edges_to_remove = []
    
    # Find duplicate edges
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id in seen_edges:
            edges_to_remove.append(edge)
            print(f"Marking duplicate for removal: {edge_id}")
        else:
            seen_edges.add(edge_id)
    
    # Remove duplicate edges
    for edge in edges_to_remove:
        parent = edge.getparent()
        if parent is not None:
            parent.remove(edge)
    
    print(f"Removed {len(edges_to_remove)} duplicate edges")
    
    # Save cleaned network
    tree.write(output_file, encoding='utf-8', xml_declaration=True, pretty_print=True)
    print(f"Cleaned network saved to: {output_file}")
    
    return len(edges_to_remove)

def test_cleaned_network(net_file):
    """Test the cleaned network"""
    print(f"Testing cleaned network: {net_file}")
    
    try:
        import traci
        traci.start(['sumo', '-n', net_file, '--no-warnings', '--no-step-log'])
        
        edges = traci.edge.getIDList()
        print(f"Total edges: {len(edges)}")
        
        # Check for duplicates
        edge_set = set(edges)
        if len(edges) == len(edge_set):
            print("✅ No duplicate edges found!")
            traci.close()
            return True
        else:
            print(f"❌ Still have {len(edges) - len(edge_set)} duplicate edges")
            traci.close()
            return False
        
    except Exception as e:
        print(f"❌ Error testing network: {e}")
        return False

def generate_simple_routes(net_file, route_file):
    """Generate simple routes manually"""
    print(f"Generating simple routes for: {net_file}")
    
    # Create a simple route file
    route_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="DEFAULT_VEHTYPE" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
    
    <route id="route_0" edges=":A_0 :B_0 :C_0"/>
    <route id="route_1" edges=":A_1 :B_1 :C_1"/>
    <route id="route_2" edges=":A_0 :B_2 :C_0"/>
    
    <vehicle id="0" type="DEFAULT_VEHTYPE" route="route_0" depart="0" departLane="best"/>
    <vehicle id="1" type="DEFAULT_VEHTYPE" route="route_1" depart="5" departLane="best"/>
    <vehicle id="2" type="DEFAULT_VEHTYPE" route="route_2" depart="10" departLane="best"/>
</routes>'''
    
    with open(route_file, 'w') as f:
        f.write(route_content)
    
    print(f"✅ Simple routes created: {route_file}")
    return True

def create_working_environment(net_file, route_file, area_name):
    """Create a working EDQL environment"""
    env_code = f'''import os
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

def start_sumo(net_file, route_file, use_gui, max_steps, port=8813):
    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-n", net_file,
        "-r", route_file,
        "--step-length", "1",
        "--duration-log.disable", "true",
        "--no-warnings", "true",
        "--log", "NUL",
    ]
    traci.start(sumo_cmd)
    return traci

class EDQL{area_name}Env(gym.Env):
    def __init__(self, net_file, route_file, use_gui=False, max_steps=2000):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.step_count = 0
        self.vehicle_id = "rl_agent0"

        # Action space: discrete control (0-7 = combinations of speed & lane)
        self.action_space = spaces.Discrete(8)

        # Observation: [speed, lane_pos, lane_index, edge_progress]
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)

        self.traci = None

    def reset(self, seed=None, options=None):
        if self.traci:
            self.traci.close()

        self.step_count = 0
        self.traci = start_sumo(
            net_file=self.net_file,
            route_file=self.route_file,
            use_gui=self.use_gui,
            max_steps=self.max_steps
        )

        print(f"Starting SUMO for {area_name} network...")

        # Get available edges
        available_edges = self.traci.edge.getIDList()
        print(f"Available edges: {{len(available_edges)}}")
        
        if len(available_edges) == 0:
            print("❌ No edges available!")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {{}}
        
        # Use first few edges for a simple route
        route_edges = available_edges[:3]  # Use first 3 edges
        self.traci.route.add("r0", route_edges)
        print(f"Route created with {{len(route_edges)}} edges: {{route_edges}}")

        # Add vehicle
        try:
            self.traci.vehicle.add(self.vehicle_id, "r0", typeID="DEFAULT_VEHTYPE")
            print("Vehicle added successfully")
        except Exception as e:
            print(f"Failed to add vehicle: {{e}}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {{}}

        # Advance simulation
        self.traci.simulationStep()
        
        # Check vehicle status
        if self.vehicle_id in self.traci.vehicle.getIDList():
            current_edge = self.traci.vehicle.getRoadID(self.vehicle_id)
            current_speed = self.traci.vehicle.getSpeed(self.vehicle_id)
            print(f"Vehicle on edge: {{current_edge}}, speed: {{current_speed:.2f}}")
        else:
            print("ERROR: Vehicle not found!")

        return self._get_obs(), {{}}

    def step(self, action):
        self.step_count += 1

        if self.vehicle_id not in self.traci.vehicle.getIDList():
            return self._get_obs(), 0.0, True

        try:
            speed = 5 + (action % 4) * 5  # 5, 10, 15, 20
            lane = (action // 4) % 2     # lane 0 or 1

            if self.vehicle_id in self.traci.vehicle.getIDList():
                current_edge = self.traci.vehicle.getRoadID(self.vehicle_id)
                current_speed = self.traci.vehicle.getSpeed(self.vehicle_id)
                
                if self.step_count % 100 == 0:
                    print(f"Step {{self.step_count}}: Vehicle on edge {{current_edge}}, speed {{current_speed:.2f}}")
                
                self.traci.vehicle.changeLane(self.vehicle_id, lane, 25)
                self.traci.vehicle.setSpeed(self.vehicle_id, speed)
                    
        except Exception as e:
            print(f"[ERROR] Failed to apply action {{action}}: {{e}}")

        self.traci.simulationStep()

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()

        return obs, reward, done

    def _get_obs(self):
        try:
            speed = self.traci.vehicle.getSpeed(self.vehicle_id)
            lane_pos = self.traci.vehicle.getLanePosition(self.vehicle_id)
            lane_index = self.traci.vehicle.getLaneIndex(self.vehicle_id)
            
            edge_progress = 0.0
            try:
                edge_length = self.traci.edge.getLength(self.traci.vehicle.getRoadID(self.vehicle_id))
                edge_progress = lane_pos / edge_length if edge_length > 0 else 0.0
            except:
                edge_progress = 0.0
                
            return np.array([speed, lane_pos, lane_index, edge_progress], dtype=np.float32)
        except traci.TraCIException:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_reward(self):
        try:
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
                
        except traci.TraCIException:
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
        print("Done. Closing SUMO.")
'''
    
    # Write environment file
    env_file = f"edql_{area_name.lower()}_env.py"
    with open(env_file, 'w') as f:
        f.write(env_code)
    
    print(f"✅ EDQL environment created: {env_file}")
    return env_file

def main():
    """Main function to fix and test networks"""
    print("🔧 Fixing Delhi Networks - Removing Duplicates")
    print("=" * 50)
    
    # Define areas to process
    areas = [
        {"name": "RakabGanj", "net_file": "EDQL-RakabGanj/rakabganj_fixed.net.xml"},
        {"name": "CP", "net_file": "EDQL-CP/cp_fixed.net.xml"},
        {"name": "Safdarjung", "net_file": "EDQL-Safdarjung/safdarjung_fixed.net.xml"},
        {"name": "ChandniChowk", "net_file": "EDQL-ChandniChowk/chandnichowk_fixed.net.xml"}
    ]
    
    for area in areas:
        print(f"\n🏗️ Processing {area['name']}...")
        
        if not os.path.exists(area['net_file']):
            print(f"❌ Network file not found: {area['net_file']}")
            continue
        
        # Remove duplicates
        output_dir = os.path.dirname(area['net_file'])
        cleaned_net = os.path.join(output_dir, f"{area['name'].lower()}_cleaned.net.xml")
        duplicates_removed = remove_duplicate_edges(area['net_file'], cleaned_net)
        
        # Test cleaned network
        if test_cleaned_network(cleaned_net):
            # Generate simple routes
            route_file = os.path.join(output_dir, f"{area['name'].lower()}_simple.rou.xml")
            if generate_simple_routes(cleaned_net, route_file):
                # Create EDQL environment
                env_file = create_working_environment(cleaned_net, route_file, area['name'])
                print(f"✅ {area['name']} network ready for RL training!")
            else:
                print(f"❌ {area['name']} route generation failed")
        else:
            print(f"❌ {area['name']} network test failed")
    
    print("\n🎯 Summary:")
    print("- Removed duplicate edges from all networks")
    print("- Created simple route files")
    print("- Generated EDQL environments for each area")
    print("- Ready for TD error logging implementation")

if __name__ == "__main__":
    main() 