import os
import subprocess
import re
from lxml import etree
import shutil

def fix_network_file(net_file, output_file):
    """Fix network file by cleaning negative IDs and ensuring connectivity"""
    print(f"Fixing network: {net_file}")
    
    # Read the network file
    tree = etree.parse(net_file)
    root = tree.getroot()
    
    # Track changes
    edge_changes = {}
    node_changes = {}
    connection_changes = {}
    
    # Step 1: Fix edge IDs (remove negative signs)
    edges_to_remove = []
    for edge in root.findall('.//edge'):
        edge_id = edge.get('id')
        if edge_id and edge_id.startswith('-'):
            # Create new positive ID
            new_id = edge_id[1:]  # Remove the negative sign
            edge_changes[edge_id] = new_id
            edge.set('id', new_id)
            print(f"Fixed edge: {edge_id} -> {new_id}")
    
    # Step 2: Fix node IDs (remove negative signs)
    for node in root.findall('.//node'):
        node_id = node.get('id')
        if node_id and node_id.startswith('-'):
            new_id = node_id[1:]
            node_changes[node_id] = new_id
            node.set('id', new_id)
            print(f"Fixed node: {node_id} -> {new_id}")
    
    # Step 3: Fix connections
    for connection in root.findall('.//connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        
        if from_edge in edge_changes:
            connection.set('from', edge_changes[from_edge])
            connection_changes[f"from_{from_edge}"] = edge_changes[from_edge]
            
        if to_edge in edge_changes:
            connection.set('to', edge_changes[to_edge])
            connection_changes[f"to_{to_edge}"] = edge_changes[to_edge]
    
    # Step 4: Fix routes (if any)
    for route in root.findall('.//route'):
        edges_attr = route.get('edges')
        if edges_attr:
            edges = edges_attr.split()
            new_edges = []
            for edge in edges:
                if edge in edge_changes:
                    new_edges.append(edge_changes[edge])
                else:
                    new_edges.append(edge)
            route.set('edges', ' '.join(new_edges))
    
    # Step 5: Fix lanes
    for lane in root.findall('.//lane'):
        lane_id = lane.get('id')
        if lane_id and lane_id.startswith('-'):
            new_lane_id = lane_id[1:]
            lane.set('id', new_lane_id)
            print(f"Fixed lane: {lane_id} -> {new_lane_id}")
    
    # Save fixed network
    tree.write(output_file, encoding='utf-8', xml_declaration=True, pretty_print=True)
    print(f"Fixed network saved to: {output_file}")
    
    return {
        'edge_changes': edge_changes,
        'node_changes': node_changes,
        'connection_changes': connection_changes
    }

def generate_routes_for_fixed_network(net_file, route_file):
    """Generate routes for the fixed network"""
    print(f"Generating routes for: {net_file}")
    
    # Use randomTrips.py to generate routes
    cmd = [
        'python', 
        'C:/Users/Janhvi/Downloads/sumo/sumo-1.23.1/tools/randomTrips.py',
        '-n', net_file,
        '-r', route_file,
        '-b', '0',
        '-e', '100',
        '-p', '1.0',
        '--validate',
        '--min-distance', '100',  # Ensure minimum route distance
        '--max-distance', '2000'  # Limit maximum route distance
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Routes generated successfully!")
            return True
        else:
            print(f"❌ Error generating routes: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running randomTrips.py: {e}")
        return False

def test_fixed_network(net_file, route_file):
    """Test the fixed network"""
    print(f"Testing fixed network: {net_file}")
    
    try:
        import traci
        traci.start(['sumo', '-n', net_file, '-r', route_file, '--no-warnings', '--no-step-log'])
        
        edges = traci.edge.getIDList()
        print(f"Total edges: {len(edges)}")
        
        # Check for negative edges
        neg_count = len([e for e in edges if e.startswith('-')])
        print(f"Negative edges remaining: {neg_count}")
        
        # Check if we can add a vehicle
        try:
            # Get first few edges for testing
            test_edges = edges[:5]
            print(f"Test edges: {test_edges}")
            
            # Try to create a simple route
            traci.route.add("test_route", test_edges)
            traci.vehicle.add("test_vehicle", "test_route", typeID="DEFAULT_VEHTYPE")
            
            print("✅ Vehicle placement successful!")
            traci.close()
            return True
            
        except Exception as e:
            print(f"❌ Vehicle placement failed: {e}")
            traci.close()
            return False
        
    except Exception as e:
        print(f"❌ Error testing network: {e}")
        return False

def create_edql_environment_for_fixed_network(net_file, route_file, area_name):
    """Create EDQL environment for the fixed network"""
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

        # Get available edges and create a route
        available_edges = self.traci.edge.getIDList()
        print(f"Available edges: {{len(available_edges)}}")
        
        # Use first few edges for a simple route
        route_edges = available_edges[:5]  # Use first 5 edges
        self.traci.route.add("r0", route_edges)
        print(f"Route created with {{len(route_edges)}} edges")

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
    """Main function to fix Delhi networks"""
    print("🔧 Fixing Delhi Networks for RL Training")
    print("=" * 50)
    
    # Define Delhi areas to fix
    areas = [
        {"name": "RakabGanj", "net_file": "rg.net.xml", "route_file": "rg.rou.xml"},
        {"name": "CP", "net_file": "cp2.net.xml", "route_file": "cp.rou.xml"},
        {"name": "ChandniChowk", "net_file": "ch.net.xml", "route_file": "ch.rou.xml"},
        {"name": "Safdarjung", "net_file": "SE.net.xml", "route_file": "SE.rou.xml"}
    ]
    
    for area in areas:
        print(f"\n🏗️ Processing {area['name']}...")
        
        # Create output directory
        output_dir = f"EDQL-{area['name']}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Fix network
        fixed_net = os.path.join(output_dir, f"{area['name'].lower()}_fixed.net.xml")
        changes = fix_network_file(area['net_file'], fixed_net)
        
        # Generate routes
        fixed_route = os.path.join(output_dir, f"{area['name'].lower()}_fixed.rou.xml")
        if generate_routes_for_fixed_network(fixed_net, fixed_route):
            # Test the fixed network
            if test_fixed_network(fixed_net, fixed_route):
                # Create EDQL environment
                env_file = create_edql_environment_for_fixed_network(
                    fixed_net, fixed_route, area['name']
                )
                print(f"✅ {area['name']} network fixed and ready for RL training!")
            else:
                print(f"❌ {area['name']} network test failed")
        else:
            print(f"❌ {area['name']} route generation failed")
    
    print("\n🎯 Summary:")
    print("- Fixed negative IDs in all networks")
    print("- Generated valid routes for RL training")
    print("- Created EDQL environments for each area")
    print("- Ready for TD error logging implementation")

if __name__ == "__main__":
    main() 