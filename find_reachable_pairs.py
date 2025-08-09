# find_reachable_pairs.py
import os
import sumolib
import networkx as nx
import json

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
AREAS = [
    {"name": "RakabGanj", "net": "rakabganj_cleaned.net.xml"},
    {"name": "CP2", "net": "cp_cleaned.net.xml"},
    {"name": "Safdarjung", "net": "safdarjung_cleaned.net.xml"},
    {"name": "ChandniChowk", "net": "chandnichowk_cleaned.net.xml"}
]

MIN_ROUTE_DIST = 300     # meters
INTERNAL_PREFIX = ":"    # SUMO internal edges start with ':'

# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------
def find_longest_pair(net_file):
    print(f"\nProcessing {net_file} ...")

    # Load network
    net = sumolib.net.readNet(net_file)

    # Build graph excluding internal edges (nodes are SUMO node IDs)
    G = nx.DiGraph()
    for edge in net.getEdges():
        if edge.getID().startswith(INTERNAL_PREFIX):
            continue
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        length = edge.getLength()
        G.add_edge(from_node, to_node, edge_id=edge.getID(), weight=length)

    # Find longest shortest path
    longest_dist = -1
    best_pair = None

    nodes = list(G.nodes)
    for i, src in enumerate(nodes):
        lengths, paths = nx.single_source_dijkstra(G, src, weight="weight")
        for dst, dist in lengths.items():
            if dist >= MIN_ROUTE_DIST and dist > longest_dist:
                if dst in paths and len(paths[dst]) > 1:
                    path_nodes = paths[dst]

                    # Extract edges along the path
                    path_edges = []
                    valid_path = True
                    for j in range(len(path_nodes) - 1):
                        from_node = path_nodes[j]
                        to_node = path_nodes[j + 1]
                        # Find edges from from_node to to_node
                        edges_between = [
                            e for e in net.getNode(from_node).getOutgoing()
                            if e.getToNode().getID() == to_node
                        ]
                        if edges_between:
                            path_edges.append(edges_between[0].getID())
                        else:
                            # No edge found between these nodes, invalid path
                            valid_path = False
                            break

                    if not valid_path or not path_edges:
                        continue

                    start_edge_id = path_edges[0]
                    end_edge_id = path_edges[-1]

                    if start_edge_id != end_edge_id:
                        longest_dist = dist
                        best_pair = (start_edge_id, end_edge_id)

    return best_pair, longest_dist

if __name__ == "__main__":
    config = {}
    for area in AREAS:
        net_file = area["net"]
        if not os.path.exists(net_file):
            print(f"❌ Missing network file: {net_file}")
            continue

        pair, dist = find_longest_pair(net_file)
        if pair:
            print(f"✅ {area['name']}: {pair[0]} → {pair[1]} ({dist:.1f} m)")
            config[area["name"]] = {
                "start_edge": pair[0],
                "goal_edge": pair[1]
            }
        else:
            print(f"⚠️ No valid pair found for {area['name']}")

    # Save to Python config
    with open("start_goal_config.py", "w") as f:
        f.write("START_GOAL = ")
        json.dump(config, f, indent=4)
    print("\n📄 start_goal_config.py created.")
