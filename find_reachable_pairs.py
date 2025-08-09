# find_reachable_pairs.py
import os
import json
import argparse
import networkx as nx
import sumolib

# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
AREAS = [
    {"name": "RakabGanj", "net": "rakabganj_cleaned.net.xml"},
    {"name": "CP2", "net": "cp_cleaned.net.xml"},
    {"name": "Safdarjung", "net": "safdarjung_cleaned.net.xml"},
    {"name": "ChandniChowk", "net": "chandnichowk_cleaned.net.xml"},
]

MIN_ROUTE_DIST = 300  # meters
INTERNAL_PREFIX = ":"  # SUMO internal edges start with ':'
DRIVABLE_VCLASS = "passenger"  # vehicle class we will consider for routing


# ----------------------------------------
# HELPERS
# ----------------------------------------

def edge_allows_vclass(edge, vclass: str) -> bool:
    """Return True if ANY lane on the edge allows the given vclass.
    If a lane has no explicit allow list, it is considered allowed unless it explicitly disallows the vclass.
    """
    try:
        for lane in edge.getLanes():
            allowed = lane.getAllowed()  # may be list or None
            disallowed = lane.getDisallowed()  # may be list or None

            if allowed is not None:
                if vclass in allowed:
                    return True
            else:
                # No explicit allow list: treat as allowed unless explicitly disallowed
                if disallowed is None or vclass not in disallowed:
                    return True
    except Exception:
        # If lane information is unavailable, be conservative and exclude
        return False

    return False


def generate_trip_file(area_name: str, start_edge: str, goal_edge: str, output_dir: str = ".", trip_id: str = "rl_agent0") -> str:
    """Write a simple trips file with one <trip> from start_edge to goal_edge.
    Returns the written file path.
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    root = Element("routes")
    SubElement(
        root,
        "trip",
        {
            "id": trip_id,
            "depart": "0",
            "from": start_edge,
            "to": goal_edge,
        },
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{area_name.lower()}_trip.trips.xml")
    tree = ElementTree(root)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path


# ----------------------------------------
# CORE LOGIC
# ----------------------------------------

def find_longest_pair(net_file: str, vclass: str = DRIVABLE_VCLASS):
    print(f"\nProcessing {net_file} ...")

    # Load network
    net = sumolib.net.readNet(net_file)

    # Build graph excluding internal edges and non-drivable edges for vclass
    G = nx.DiGraph()

    # Map from (from_node, to_node) -> list of edge_ids that are drivable
    directed_edges = {}

    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(INTERNAL_PREFIX):
            continue
        if not edge_allows_vclass(edge, vclass):
            continue

        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        length = edge.getLength()

        G.add_edge(from_node, to_node, weight=length)
        directed_edges.setdefault((from_node, to_node), []).append(edge_id)

    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return None, -1

    # Find longest shortest path (by distance) subject to MIN_ROUTE_DIST
    longest_dist = -1
    best_pair = None

    nodes = list(G.nodes)
    for src in nodes:
        try:
            lengths, paths = nx.single_source_dijkstra(G, src, weight="weight")
        except Exception:
            continue

        for dst, dist in lengths.items():
            if dist < MIN_ROUTE_DIST or dist <= longest_dist:
                continue
            if dst not in paths or len(paths[dst]) <= 1:
                continue

            path_nodes = paths[dst]

            # Extract a drivable edge sequence along the node path
            path_edges = []
            valid_path = True
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                edge_ids = directed_edges.get((u, v), [])
                if edge_ids:
                    # take first candidate (graph is already restricted to drivable edges)
                    path_edges.append(edge_ids[0])
                else:
                    valid_path = False
                    break

            if not valid_path or not path_edges:
                continue

            start_edge_id = path_edges[0]
            end_edge_id = path_edges[-1]
            if start_edge_id == end_edge_id:
                continue

            longest_dist = dist
            best_pair = (start_edge_id, end_edge_id)

    return best_pair, longest_dist


def main():
    parser = argparse.ArgumentParser(description="Find drivable start/goal edge pairs and optionally emit trips files.")
    parser.add_argument("--write-trips", action="store_true", help="Write per-area trips files for duarouter")
    parser.add_argument("--trips-dir", default=".", help="Directory to write trips files")
    parser.add_argument("--min-dist", type=float, default=MIN_ROUTE_DIST, help="Minimum route distance in meters")
    parser.add_argument("--vclass", default=DRIVABLE_VCLASS, help="Vehicle class to consider (e.g., passenger)")
    args = parser.parse_args()

    global MIN_ROUTE_DIST
    MIN_ROUTE_DIST = args.min_dist

    config = {}

    for area in AREAS:
        net_file = area["net"]
        area_name = area["name"]

        if not os.path.exists(net_file):
            print(f"❌ Missing network file: {net_file}")
            continue

        pair, dist = find_longest_pair(net_file, vclass=args.vclass)
        if pair:
            print(f"✅ {area_name}: {pair[0]} → {pair[1]} ({dist:.1f} m)")
            config[area_name] = {
                "start_edge": pair[0],
                "goal_edge": pair[1],
            }

            if args.write_trips:
                trips_path = generate_trip_file(area_name, pair[0], pair[1], output_dir=args.trips_dir)
                print(f"   📄 trips file: {trips_path}")
        else:
            print(f"⚠️ No valid drivable pair found for {area_name}")

    # Save to Python config for downstream use
    if config:
        with open("start_goal_config.py", "w") as f:
            f.write("START_GOAL = ")
            json.dump(config, f, indent=4)
        print("\n📄 start_goal_config.py created.")

        if args.write_trips:
            print("\nNext, run duarouter to create route files. Examples:")
            for area in AREAS:
                name = area["name"]
                if name not in config:
                    continue
                net = area["net"]
                trips = os.path.join(args.trips_dir, f"{name.lower()}_trip.trips.xml")
                out_rou = f"{name.lower()}_dynamic.rou.xml"
                print(
                    f"  duarouter -n {net} -t {trips} -o {out_rou} --ignore-errors true --remove-loops true --repair true"
                )
    else:
        print("No config produced. Nothing to write.")


if __name__ == "__main__":
    main()
