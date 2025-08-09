# generate_dynamic_routes.py
import sys
import os
import xml.etree.ElementTree as ET
from start_goal_config import START_GOAL

def generate_route(area_name, output_dir="."):
    if area_name not in START_GOAL:
        print(f"❌ Area '{area_name}' not found in start_goal_config.py")
        sys.exit(1)

    start_edge = START_GOAL[area_name]["start_edge"]
    goal_edge = START_GOAL[area_name]["goal_edge"]

    # Output file path
    output_file = os.path.join(output_dir, f"{area_name.lower()}_dynamic.rou.xml")

    # XML structure
    root = ET.Element("routes")

    # Vehicle type
    vtype = ET.SubElement(root, "vType", {
        "id": "car",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "maxSpeed": "70",
        "color": "1,0,0"
    })

    # Vehicle with route
    veh = ET.SubElement(root, "vehicle", {
        "id": "rl_agent0",
        "type": "car",
        "depart": "0"
    })

    ET.SubElement(veh, "route", {
        "edges": f"{start_edge} {goal_edge}"
    })

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)

    print(f"✅ Dynamic route file created: {output_file}")
    print(f"   Start: {start_edge}")
    print(f"   Goal: {goal_edge}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dynamic_routes.py <AreaName> [output_dir]")
        sys.exit(1)

    area = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    generate_route(area, out_dir)
