import cv2
import json

# Global data
nodes = {}
edges = {}
selected_node = None
node_id = 0

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global node_id, selected_node

    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click = add node
        name = f"node{node_id}"
        nodes[name] = (x, y)
        edges[name] = []
        node_id += 1
        print(f"Added node {name} at {(x, y)}")

    elif event == cv2.EVENT_RBUTTONDOWN:  # Right-click = select node for connection
        for name, pos in nodes.items():
            if abs(x - pos[0]) < 10 and abs(y - pos[1]) < 10:
                if selected_node is None:
                    selected_node = name
                    print(f"Selected {name} for connection")
                else:
                    edges[selected_node].append(name)
                    print(f"Connected {selected_node} -> {name}")
                    selected_node = None

def main():
    global nodes, edges
    img = cv2.imread("C:/Users/lyzza/UR_SY2526/ANPR1/parking_slant.jpg")
    clone = img.copy()

    cv2.namedWindow("Path Editor")
    cv2.setMouseCallback("Path Editor", mouse_callback)

    while True:
        display = clone.copy()

        # Draw nodes
        for name, (x, y) in nodes.items():
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, name, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw edges
        for src, dests in edges.items():
            for dst in dests:
                cv2.line(display, nodes[src], nodes[dst], (255, 0, 0), 2)

        cv2.imshow("Path Editor", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save
            with open("paths.json", "w") as f:
                json.dump({"nodes": nodes, "edges": edges}, f, indent=4)
            print("Saved paths.json")
        elif key == ord('q'):  # Quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
