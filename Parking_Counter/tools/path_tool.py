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

    # RTSP stream source
    cap = cv2.VideoCapture("rtsp://ctrlpark_admin:mingae123@192.168.100.168:554/stream1")
    if not cap.isOpened():
        print("Error opening RTSP stream.")
        return

    cv2.namedWindow("Path Editor")
    cv2.setMouseCallback("Path Editor", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame if needed (optional)
        frame = cv2.resize(frame, (960, 540))

        display = frame.copy()

        # Draw nodes
        for name, (x, y) in nodes.items():
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
