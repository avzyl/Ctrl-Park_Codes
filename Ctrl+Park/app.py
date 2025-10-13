import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import webbrowser
import sys
import os
from datetime import datetime

# ---------------- BASE DIR ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARKING_SCRIPT = os.path.join(BASE_DIR, "parking_monitor.py")
SCANNER_SCRIPT = os.path.join(BASE_DIR, "plate_scanner.py")
CLOUDFLARE_TUNNEL = "cloudflared"

# ---------------- Remote URLs ----------------
PARKING_REMOTE_URL = "https://stream.bsitport2026.com/parking/video_feed"
SCANNER_REMOTE_URL = "https://stream.bsitport2026.com/scanner/video_feed"

# ---------------- Globals ----------------
parking_process = None
scanner_process = None
tunnel_process = None

# ---------------- Tkinter Setup ----------------
root = tk.Tk()
root.title("Ctrl+Park Remote Controller")
root.configure(bg="#121212")
root.geometry("1100x700")  # windowed size (no fullscreen)
root.resizable(True, True)

# ---------------- Logging ----------------
def log_message(msg, tag="info"):
    timestamp = datetime.now().strftime("[%H:%M:%S] ")
    log_area.insert(tk.END, timestamp + msg + "\n", tag)
    log_area.see(tk.END)

# ---------------- Status Lights ----------------
def update_status(label, running):
    label.config(bg="#00FF55" if running else "#FF3333")

# ---------------- Helper Functions ----------------
def start_parking():
    global parking_process
    if parking_process is None or parking_process.poll() is not None:
        parking_process = subprocess.Popen([sys.executable, PARKING_SCRIPT])
        log_message("Parking monitor started.", "parking")
        update_status(parking_status, True)
    else:
        log_message("Parking monitor already running.", "info")

def stop_parking():
    global parking_process
    if parking_process and parking_process.poll() is None:
        parking_process.terminate()
        parking_process.wait()
        log_message("Parking monitor stopped.", "parking")
        update_status(parking_status, False)

def start_scanner():
    global scanner_process
    if scanner_process is None or scanner_process.poll() is not None:
        scanner_process = subprocess.Popen([sys.executable, SCANNER_SCRIPT])
        log_message("Plate scanner started.", "scanner")
        update_status(scanner_status, True)
    else:
        log_message("Plate scanner already running.", "info")

def stop_scanner():
    global scanner_process
    if scanner_process and scanner_process.poll() is None:
        scanner_process.terminate()
        scanner_process.wait()
        log_message("Plate scanner stopped.", "scanner")
        update_status(scanner_status, False)

def start_tunnel():
    global tunnel_process
    if tunnel_process is None or tunnel_process.poll() is not None:
        tunnel_process = subprocess.Popen([CLOUDFLARE_TUNNEL, "tunnel", "run", "ctrlpark-tunnel"])
        log_message("Cloudflare tunnel started.", "tunnel")
        update_status(tunnel_status, True)
    else:
        log_message("Tunnel already running.", "info")

def stop_tunnel():
    global tunnel_process
    if tunnel_process and tunnel_process.poll() is None:
        tunnel_process.terminate()
        tunnel_process.wait()
        log_message("Cloudflare tunnel stopped.", "tunnel")
        update_status(tunnel_status, False)

# ---------------- Open Remote in Browser ----------------
def open_parking_remote():
    log_message("Opening Parking Monitor in browser...", "parking")
    webbrowser.open_new_tab(PARKING_REMOTE_URL)

def open_scanner_remote():
    log_message("Opening Plate Scanner in browser...", "scanner")
    webbrowser.open_new_tab(SCANNER_REMOTE_URL)

# ---------------- Exit Confirmation ----------------
def confirm_exit():
    if messagebox.askyesno("Exit Ctrl+Park", "Are you sure you want to exit the system?"):
        stop_parking()
        stop_scanner()
        stop_tunnel()
        root.destroy()

# ---------------- UI Layout ----------------
title_label = tk.Label(
    root,
    text="Ctrl+Park Remote Controller",
    font=("Segoe UI", 20, "bold"),
    fg="#00FFFF",
    bg="#121212"
)
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#121212")
frame.pack(expand=True)

def create_module(row, name, color, start_cmd, stop_cmd, open_cmd, status_ref, show_remote=True):
    tk.Label(frame, text=name, fg=color, bg="#121212", font=("Segoe UI", 12, "bold")).grid(row=row, column=0, padx=10, pady=5)
    tk.Button(frame, text="Start", command=start_cmd, width=10, bg=color, fg="black").grid(row=row, column=1, padx=5)
    tk.Button(frame, text="Stop", command=stop_cmd, width=10, bg="#880000", fg="white").grid(row=row, column=2, padx=5)
    if show_remote:
        tk.Button(frame, text="Open Remote", command=open_cmd, width=15, bg="#333333", fg="white").grid(row=row, column=3, padx=5)
    else:
        tk.Label(frame, text=" ", bg="#121212").grid(row=row, column=3, padx=5)
    status_light = tk.Label(frame, width=2, height=1, bg="#FF3333", relief="sunken")
    status_light.grid(row=row, column=4, padx=10)
    status_ref.append(status_light)

# status references
parking_status = []
scanner_status = []
tunnel_status = []

create_module(0, "Parking Monitor", "#00FFAA", start_parking, stop_parking, open_parking_remote, parking_status, show_remote=True)
create_module(1, "Plate Scanner", "#FFDD00", start_scanner, stop_scanner, open_scanner_remote, scanner_status, show_remote=True)
create_module(2, "Cloudflare Tunnel", "#00A2FF", start_tunnel, stop_tunnel, lambda: None, tunnel_status, show_remote=False)

# Convert to single label for each
parking_status = parking_status[0]
scanner_status = scanner_status[0]
tunnel_status = tunnel_status[0]

# ---------------- Log Area ----------------
log_label = tk.Label(root, text="System Monitor", font=("Segoe UI", 14, "bold"), fg="#00FFBB", bg="#121212")
log_label.pack(pady=(15, 0))

log_area = scrolledtext.ScrolledText(root, width=130, height=18, wrap=tk.WORD, bg="#1E1E1E", fg="white", insertbackground="white", font=("Consolas", 10))
log_area.pack(padx=20, pady=10)

log_area.tag_config("info", foreground="#00DDFF")
log_area.tag_config("parking", foreground="#00FFAA")
log_area.tag_config("scanner", foreground="#FFDD55")
log_area.tag_config("tunnel", foreground="#00A2FF")
log_area.tag_config("error", foreground="#FF5555", font=("Consolas", 10, "bold"))

# ---------------- Exit Button ----------------
exit_button = tk.Button(root, text="Exit Application", command=confirm_exit, bg="#FF3333", fg="white", font=("Segoe UI", 12, "bold"), width=20)
exit_button.pack(pady=15)

# ---------------- Main Loop ----------------
root.mainloop()
