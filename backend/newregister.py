
#!/usr/bin/env python3
"""
Driver Manager - single file
Buttons: white pill with gold border (normal) -> glossy red on hover
All functions integrated: register, show, delete, rename, recognize
"""

import os, sys, json, uuid, time, threading, re, subprocess, shutil
from typing import Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Optional libs (graceful fallback)
try:
    from deepface import DeepFace
except Exception as e:
    print("[Warning] DeepFace import failed:", e)
    DeepFace = None

try:
    import mediapipe as mp
except Exception as e:
    print("[Warning] Mediapipe import failed:", e)
    mp = None

try:
    import pyttsx3
    engine = pyttsx3.init()
except Exception as e:
    engine = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# -------------------------
# Config / storage
# -------------------------
DATA_FILE = "faces.json"
FACES_DIR = "faces"
DRIVER_PROFILES_FILE = "driver_profiles.json"
os.makedirs(FACES_DIR, exist_ok=True)

CONFIG = {
    "similarity_threshold": 0.70,
    "camera_index": 0,
    "recognition_submit_interval": 0.8,
    "result_display_time": 3.0
}

# -------------------------
# audio helper
# -------------------------
def speak(text: str):
    if engine is None:
        print("[speak]", text)
        return
    try:
        threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()
    except Exception as e:
        print("[Audio Error]", e)

# -------------------------
# json helpers
# -------------------------
def load_data() -> Dict[str, Any]:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                s = f.read().strip()
                return json.loads(s) if s else {}
        except Exception as e:
            print("[load_data] warning:", e)
            return {}
    return {}

def save_data(data: Dict[str, Any]):
    out = {}
    for uid, info in data.items():
        emb = info.get("embedding")
        if isinstance(emb, np.ndarray):
            emb_to_save = emb.tolist()
        else:
            emb_to_save = emb
        out[uid] = {"name": info.get("name"), "embedding": emb_to_save, "image": info.get("image")}
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def load_profiles() -> Dict[str, Any]:
    if os.path.exists(DRIVER_PROFILES_FILE):
        try:
            with open(DRIVER_PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_profiles(profiles: Dict[str, Any]):
    with open(DRIVER_PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4)

# -------------------------
# Face capture & pose (MediaPipe)
# -------------------------
mp_face_mesh = mp.solutions.face_mesh if mp is not None else None

def get_head_pose(image, face_mesh):
    if face_mesh is None:
        return None
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    idxs = [1,4,33,133,362,263,61,291,199,152,105,334,50,280]
    try:
        pts2 = np.array([[int(landmarks.landmark[i].x*w), int(landmarks.landmark[i].y*h)] for i in idxs], dtype=np.float64)
    except Exception:
        return None
    pts3 = np.array([[0.0,0.0,0.0],[0.0,-10.0,-5.0],[-30.0,-40.0,-30.0],[-10.0,-40.0,-30.0],
                     [10.0,-40.0,-30.0],[30.0,-40.0,-30.0],[-25.0,-70.0,-20.0],[25.0,-70.0,-20.0],
                     [0.0,-100.0,-30.0],[0.0,-120.0,-20.0],[-20.0,-60.0,-20.0],[20.0,-60.0,-20.0],
                     [0.0,-50.0,-20.0],[0.0,-80.0,-20.0]])
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1), dtype=np.float64)
    try:
        success, rvec, tvec = cv2.solvePnP(pts3, pts2, cam, dist)
    except Exception:
        return None
    if not success:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _,_,_,_,_,_,euler = cv2.decomposeProjectionMatrix(proj)
    return euler.flatten()

def capture_faces(angles=["front","left","right","up","down"], user_id="unknown"):
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    os.makedirs(FACES_DIR, exist_ok=True)
    embeddings = []
    if mp_face_mesh is None:
        cap.release()
        raise RuntimeError("MediaPipe Face Mesh not available")
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        for angle in angles:
            speak(f"Please look {angle}")
            captured = False
            while not captured:
                ret, frame = cap.read()
                if not ret:
                    continue
                pose = get_head_pose(frame, face_mesh)
                if pose is not None:
                    pitch, yaw, roll = pose
                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Face: {angle.upper()}", (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                cv2.imshow("Face Capture - Press 'c' to capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("c"):
                    try:
                        if DeepFace is None:
                            raise RuntimeError("DeepFace not available for embedding.")
                        emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                        embeddings.append(np.array(emb, dtype=np.float32))
                        captured = True
                        if angle == "front":
                            filename = os.path.join(FACES_DIR, f"{user_id}.jpg")
                            cv2.imwrite(filename, frame)
                    except Exception as e:
                        print("Capture error:", e)
                        messagebox.showerror("Capture Error", f"{e}")
    cap.release()
    cv2.destroyAllWindows()
    if not embeddings:
        raise RuntimeError("No embeddings captured")
    if len(embeddings) > 1:
        if KMeans is None:
            clustered = np.mean(embeddings, axis=0)
        else:
            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(embeddings)
            clustered = kmeans.cluster_centers_[0]
    else:
        clustered = embeddings[0]
    return clustered

# -------------------------
# Registration
# -------------------------
def register_user():
    data = load_data()
    profiles = load_profiles()
    name = simpledialog.askstring("Register", "Enter user name:")
    if not name:
        return
    mobile_pattern = re.compile(r"^\d{10}$")
    mobile_number = None
    for attempt in range(3):
        input_num = simpledialog.askstring("Register", f"Enter 10-digit mobile number for {name}:")
        if input_num is None:
            messagebox.showwarning("Warning", "Mobile number registration cancelled.")
            return
        if mobile_pattern.match(input_num):
            mobile_number = input_num
            break
        else:
            messagebox.showerror("Error", f"Invalid mobile number: '{input_num}'.")
            if attempt < 2:
                speak("Invalid mobile number. Please try again.")
            else:
                messagebox.showerror("Error", "Too many invalid attempts. Registration aborted.")
                speak("Registration aborted.")
                return
    if mobile_number is None:
        return
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera"); speak("Cannot open camera"); return
    speak("Look into the camera and press 'c' to capture for duplicate face check.")
    emb_np = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Face Check - Press 'c' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("c"):
            try:
                if DeepFace is None:
                    raise RuntimeError("DeepFace not available.")
                emb = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                emb_np = np.array(emb, dtype=np.float32)
            except Exception as e:
                messagebox.showerror("Error", f"Face not detected, try again. ({e})")
                speak("Face not detected. Please try again.")
            break
    cap.release(); cv2.destroyAllWindows()
    if emb_np is None:
        return
    threshold = CONFIG.get("similarity_threshold", 0.70)
    for uid, info in data.items():
        try:
            db_emb = np.array(info.get("embedding"), dtype=np.float32)
            sim = float(np.dot(emb_np, db_emb) / (np.linalg.norm(emb_np) * np.linalg.norm(db_emb) + 1e-8))
            if sim >= threshold:
                if info['name'].lower() == name.lower():
                    messagebox.showwarning("Registration blocked", f"'{name}' is already registered with this face.")
                    speak("Registration cancelled. This person is already registered.")
                    return
                else:
                    messagebox.showwarning("Registration blocked", "This face is already registered under a different name.")
                    speak("Registration cancelled. This face is already registered.")
                    return
        except Exception:
            continue
    temp_id = "tmp_" + str(int(time.time()))
    try:
        from_face = capture_faces(user_id=temp_id)
    except Exception as e:
        messagebox.showerror("Error", f"Face capture failed: {e}"); speak("Face capture failed"); return
    new_id = str(uuid.uuid4())[:10]
    tmp_img = os.path.join(FACES_DIR, f"{temp_id}.jpg")
    new_img_path = os.path.join(FACES_DIR, f"{new_id}.jpg")
    if os.path.exists(tmp_img):
        try: os.replace(tmp_img, new_img_path)
        except Exception:
            try: shutil.copy(tmp_img, new_img_path)
            except Exception: pass
    data[new_id] = {"name": name, "embedding": np.array(from_face, dtype=np.float32).tolist(), "image": new_img_path}
    save_data(data)
    profiles[name] = {"mobile": mobile_number, "uid": new_id}
    save_profiles(profiles)
    messagebox.showinfo("Success", f"Registered {name} (id: {new_id}) with mobile: {mobile_number}")
    speak("Registration successful")

# -------------------------
# Show users
# -------------------------
def show_all_users():
    win = tk.Toplevel(root)
    win.title("All Users"); win.geometry("600x420")
    users_face_data = load_data(); users_profiles = load_profiles()
    search_var = tk.StringVar(); tk.Label(win, text="Search user:").pack(anchor="w", padx=8, pady=(6,0))
    search_entry = tk.Entry(win, textvariable=search_var); search_entry.pack(fill=tk.X, padx=8)
    listbox = tk.Listbox(win, width=80, height=15); listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0), pady=6)
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview); scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=6)
    listbox.config(yscrollcommand=scrollbar.set); label_img = tk.Label(win, text="Select user"); label_img.pack(pady=5)
    users_list = list(users_face_data.items())
    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            profile = users_profiles.get(info['name'], {})
            mobile = profile.get('mobile', 'N/A')
            display_text = f"{uid}: {info['name']} (Mobile: {mobile})"
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, display_text)
    update_listbox()
    def on_search(event): update_listbox(search_var.get())
    search_entry.bind("<KeyRelease>", on_search)
    def on_select(event):
        sel = listbox.curselection()
        if not sel: return
        uid = listbox.get(sel[0]).split(":")[0]; user = users_face_data.get(uid, {})
        img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150,150)); img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk, text=""); label_img.image = img_tk
        else: label_img.config(image="", text="No image available")
    listbox.bind("<<ListboxSelect>>", on_select)

# -------------------------
# Delete user
# -------------------------
def delete_user():
    data = load_data(); profiles = load_profiles()
    if not data:
        messagebox.showwarning("Warning", "No users exist. Please register first."); return
    win = tk.Toplevel(root); win.title("Delete User"); win.geometry("540x420")
    search_var = tk.StringVar(); tk.Label(win, text="Search user:").pack(anchor="w", padx=8, pady=(6,0))
    search_entry = tk.Entry(win, textvariable=search_var); search_entry.pack(fill=tk.X, padx=8)
    listbox = tk.Listbox(win, width=60, height=15, selectmode=tk.EXTENDED); listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(8,0), pady=6)
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview); scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=6)
    listbox.config(yscrollcommand=scrollbar.set); label_img = tk.Label(win, text="Select user(s) to delete"); label_img.pack(pady=5)
    users_list = list(data.items())
    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, f"{uid}: {info['name']}")
    update_listbox()
    def on_search(event): update_listbox(search_var.get())
    search_entry.bind("<KeyRelease>", on_search)
    def on_select(event):
        sel = listbox.curselection()
        if not sel: label_img.config(image="", text="Select user(s)"); return
        uid = listbox.get(sel[-1]).split(":")[0]; user = data.get(uid, {})
        img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150,150)); img_tk = ImageTk.PhotoImage(img)
            label_img.config(image=img_tk, text=""); label_img.image = img_tk
        else: label_img.config(image="", text="No image available")
    listbox.bind("<<ListboxSelect>>", on_select)
    def delete_selected():
        sel = listbox.curselection()
        if not sel: messagebox.showwarning("Warning", "No users selected"); return
        selected_users = [listbox.get(i) for i in sel]
        confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {len(selected_users)} user(s)?\n\n" + "\n".join(selected_users))
        if not confirm: return
        for idx in sel[::-1]:
            list_item = listbox.get(idx); uid = list_item.split(":")[0]; name = list_item.split(":")[1].strip()
            user = data.pop(uid, None)
            if user and user.get("image") and os.path.exists(user["image"]):
                try: os.remove(user["image"])
                except Exception: pass
            profiles.pop(name, None)
        save_data(data); save_profiles(profiles); update_listbox(search_var.get()); label_img.config(image="", text="User(s) deleted")
        messagebox.showinfo("Deleted", f"Deleted {len(selected_users)} user(s).")
    def delete_all():
        confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete ALL users?")
        if not confirm: return
        for uid, user in list(data.items()):
            if user.get("image") and os.path.exists(user["image"]):
                try: os.remove(user["image"])
                except Exception: pass
        data.clear(); profiles.clear(); save_data(data); save_profiles(profiles); update_listbox(); label_img.config(image="", text="All users deleted"); messagebox.showinfo("Deleted", "All users have been deleted.")
    btn_frame = tk.Frame(win); btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Delete Selected", command=delete_selected).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Delete All", command=delete_all).pack(side=tk.LEFT, padx=5)

# -------------------------
# Rename user
# -------------------------
def change_user_name():
    data = load_data()
    if not data:
        messagebox.showwarning("Warning", "No users exist. Please register first."); speak("No users exist. Please register first."); return
    win = tk.Toplevel(root); win.title("Change User Name"); win.geometry("480x380")
    search_var = tk.StringVar(); tk.Label(win, text="Search user:").pack(); search_entry = tk.Entry(win, textvariable=search_var); search_entry.pack(fill=tk.X, padx=5)
    listbox = tk.Listbox(win, width=60, height=12); listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
    scrollbar = tk.Scrollbar(win, orient=tk.VERTICAL, command=listbox.yview); scrollbar.pack(side=tk.LEFT, fill=tk.Y); listbox.config(yscrollcommand=scrollbar.set)
    label_img = tk.Label(win, text="Select a user to rename"); label_img.pack(pady=5)
    users_list = list(data.items())
    def update_listbox(filter_text=""):
        listbox.delete(0, tk.END)
        for uid, info in users_list:
            if filter_text.lower() in info["name"].lower() or filter_text.lower() in uid.lower():
                listbox.insert(tk.END, f"{uid}: {info['name']}")
    update_listbox()
    def on_search(event): update_listbox(search_var.get())
    search_entry.bind("<KeyRelease>", on_search)
    def on_select(event):
        sel = listbox.curselection()
        if not sel: return
        uid = listbox.get(sel[0]).split(":")[0]; user = data.get(uid, {}); img_path = user.get("image")
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((150,150)); img_tk = ImageTk.PhotoImage(img); label_img.config(image=img_tk, text=""); label_img.image = img_tk
        else: label_img.config(image="", text="No image available")
    listbox.bind("<<ListboxSelect>>", on_select)
    def rename_selected():
        sel = listbox.curselection()
        if not sel: messagebox.showwarning("Warning", "No user selected"); return
        uid = listbox.get(sel[0]).split(":")[0]; old_name = data[uid]["name"]
        new_name = simpledialog.askstring("Rename", f"Enter new name for {old_name}:")
        if not new_name: speak("Rename cancelled"); return
        data[uid]["name"] = new_name; save_data(data)
        profiles = load_profiles(); 
        if old_name in profiles:
            profile_data = profiles.pop(old_name); profiles[new_name] = profile_data; save_profiles(profiles)
        update_listbox(search_var.get()); messagebox.showinfo("Renamed", f"Changed name from {old_name} to {new_name}"); speak(f"Changed name from {old_name} to {new_name}")
    btn_frame = tk.Frame(win); btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Rename Selected", command=rename_selected).pack(side=tk.LEFT, padx=5)

# -------------------------
# Recognition
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def recognize_faces():
    data = load_data()
    if not data:
        messagebox.showwarning("Warning", "No registered users to recognize."); return
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open camera"); speak("Cannot open camera"); return
    welcomed = set(); start_time = time.time(); recognized = False
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,5,minSize=(80,80))
        for (x,y,w,h) in faces:
            face_crop = frame[y:y+h,x:x+w]
            try:
                if DeepFace is None:
                    raise RuntimeError("DeepFace not available.")
                emb = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                emb = np.array(emb, dtype=np.float32)
                best, best_sim = "Unknown", -1
                for uid, info in data.items():
                    db_emb = np.array(info["embedding"], dtype=np.float32)
                    sim = float(np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb) + 1e-8))
                    if sim > best_sim: best_sim, best = sim, info["name"]
                if best_sim < CONFIG["similarity_threshold"]:
                    best = "Unknown"
                if best != "Unknown" and best not in welcomed:
                    welcomed.add(best); recognized = True; speak(f"Welcome {best}."); messagebox.showinfo("Welcome", f"Welcome {best}, starting DMS")
                    try:
                        with open("recognized_driver.txt", "w") as f: f.write(best)
                    except Exception as e: print("save error", e)
                    cap.release(); cv2.destroyAllWindows()
                    try: subprocess.run(["python3","update_11_12_25.py"])
                    except Exception as e: print("update script error", e)
                    return
            except Exception as e:
                print("Recognition error:", e)
        if not recognized and (time.time() - start_time) > 10:
            messagebox.showwarning("Unknown", "No recognized faces within time limit."); speak("No recognized faces within time limit."); break
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()

# -------------------------
# Quit
# -------------------------
def quit_app():
    speak("Exiting application"); root.quit()

# -------------------------
# UI: pill-shaped white buttons with gold border -> glossy red on hover
# -------------------------
root = tk.Tk(); root.title("Driver Monitoring System"); root.geometry("1100x700"); root.configure(bg="#f5f5f5")
style = ttk.Style()
try: style.theme_use("clam")
except Exception: pass

left_frame = tk.Frame(root, width=360, bg="#ffffff", relief=tk.GROOVE); left_frame.pack(side=tk.LEFT, fill=tk.Y); left_frame.pack_propagate(False)
header = tk.Frame(left_frame, bg="#ffffff"); header.pack(pady=18)
tk.Label(header, text="SAMSAN TECHNOLOGIES", font=("Helvetica", 18, "bold"), bg="#ffffff").pack(); tk.Label(header, text="", font=("Helvetica", 10), bg="#ffffff").pack()
btn_container = tk.Frame(left_frame, bg="#ffffff"); btn_container.pack(pady=22, fill=tk.X, padx=20)

# make the pill-like button using a Frame + Label so we control border (gold) and rounded visuals
def pill_button(parent, text, command, pady_top=12, pady_bottom=18):
    outer = tk.Frame(parent, bg="#C89A00", bd=0)  # gold border color (#C89A00)
    outer.pack(fill=tk.X, pady=(pady_top, pady_bottom))
    # inner frame white (normal)
    inner = tk.Frame(outer, bg="#ffffff", bd=0)
    inner.pack(padx=4, pady=4, fill=tk.X)
    lbl = tk.Label(inner, text=text, bg="#ffffff", fg="black", font=("Helvetica", 12, "bold"))
    lbl.pack(fill=tk.X, padx=12, pady=10)
    # click binding on both inner and label
    def on_click(e=None):
        try:
            command()
        except Exception as e:
            print("button command error:", e)
    lbl.bind("<Button-1>", on_click); inner.bind("<Button-1>", on_click); outer.bind("<Button-1>", on_click)
    # hover effect -> glossy red
    def on_enter(e):
        inner.config(bg="#d32f2f")
        lbl.config(bg="#d32f2f", fg="white")
    def on_leave(e):
        inner.config(bg="#ffffff")
        lbl.config(bg="#ffffff", fg="black")
    lbl.bind("<Enter>", on_enter); inner.bind("<Enter>", on_enter); outer.bind("<Enter>", on_enter)
    lbl.bind("<Leave>", on_leave); inner.bind("<Leave>", on_leave); outer.bind("<Leave>", on_leave)
    return outer

# add buttons with desired spacing
pill_button(btn_container, "Register User", register_user, pady_top=12, pady_bottom=16)
pill_button(btn_container, "Show Users", show_all_users, pady_top=12, pady_bottom=16)
pill_button(btn_container, "Delete User", delete_user, pady_top=12, pady_bottom=16)
pill_button(btn_container, "Change User Name", change_user_name, pady_top=12, pady_bottom=16)
pill_button(btn_container, "Recognize Faces", recognize_faces, pady_top=12, pady_bottom=16)
ttk.Separator(left_frame, orient="horizontal").pack(fill=tk.X, padx=15, pady=8)
pill_button(btn_container, "Exit", quit_app, pady_top=8, pady_bottom=14)

status_frame = tk.Frame(left_frame, bg="#ffffff"); status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=12)
tk.Label(status_frame, text="", bg="#ffffff", font=("Helvetica", 9)).pack(padx=10, pady=6)

# right panel video preview
right_frame = tk.Frame(root, bg="#fafafa"); right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True); right_frame.pack_propagate(True)
title_bar = tk.Frame(right_frame, bg="#fafafa"); title_bar.pack(fill=tk.X, pady=10, padx=10)
tk.Label(title_bar, text="", font=("Helvetica", 14, "bold"), bg="#fafafa").pack(side=tk.LEFT)
video_canvas = tk.Label(right_frame, bg="#000000"); video_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0,12))

# video player class (plays file or camera)
class VideoPlayer:
    def __init__(self, canvas_widget, video_path: str, loop=True):
        self.canvas = canvas_widget; self.video_path = video_path; self.loop = loop
        self._stop = threading.Event(); self.thread = None; self.cap = None; self.frame_image = None
    def start(self):
        if self.thread and self.thread.is_alive(): return
        self._stop.clear(); self.thread = threading.Thread(target=self._play, daemon=True); self.thread.start()
    def stop(self):
        self._stop.set()
        if self.cap:
            try: self.cap.release()
            except Exception: pass
        self.cap = None
    def _play(self):
        if os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            try:
                idx = int(self.video_path); self.cap = cv2.VideoCapture(idx)
            except Exception:
                self.cap = cv2.VideoCapture(CONFIG["camera_index"])
        if not self.cap or not self.cap.isOpened():
            print("[VideoPlayer] Unable to open video source."); return
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    try: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                    except Exception: break
                else: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            canvas_w = self.canvas.winfo_width() or 800; canvas_h = self.canvas.winfo_height() or 600
            h, w, _ = frame.shape; scale = min(canvas_w / w, canvas_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(resized); self.frame_image = ImageTk.PhotoImage(img)
            try: self.canvas.after(0, lambda img=self.frame_image: self.canvas.configure(image=img))
            except Exception as e: print("[VideoPlayer] UI update error:", e); break
            fps = self.cap.get(cv2.CAP_PROP_FPS); delay = 1.0 / fps if fps and fps > 0 and fps < 1000 else 1.0 / 24.0
            if self._stop.wait(delay): break
        if self.cap:
            try: self.cap.release()
            except Exception: pass
        self.cap = None

VIDEO_FILENAME = "Samsan-Video-2.mp4"
player = VideoPlayer(video_canvas, VIDEO_FILENAME, loop=True)
player.start()

def on_close():
    try: player.stop()
    except Exception: pass
    speak("Exiting application")
    try: root.destroy()
    except Exception: pass
    try: sys.exit(0)
    except SystemExit: pass

root.protocol("WM_DELETE_WINDOW", on_close)
root.update_idletasks()
root.mainloop()
