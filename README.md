<<<<<<< HEAD
# 🚗 SDV + DMS Real-Time Telemetry Dashboard  
A unified cockpit logging and visualization system that streams **SDV (vehicle controls)** and **DMS (driver monitoring system)** events in real-time using Python backend + React frontend.

---

# 📦 Project Overview

This system processes **two independent data sources**:

### ✅ SDV (Signal Data Visualization)
- Logs events sent from the **mobile application**
- Uses AWS IoT Core MQTT subscriber
- Processes signals like: TPMS, LDW, FCW, Door Locks, etc.
- Sent to frontend via WebSocket as **sdv_event**

### ✅ DMS (Driver Monitoring System)
- Streams live telemetry from **ThingsBoard Cloud**
- Listens over WebSocket telemetry API
- Filters & logs only **non-zero events**
- Sends events to frontend as **dms_event**

Both streams are forwarded through a single Python backend running **Flask + Socket.IO**, and the React dashboard listens in real-time.

---

# 📁 Folder Structure

```
cockpit_dashboard/
│
├── backend/
│   ├── server.py             # Unified Flask backend + WebSocket bridge
│   ├── sdv_logging.py        # AWS IoT subscriber (SDV)
│   ├── dms_subscriber.py     # ThingsBoard WebSocket subscriber (DMS)
│   ├── SDV_mobile_logs.txt   # SDV log storage
│   ├── DMS_logging.txt       # DMS log storage
│   ├── .venv/                # Python environment
│
└── frontend/
    └── dms-dashboard/        # React UI
        ├── src/
        │   ├── components/
        │   │   ├── DmsCard.js
        │   │   ├── SdvCard.js
        │   │   ├── RealtimeChart.js
        │   │   └── EventList.js
        │   ├── App.js        # Main UI layout
        │   ├── styles.css    # Dashboard theme
        ├── package.json
```

---

# ⚙️ Backend Setup (Python)

### 1️⃣ Create & activate venv
```
cd backend
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
```

### 2️⃣ Run the unified backend
```
python server.py
```

You should see:
```
Unified backend running at http://localhost:5001
Starting SDV stream...
Starting DMS stream...
```

### 3️⃣ In two new terminals, run subscribers

#### SDV subscriber:
```
python sdv_logging.py
```

#### DMS subscriber:
```
python dms_subscriber.py
```

You will see:
```
DMS: WebSocket CONNECTED
DMS: Raw TB message →
SDV: Connected to AWS IoT
```

---

# ⚙️ Frontend Setup (React)

### 1️⃣ Install dependencies
```
cd frontend/dms-dashboard
npm install
```

### 2️⃣ Start React dashboard
```
npm start
```

Runs at:
```
http://localhost:3000
```

---

# 🔌 How Real-Time Flow Works

```
 AWS IoT Core --> sdv_logging.py ---------                                                                                       --> server.py --> React Dashboard
                                           /
ThingsBoard Cloud --> dms_subscriber.py --/
```

### Backend pushes:
```
socketio.emit("sdv_event", {...})
socketio.emit("dms_event", {...})
```

React listens via:
```js
socket.on("sdv_event", handler)
socket.on("dms_event", handler)
```

---

# 📊 Dashboard Features

| Feature | SDV | DMS |
|--------|-----|-----|
| Live event streaming | ✅ | ✅ |
| Auto-updating charts | ❌ (optional) | ✅ |
| Event log history | ✅ | ✅ |
| Filters | Planned | Planned |

---

# 📝 Log Files

Stored in backend directory:

| File | Description |
|------|-------------|
| `SDV_mobile_logs.txt` | Every signal change from mobile |
| `DMS_logging.txt` | Only **non-zero** DMS events (filtered automatically) |

---

# 🚀 Running the Entire System (Full Workflow)

### 1️⃣ Start backend
```
python server.py
```

### 2️⃣ Start SDV subscriber
```
python sdv_logging.py
```

### 3️⃣ Start DMS subscriber
```
python dms_subscriber.py
```

### 4️⃣ Run React dashboard
```
npm start
```

You now have a **real-time cockpit dashboard** showing:

- Vehicle signal telemetry ✔  
- Driver monitoring alerts ✔  
- Time‑stamped logs ✔  
- Live updating UI ✔  

---

# 🧩 Troubleshooting

### ❌ React fails with "`react-scripts` not found"
Fix:
```
rm -r node_modules package-lock.json
npm install
```

### ❌ Backend not receiving SDV events
Check your AWS IoT credentials inside `sdv_logging.py`

### ❌ Backend not receiving DMS events
Verify:
- JWT token is generated
- ThingsBoard device ID is correct
- Telemetry keys exist

### ❌ Dashboard empty
Make sure backend (`server.py`) is running before starting React.

---

# ✔ Future Enhancements

- Real-time charts for all DMS keys  
- SDV trip summary screen  
- Dashboard theming (dark/light mode)  
- Docker-compose deployment  
- Cloud Grafana integration  

---

# 👤 Author

**Angad Bandal**  
IoT + Edge + DevOps Engineer  
AWS | Azure | MQTT | Python | React Dashboards  

---

# ✅ End of README
=======
# Integrated-cockpit-platform
The repo contains the SDV as in umbrella factors as SDV and DMS logs starting with version1
>>>>>>> 51719bbf3d36c6e6e5f1f6e46ff9563d0f54a40a
