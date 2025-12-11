import requests
import websocket
import json
from datetime import datetime
import socketio

sio = socketio.Client()
LOG_FILE = "DMS_logging.txt"

TB_HOST = "https://thingsboard.cloud"
USERNAME = "rutuja.arekar@samsanlabs.com"
PASSWORD = "Rutuja@Samsan113"
DEVICE_ID = "e40b8950-d014-11f0-8d27-9f87c351edd8"


def write_log(key, value, ts):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {key}: {value}\n")


def get_jwt_token():
    print("DMS: Logging in to ThingsBoard…")
    resp = requests.post(
        f"{TB_HOST}/api/auth/login",
        json={"username": USERNAME, "password": PASSWORD}
    )
    print("DMS: Login status:", resp.status_code)

    resp.raise_for_status()
    token = resp.json()["token"]
    print("DMS: JWT token generated OK.")
    return token


def on_message(ws, message):
    print("DMS: Raw TB message →", message)

    data = json.loads(message)
    if "data" not in data:
        return

    for key, values in data["data"].items():
        for entry in values:
            ts = datetime.fromtimestamp(entry[0] / 1000)
            val = entry[1]

            if str(val).strip() == "0":
                continue

            write_log(key, val, ts)

            sio.emit("dms_event", {
                "timestamp": str(ts),
                "key": key,
                "value": val
            })


def on_open(ws):
    print("DMS: WebSocket CONNECTED. Subscribing to telemetry…")

    sub = {
        "tsSubCmds": [{
            "entityType": "DEVICE",
            "entityId": DEVICE_ID,
            "scope": "LATEST_TELEMETRY",
            "cmdId": 1
        }],
        "historyCmds": [],
        "attrSubCmds": []
    }

    print("DMS: Sending subscribe message:", sub)
    ws.send(json.dumps(sub))


def start_dms_stream():
    print("DMS: Connecting to Socket.IO backend…")
    sio.connect("http://localhost:5001")

    print("DMS: Getting JWT token…")
    jwt = get_jwt_token()

    ws_url = f"wss://thingsboard.cloud/api/ws/plugins/telemetry?token={jwt}"
    print("DMS: Connecting to TB WebSocket…")
    print("DMS: WS URL:", ws_url)

    websocket.enableTrace(True)

    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_open=on_open,
    )
    ws.run_forever()

if __name__ == "__main__":
    start_dms_stream()

