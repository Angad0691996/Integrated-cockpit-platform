import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import threading
import dms_subscriber
import sdv_subscriber

sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
application = socketio.WSGIApp(sio, app)


# Forward DMS events to all clients
@sio.event
def dms_event(sid, data):
    sio.emit("dms_event", data)


# Forward SDV events to all clients
@sio.event
def sdv_event(sid, data):
    sio.emit("sdv_event", data)


def start_dms():
    dms_subscriber.start_dms_stream()


def start_sdv():
    sdv_subscriber.start_sdv_stream()


if __name__ == "__main__":
    print("Starting DMS stream...")
    threading.Thread(target=start_dms, daemon=True).start()

    print("Starting SDV stream...")
    threading.Thread(target=start_sdv, daemon=True).start()

    print("Unified backend running at http://localhost:5001")
    eventlet.wsgi.server(eventlet.listen(('', 5001)), application)
