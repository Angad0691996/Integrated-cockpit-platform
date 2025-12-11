import socketio

sio = socketio.Client()

@sio.event
def dms_event(data):
    print("DMS EVENT:", data)

@sio.event
def sdv_event(data):
    print("SDV EVENT:", data)

sio.connect("http://localhost:5001")
print("Connected. Waiting for events...")

sio.wait()
