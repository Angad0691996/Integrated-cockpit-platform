import paho.mqtt.client as mqtt
import ssl
import json
from datetime import datetime
import socketio
import pytz
sio = socketio.Client()



LOG_FILE = "logs/SDV_mobile_logs.txt"

awshost = "a1w5rvqhps48wc-ats.iot.ap-south-1.amazonaws.com"
awsport = 8883
clientId = "iotconsole-your-id"
caPath = "AmazonRootCA1.pem"
certPath = "certificate.pem.crt"
keyPath = "private.pem.key"


def write_log(topic, data):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] Topic: {topic} | Data: {data}\n")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to AWS IoT")
        client.subscribe("carDoor/status", 1)
        client.subscribe("HVAC/Temp", 1)
    else:
        print("Failed:", rc)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except:
        return

    write_log(msg.topic, payload)

    # Emit to dashboard backend
    sio.emit("sdv_event", {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topic": msg.topic,
        "data": payload
    })


def start_sdv_stream():
    sio.connect("http://localhost:5001")

    mqttc = mqtt.Client(client_id=clientId)
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message

    mqttc.tls_set(
        ca_certs=caPath,
        certfile=certPath,
        keyfile=keyPath,
        tls_version=ssl.PROTOCOL_TLSv1_2,
    )

    mqttc.connect(awshost, awsport, 60)
    mqttc.loop_forever()


if __name__ == "__main__":
    start_sdv_stream()
