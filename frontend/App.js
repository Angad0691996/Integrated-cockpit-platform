import React, { useEffect, useState } from "react";
import { io } from "socket.io-client";

function App() {
  const [dmsEvents, setDmsEvents] = useState([]);
  const [sdvEvents, setSdvEvents] = useState([]);

  useEffect(() => {
    const socket = io("http://localhost:5001");

    socket.on("dms_event", (data) => {
      setDmsEvents((prev) => [data, ...prev]);
    });

    socket.on("sdv_event", (data) => {
      setSdvEvents((prev) => [data, ...prev]);
    });

    return () => socket.disconnect();
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>DMS Real-Time Events</h2>
      {dmsEvents.map((e, i) => (
        <div key={i}>{e.timestamp} — {e.key}: {e.value}</div>
      ))}

      <h2 style={{ marginTop: 40 }}>SDV Mobile Events</h2>
      {sdvEvents.map((e, i) => (
        <div key={i}>{e.timestamp} — {e.topic}: {JSON.stringify(e.data)}</div>
      ))}
    </div>
  );
}

export default App;
