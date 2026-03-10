#!/usr/bin/env python3
"""
MQTT publisher for industrial inspection results.

Publishes detection results and defect alerts to configurable topics
with QoS 1 and automatic reconnection.
"""

import json
import ssl
import time
import threading
from typing import Any, Optional

import paho.mqtt.client as mqtt


class MQTTPublisher:
    """Publish inspection results and alerts to an MQTT broker.

    Usage::

        pub = MQTTPublisher(broker="localhost", port=1883)
        pub.connect()
        pub.publish("inspection/results", {"detections": [...]})
        pub.disconnect()
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        client_id: str = "jetson-inspector",
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False,
        ca_certs: Optional[str] = None,
        keepalive: int = 60,
        qos: int = 1,
        reconnect_delay_min: float = 1.0,
        reconnect_delay_max: float = 30.0,
    ):
        self.broker = broker
        self.port = port
        self.qos = qos
        self._connected = threading.Event()

        # Paho client
        self._client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )

        # Authentication
        if username:
            self._client.username_pw_set(username, password)

        # TLS
        if use_tls:
            self._client.tls_set(
                ca_certs=ca_certs,
                cert_reqs=ssl.CERT_REQUIRED if ca_certs else ssl.CERT_NONE,
                tls_version=ssl.PROTOCOL_TLS,
            )

        # Auto-reconnect parameters
        self._client.reconnect_delay_set(
            min_delay=int(reconnect_delay_min),
            max_delay=int(reconnect_delay_max),
        )

        # Callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        self._keepalive = keepalive
        self._message_count = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[MQTT] Connected to {self.broker}:{self.port}")
            self._connected.set()
        else:
            reason = mqtt.connack_string(rc)
            print(f"[MQTT] Connection failed: {reason} (rc={rc})")
            self._connected.clear()

    def _on_disconnect(self, client, userdata, rc):
        self._connected.clear()
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}). Auto-reconnecting...")
        else:
            print("[MQTT] Disconnected cleanly.")

    def _on_publish(self, client, userdata, mid):
        with self._lock:
            self._message_count += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to the broker. Returns True on success."""
        try:
            self._client.connect(self.broker, self.port, keepalive=self._keepalive)
            self._client.loop_start()
            connected = self._connected.wait(timeout=timeout)
            if not connected:
                print("[MQTT] Connection timed out.")
            return connected
        except Exception as exc:
            print(f"[MQTT] Connection error: {exc}")
            return False

    def disconnect(self):
        """Disconnect from the broker and stop the network loop."""
        self._client.loop_stop()
        self._client.disconnect()
        self._connected.clear()

    def publish(self, topic: str, payload: Any, qos: Optional[int] = None) -> bool:
        """Publish a JSON-serialisable payload to a topic.

        Args:
            topic: MQTT topic string.
            payload: Any JSON-serialisable object.
            qos: Override instance-level QoS for this message.

        Returns:
            True if the message was queued successfully.
        """
        if not self._connected.is_set():
            print("[MQTT] Not connected. Message dropped.")
            return False

        qos = qos if qos is not None else self.qos

        try:
            message = json.dumps(payload, default=str)
        except (TypeError, ValueError) as exc:
            print(f"[MQTT] Serialization error: {exc}")
            return False

        info = self._client.publish(topic, message, qos=qos)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] Publish failed on topic '{topic}': rc={info.rc}")
            return False

        return True

    def publish_result(self, result: dict) -> bool:
        """Convenience: publish to the standard results topic."""
        return self.publish("inspection/results", result)

    def publish_alert(self, alert: dict) -> bool:
        """Convenience: publish to the standard alerts topic."""
        return self.publish("inspection/alerts", alert)

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def message_count(self) -> int:
        with self._lock:
            return self._message_count
