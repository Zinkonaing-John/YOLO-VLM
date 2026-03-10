"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export interface Defect {
  defect_class: string;
  confidence: number;
  bbox_x1: number;
  bbox_y1: number;
  bbox_x2: number;
  bbox_y2: number;
  delta_e?: number;
  vlm_description?: string;
}

export interface InspectionResult {
  id: string;
  timestamp: string;
  verdict: "PASS" | "FAIL";
  defects: Defect[];
  processing_ms: number;
  image_path?: string;
  image_url?: string;
  total_defects: number;
  vlm_response?: string;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

const MAX_RESULTS = 100;
const MAX_RECONNECT_DELAY = 30000;
const BASE_RECONNECT_DELAY = 1000;

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_URL = API_URL.replace(/^http/, "ws");

export function useInspection() {
  const [results, setResults] = useState<InspectionResult[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus("connecting");

    try {
      const ws = new WebSocket(`${WS_URL}/ws/inspection`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("connected");
        reconnectAttemptRef.current = 0;
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const data: InspectionResult = JSON.parse(event.data);
          setResults((prev) => {
            const updated = [data, ...prev];
            return updated.slice(0, MAX_RESULTS);
          });
        } catch {
          console.error("Failed to parse inspection result");
        }
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("disconnected");
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        setConnectionStatus("error");
        ws.close();
      };
    } catch {
      setConnectionStatus("error");
      scheduleReconnect();
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return;

    const delay = Math.min(
      BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttemptRef.current),
      MAX_RECONNECT_DELAY
    );
    reconnectAttemptRef.current += 1;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    reconnectTimerRef.current = setTimeout(() => {
      if (mountedRef.current) {
        connect();
      }
    }, delay);
  }, [connect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { results, connectionStatus };
}
