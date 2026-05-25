"use client";

import { useEffect, useRef, useState } from "react";

// Kakao Maps SDK 타입은 외부 전역. 최소한으로만 선언.
declare global {
  interface Window {
    kakao?: KakaoNamespace;
  }
}
interface KakaoNamespace {
  maps: {
    load: (cb: () => void) => void;
    Map: new (el: HTMLElement, opts: unknown) => KakaoMapInstance;
    LatLng: new (lat: number, lng: number) => unknown;
    Marker: new (opts: unknown) => KakaoMarker;
    Polyline: new (opts: unknown) => { setMap: (m: unknown) => void };
    event: { addListener: (t: unknown, type: string, cb: () => void) => void };
  };
}
interface KakaoMapInstance {
  setCenter: (latlng: unknown) => void;
  setLevel: (level: number) => void;
}
interface KakaoMarker {
  setMap: (m: unknown) => void;
}

export interface MapMarker {
  id: string;
  lat: number;
  lng: number;
  title: string;
}

const JS_KEY = process.env.NEXT_PUBLIC_KAKAO_MAP_JS_KEY;
const JEJU_CENTER = { lat: 33.3846, lng: 126.5535 };

let sdkPromise: Promise<void> | null = null;

function loadSdk(): Promise<void> {
  if (!JS_KEY) return Promise.reject(new Error("no-key"));
  if (typeof window === "undefined") return Promise.reject(new Error("no-window"));
  if (window.kakao?.maps) return Promise.resolve();
  if (sdkPromise) return sdkPromise;

  sdkPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${JS_KEY}&autoload=false`;
    script.async = true;
    script.onload = () => {
      window.kakao!.maps.load(() => resolve());
    };
    script.onerror = () => reject(new Error("sdk-load-failed"));
    document.head.appendChild(script);
  });
  return sdkPromise;
}

export function KakaoMap({
  markers,
  polyline,
  onMarkerClick,
  className,
}: {
  markers: MapMarker[];
  polyline?: { lat: number; lng: number }[];
  onMarkerClick?: (id: string) => void;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<KakaoMapInstance | null>(null);
  const markerObjsRef = useRef<KakaoMarker[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "no-key" | "error">(
    JS_KEY ? "loading" : "no-key"
  );

  // SDK 로드 + 지도 생성
  useEffect(() => {
    if (!JS_KEY) {
      setStatus("no-key");
      return;
    }
    let cancelled = false;
    loadSdk()
      .then(() => {
        if (cancelled || !containerRef.current || !window.kakao) return;
        const kakao = window.kakao;
        const map = new kakao.maps.Map(containerRef.current, {
          center: new kakao.maps.LatLng(JEJU_CENTER.lat, JEJU_CENTER.lng),
          level: 9,
        });
        mapRef.current = map;
        setStatus("ready");
      })
      .catch(() => !cancelled && setStatus("error"));
    return () => {
      cancelled = true;
    };
  }, []);

  // 마커 렌더링
  useEffect(() => {
    if (status !== "ready" || !window.kakao || !mapRef.current) return;
    const kakao = window.kakao;

    markerObjsRef.current.forEach((m) => m.setMap(null));
    markerObjsRef.current = [];

    markers.forEach((mk) => {
      const marker = new kakao.maps.Marker({
        position: new kakao.maps.LatLng(mk.lat, mk.lng),
        title: mk.title,
      });
      marker.setMap(mapRef.current);
      if (onMarkerClick) {
        kakao.maps.event.addListener(marker, "click", () => onMarkerClick(mk.id));
      }
      markerObjsRef.current.push(marker);
    });

    if (markers.length > 0) {
      mapRef.current.setCenter(
        new kakao.maps.LatLng(markers[0].lat, markers[0].lng)
      );
    }
  }, [markers, status, onMarkerClick]);

  // 폴리라인(여행경로)
  useEffect(() => {
    if (status !== "ready" || !window.kakao || !mapRef.current || !polyline?.length)
      return;
    const kakao = window.kakao;
    const path = polyline.map((p) => new kakao.maps.LatLng(p.lat, p.lng));
    const line = new kakao.maps.Polyline({
      path,
      strokeWeight: 4,
      strokeColor: "#0d9488",
      strokeOpacity: 0.8,
      strokeStyle: "solid",
    });
    line.setMap(mapRef.current);
    return () => line.setMap(null);
  }, [polyline, status]);

  if (status === "no-key" || status === "error") {
    return (
      <div
        className={`flex flex-col items-center justify-center gap-2 bg-jeju-light/40 p-6 text-center ${className ?? ""}`}
      >
        <span className="text-3xl">🗺️</span>
        <p className="text-sm font-medium text-slate-600">
          {status === "error" ? "지도를 불러오지 못했습니다." : "지도 미리보기"}
        </p>
        <p className="max-w-xs text-xs text-slate-400">
          {status === "error"
            ? "Kakao 지도 SDK 로드에 실패했습니다. JS 키와 도메인 설정을 확인하세요."
            : "NEXT_PUBLIC_KAKAO_MAP_JS_KEY 를 설정하면 카카오 지도가 표시됩니다. 키가 없어도 아래 목록은 정상 동작합니다."}
        </p>
      </div>
    );
  }

  return (
    <div className={`relative ${className ?? ""}`}>
      <div ref={containerRef} className="h-full w-full" />
      {status === "loading" ? (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-100/60 text-sm text-slate-400">
          지도 로딩 중…
        </div>
      ) : null}
    </div>
  );
}

/** 카카오맵 길찾기 링크 (도착지 좌표/이름). */
export function kakaoDirectionsUrl(name: string, lat: number, lng: number) {
  return `https://map.kakao.com/link/to/${encodeURIComponent(name)},${lat},${lng}`;
}
