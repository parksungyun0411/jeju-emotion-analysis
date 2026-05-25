"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { KakaoMap, kakaoDirectionsUrl, type MapMarker } from "@/components/KakaoMap";
import { BottomSheet } from "@/components/BottomSheet";
import type { Place } from "@/lib/types";

export default function FoodPage() {
  const [places, setPlaces] = useState<Place[]>([]);
  const [source, setSource] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Place | null>(null);

  const fetchPlaces = useCallback(async (q: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/places?query=${encodeURIComponent(q)}`);
      const data = await res.json();
      setPlaces(data.places ?? []);
      setSource(data.source ?? "");
    } catch {
      setPlaces([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchPlaces("");
  }, [fetchPlaces]);

  const markers: MapMarker[] = useMemo(
    () =>
      places.map((p) => ({ id: p.id, lat: p.lat, lng: p.lng, title: p.name })),
    [places]
  );

  const onMarkerClick = useCallback(
    (id: string) => {
      const p = places.find((x) => x.id === id);
      if (p) setSelected(p);
    },
    [places]
  );

  return (
    <div>
      <PageHeader title="맛집" subtitle="제주 맛집 지도" />

      {/* 검색 */}
      <div className="sticky top-[57px] z-20 bg-slate-50 p-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            fetchPlaces(query);
          }}
          className="flex gap-2"
        >
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="맛집 검색 (예: 흑돼지, 국수)"
            className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-jeju"
          />
          <button className="rounded-lg bg-jeju px-4 text-sm font-semibold text-white">
            검색
          </button>
        </form>
      </div>

      {/* 지도 */}
      <KakaoMap
        markers={markers}
        onMarkerClick={onMarkerClick}
        className="h-56 w-full"
      />

      {/* 목록 */}
      <div className="p-3">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-600">
            {loading ? "불러오는 중…" : `맛집 ${places.length}곳`}
          </p>
          {source === "seed" ? (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-700">
              seed 데이터
            </span>
          ) : source === "kakao" ? (
            <span className="rounded-full bg-jeju-light px-2 py-0.5 text-[10px] font-medium text-jeju-dark">
              Kakao Local
            </span>
          ) : null}
        </div>

        <ul className="space-y-2">
          {places.map((p) => (
            <li key={p.id}>
              <button
                onClick={() => setSelected(p)}
                className="w-full rounded-xl bg-white p-3 text-left shadow-sm active:bg-slate-50"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-semibold text-slate-800">{p.name}</p>
                    <p className="text-xs text-slate-500">{p.category}</p>
                    <p className="mt-0.5 text-xs text-slate-400">{p.address}</p>
                  </div>
                  {typeof p.rating === "number" ? (
                    <span className="shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">
                      ⭐ {p.rating}
                    </span>
                  ) : null}
                </div>
              </button>
            </li>
          ))}
        </ul>

        {!loading && places.length === 0 ? (
          <p className="py-8 text-center text-sm text-slate-400">
            검색 결과가 없습니다.
          </p>
        ) : null}
      </div>

      {/* 상세 바텀시트 */}
      <BottomSheet open={!!selected} onClose={() => setSelected(null)}>
        {selected ? (
          <div className="space-y-3">
            <div className="flex h-32 items-center justify-center rounded-xl bg-slate-100 text-4xl">
              🍽️
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-800">{selected.name}</h2>
              <p className="text-sm text-slate-500">{selected.category}</p>
            </div>
            <dl className="space-y-1 text-sm">
              <div className="flex gap-2">
                <dt className="w-12 shrink-0 text-slate-400">주소</dt>
                <dd className="text-slate-700">{selected.address}</dd>
              </div>
              {selected.phone ? (
                <div className="flex gap-2">
                  <dt className="w-12 shrink-0 text-slate-400">전화</dt>
                  <dd className="text-slate-700">{selected.phone}</dd>
                </div>
              ) : null}
              {selected.description ? (
                <div className="flex gap-2">
                  <dt className="w-12 shrink-0 text-slate-400">소개</dt>
                  <dd className="text-slate-700">{selected.description}</dd>
                </div>
              ) : null}
            </dl>
            <div className="flex gap-2 pt-1">
              <a
                href={kakaoDirectionsUrl(selected.name, selected.lat, selected.lng)}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 rounded-lg bg-jeju py-2.5 text-center text-sm font-semibold text-white"
              >
                🧭 길찾기
              </a>
              {selected.placeUrl ? (
                <a
                  href={selected.placeUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-1 rounded-lg border border-jeju py-2.5 text-center text-sm font-semibold text-jeju"
                >
                  상세보기
                </a>
              ) : null}
            </div>
          </div>
        ) : null}
      </BottomSheet>
    </div>
  );
}
