"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { KakaoMap, kakaoDirectionsUrl, type MapMarker } from "@/components/KakaoMap";
import { BottomSheet } from "@/components/BottomSheet";
import type { Attraction, RoutePlan } from "@/lib/types";

const CATEGORIES = ["전체", "자연", "문화", "체험", "포토"] as const;

export default function SpotsPage() {
  const [attractions, setAttractions] = useState<Attraction[]>([]);
  const [category, setCategory] = useState<string>("전체");
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState<Attraction | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // AI 경로 상태
  const [days, setDays] = useState(2);
  const [transport, setTransport] = useState("렌터카");
  const [prefs, setPrefs] = useState("");
  const [planning, setPlanning] = useState(false);
  const [plan, setPlan] = useState<RoutePlan | null>(null);
  const [showPlanForm, setShowPlanForm] = useState(false);

  const fetchAttractions = useCallback(async (cat: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/attractions?category=${encodeURIComponent(cat)}`);
      const data = await res.json();
      setAttractions(data.attractions ?? []);
    } catch {
      setAttractions([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAttractions(category);
  }, [category, fetchAttractions]);

  const selectedSpots = useMemo(
    () => attractions.filter((a) => selectedIds.has(a.id)),
    [attractions, selectedIds]
  );

  // 지도 마커: 경로가 있으면 경로 stops, 없으면 명소 전체
  const markers: MapMarker[] = useMemo(() => {
    if (plan) {
      return plan.days
        .flatMap((d) => d.stops)
        .filter((s) => typeof s.lat === "number" && typeof s.lng === "number")
        .map((s, i) => ({ id: `p${i}`, lat: s.lat!, lng: s.lng!, title: s.name }));
    }
    return attractions.map((a) => ({ id: a.id, lat: a.lat, lng: a.lng, title: a.name }));
  }, [attractions, plan]);

  const polyline = useMemo(() => {
    if (!plan) return undefined;
    return plan.days
      .flatMap((d) => d.stops)
      .filter((s) => typeof s.lat === "number" && typeof s.lng === "number")
      .map((s) => ({ lat: s.lat!, lng: s.lng! }));
  }, [plan]);

  function toggleSelect(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const onMarkerClick = useCallback(
    (id: string) => {
      if (plan) return;
      const a = attractions.find((x) => x.id === id);
      if (a) setDetail(a);
    },
    [attractions, plan]
  );

  async function generateRoute() {
    if (selectedSpots.length === 0 || planning) return;
    setPlanning(true);
    setPlan(null);
    try {
      const res = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spots: selectedSpots, days, transport, prefs }),
      });
      const data = await res.json();
      if (res.ok) {
        setPlan(data);
        setShowPlanForm(false);
      }
    } catch {
      /* noop */
    }
    setPlanning(false);
  }

  return (
    <div>
      <PageHeader title="명소·체험" subtitle="큐레이션 + AI 여행경로" />

      {/* 카테고리 필터 */}
      <div className="sticky top-[57px] z-20 flex gap-2 overflow-x-auto bg-slate-50 p-3">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => {
              setCategory(c);
              setPlan(null);
            }}
            className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-semibold ${
              category === c ? "bg-jeju text-white" : "bg-white text-slate-500"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      {/* 지도 */}
      <KakaoMap
        markers={markers}
        polyline={polyline}
        onMarkerClick={onMarkerClick}
        className="h-56 w-full"
      />

      {/* AI 경로 결과 */}
      {plan ? (
        <div className="space-y-3 p-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-700">🗺️ AI 여행경로</h2>
            <button
              onClick={() => setPlan(null)}
              className="text-xs font-medium text-jeju"
            >
              명소 목록으로
            </button>
          </div>
          {plan.mock ? (
            <p className="rounded-lg bg-amber-50 p-2 text-xs text-amber-700">
              예시(mock) 일정입니다. ANTHROPIC_API_KEY 설정 시 AI 맞춤 일정이 생성됩니다.
            </p>
          ) : null}
          <p className="text-sm text-slate-600">{plan.summary}</p>
          {plan.days.map((d) => (
            <div key={d.day} className="rounded-xl bg-white p-3 shadow-sm">
              <p className="mb-2 font-semibold text-jeju-dark">
                {d.title || `${d.day}일차`}
              </p>
              <ol className="space-y-2">
                {d.stops.map((s, i) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-jeju-light text-[10px] font-bold text-jeju-dark">
                      {i + 1}
                    </span>
                    <div>
                      <p className="font-medium text-slate-800">
                        {s.name}
                        {s.time ? (
                          <span className="ml-1 text-xs text-slate-400">{s.time}</span>
                        ) : null}
                      </p>
                      {s.note ? (
                        <p className="text-xs text-slate-500">{s.note}</p>
                      ) : null}
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      ) : (
        <>
          {/* 명소 목록 (선택 가능) */}
          <div className="p-3">
            <p className="mb-2 text-sm font-semibold text-slate-600">
              {loading ? "불러오는 중…" : `명소·체험 ${attractions.length}곳`}
              {selectedIds.size > 0 ? (
                <span className="ml-2 text-jeju">· {selectedIds.size}곳 선택</span>
              ) : null}
            </p>
            <ul className="space-y-2">
              {attractions.map((a) => {
                const sel = selectedIds.has(a.id);
                return (
                  <li
                    key={a.id}
                    className={`rounded-xl bg-white p-3 shadow-sm ${
                      sel ? "ring-2 ring-jeju" : ""
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <button
                        onClick={() => setDetail(a)}
                        className="flex-1 text-left"
                      >
                        <p className="font-semibold text-slate-800">{a.name}</p>
                        <p className="text-xs text-slate-500">
                          {a.category}
                          {a.duration ? ` · ${a.duration}` : ""}
                        </p>
                        <p className="mt-0.5 text-xs text-slate-400">{a.address}</p>
                      </button>
                      <button
                        onClick={() => toggleSelect(a.id)}
                        className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold ${
                          sel
                            ? "bg-jeju text-white"
                            : "border border-jeju text-jeju"
                        }`}
                      >
                        {sel ? "선택됨" : "담기"}
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        </>
      )}

      {/* AI 경로 생성 버튼 (선택 시 표시) */}
      {!plan && selectedIds.size > 0 ? (
        <div className="fixed bottom-16 left-1/2 z-30 w-full max-w-md -translate-x-1/2 p-3">
          <button
            onClick={() => setShowPlanForm(true)}
            className="w-full rounded-xl bg-jeju py-3 text-sm font-bold text-white shadow-lg"
          >
            ✨ AI 여행경로 만들기 ({selectedIds.size}곳)
          </button>
        </div>
      ) : null}

      {/* 경로 조건 입력 시트 */}
      <BottomSheet open={showPlanForm} onClose={() => setShowPlanForm(false)}>
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-slate-800">여행 조건</h2>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              여행 일수
            </label>
            <div className="flex gap-2">
              {[1, 2, 3, 4].map((d) => (
                <button
                  key={d}
                  onClick={() => setDays(d)}
                  className={`flex-1 rounded-lg py-2 text-sm font-semibold ${
                    days === d ? "bg-jeju text-white" : "bg-slate-100 text-slate-500"
                  }`}
                >
                  {d}일
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              이동수단
            </label>
            <div className="flex gap-2">
              {["렌터카", "대중교통", "도보"].map((t) => (
                <button
                  key={t}
                  onClick={() => setTransport(t)}
                  className={`flex-1 rounded-lg py-2 text-sm font-semibold ${
                    transport === t ? "bg-jeju text-white" : "bg-slate-100 text-slate-500"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">
              취향 (선택)
            </label>
            <input
              value={prefs}
              onChange={(e) => setPrefs(e.target.value)}
              placeholder="예) 사진 명소 위주, 여유롭게"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-jeju"
            />
          </div>
          <button
            onClick={generateRoute}
            disabled={planning}
            className="w-full rounded-xl bg-jeju py-3 text-sm font-bold text-white disabled:opacity-50"
          >
            {planning ? "일정 생성 중…" : "일정 생성하기"}
          </button>
        </div>
      </BottomSheet>

      {/* 명소 상세 시트 */}
      <BottomSheet open={!!detail} onClose={() => setDetail(null)}>
        {detail ? (
          <div className="space-y-3">
            <div className="flex h-32 items-center justify-center rounded-xl bg-slate-100 text-4xl">
              🏝️
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-800">{detail.name}</h2>
              <p className="text-sm text-slate-500">
                {detail.category}
                {detail.duration ? ` · 권장 ${detail.duration}` : ""}
              </p>
            </div>
            {detail.description ? (
              <p className="text-sm text-slate-700">{detail.description}</p>
            ) : null}
            <p className="text-xs text-slate-400">{detail.address}</p>
            <div className="flex gap-2 pt-1">
              <a
                href={kakaoDirectionsUrl(detail.name, detail.lat, detail.lng)}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 rounded-lg bg-jeju py-2.5 text-center text-sm font-semibold text-white"
              >
                🧭 길찾기
              </a>
              <button
                onClick={() => {
                  toggleSelect(detail.id);
                  setDetail(null);
                }}
                className="flex-1 rounded-lg border border-jeju py-2.5 text-center text-sm font-semibold text-jeju"
              >
                {selectedIds.has(detail.id) ? "선택 해제" : "경로에 담기"}
              </button>
            </div>
          </div>
        ) : null}
      </BottomSheet>
    </div>
  );
}
