import { NextResponse } from "next/server";
import seed from "@/data/restaurants.seed.json";
import type { Place } from "@/lib/types";

const KAKAO_KEY = process.env.KAKAO_REST_API_KEY;

// 탭2: 맛집 — Kakao Local 프록시(키 있을 때) + seed 폴백.
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const query = (searchParams.get("query") || "").trim();
  const category = (searchParams.get("category") || "").trim();
  const lat = searchParams.get("lat");
  const lng = searchParams.get("lng");

  // 키가 없으면 seed 데이터로 응답.
  if (!KAKAO_KEY) {
    return NextResponse.json({ source: "seed", places: filterSeed(query, category) });
  }

  try {
    // Kakao Local keyword 검색. (제주 음식점 FD6 카테고리)
    const params = new URLSearchParams({
      query: query || "제주 맛집",
      category_group_code: "FD6",
      size: "15",
    });
    if (lat && lng) {
      params.set("x", lng);
      params.set("y", lat);
      params.set("radius", "20000");
    }

    const res = await fetch(
      `https://dapi.kakao.com/v2/local/search/keyword.json?${params.toString()}`,
      {
        headers: { Authorization: `KakaoAK ${KAKAO_KEY}` },
        cache: "no-store",
        signal: AbortSignal.timeout(10000),
      }
    );

    if (!res.ok) {
      return NextResponse.json({ source: "seed", places: filterSeed(query, category) });
    }

    const data = await res.json();
    const places: Place[] = (data.documents ?? []).map(
      (d: KakaoDoc): Place => ({
        id: d.id,
        name: d.place_name,
        category: d.category_name?.split(">").pop()?.trim() || "음식점",
        address: d.road_address_name || d.address_name || "",
        lat: parseFloat(d.y),
        lng: parseFloat(d.x),
        phone: d.phone || undefined,
        placeUrl: d.place_url || undefined,
      })
    );

    return NextResponse.json({ source: "kakao", places });
  } catch {
    return NextResponse.json({ source: "seed", places: filterSeed(query, category) });
  }
}

interface KakaoDoc {
  id: string;
  place_name: string;
  category_name?: string;
  road_address_name?: string;
  address_name?: string;
  x: string;
  y: string;
  phone?: string;
  place_url?: string;
}

function filterSeed(query: string, category: string): Place[] {
  let list = seed as Place[];
  if (query) {
    const q = query.toLowerCase();
    list = list.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        p.category.toLowerCase().includes(q) ||
        p.address.toLowerCase().includes(q)
    );
  }
  if (category) {
    list = list.filter((p) => p.category.includes(category));
  }
  return list;
}
