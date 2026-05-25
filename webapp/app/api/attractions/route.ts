import { NextResponse } from "next/server";
import seed from "@/data/attractions.seed.json";
import type { Attraction } from "@/lib/types";

// 탭3: 명소·체험 seed 제공 (+ category 필터).
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const category = (searchParams.get("category") || "").trim();

  let list = seed as Attraction[];
  if (category && category !== "전체") {
    list = list.filter((a) => a.category === category);
  }
  return NextResponse.json({ attractions: list });
}
