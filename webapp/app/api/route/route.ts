import { NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { Attraction, RoutePlan, RoutePlanDay } from "@/lib/types";

const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = "claude-sonnet-4-5";

interface RouteRequestBody {
  spots: Attraction[];
  days?: number;
  transport?: string;
  prefs?: string;
}

// 탭3: AI 여행경로 — 선택 장소 + 조건을 Claude 로 일정 생성. 키 없으면 mock.
export async function POST(req: Request) {
  let body: RouteRequestBody;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  const spots = Array.isArray(body.spots) ? body.spots : [];
  const days = Math.max(1, Math.min(7, Number(body.days) || 1));
  const transport = body.transport || "렌터카";
  const prefs = body.prefs || "";

  if (spots.length === 0) {
    return NextResponse.json(
      { error: "최소 1개 이상의 장소를 선택해 주세요." },
      { status: 400 }
    );
  }

  // 키 없으면 mock 일정 (좌표 순서대로 균등 분배).
  if (!ANTHROPIC_KEY) {
    return NextResponse.json(mockPlan(spots, days, transport));
  }

  try {
    const client = new Anthropic({ apiKey: ANTHROPIC_KEY });
    const spotList = spots
      .map((s) => `- ${s.name} (${s.category}, ${s.address})`)
      .join("\n");

    const prompt = `당신은 제주도 여행 플래너입니다. 아래 선택된 장소들을 ${days}일 일정으로,
이동수단 "${transport}"${prefs ? `, 취향 "${prefs}"` : ""} 를 고려해 동선이 효율적인 일자별 일정으로 짜주세요.

선택 장소:
${spotList}

반드시 아래 JSON 스키마로만 응답하세요 (설명 텍스트 없이 JSON만):
{
  "summary": "전체 일정 요약 한 문장",
  "days": [
    { "day": 1, "title": "1일차 제목", "stops": [ { "name": "장소명", "time": "오전", "note": "팁" } ] }
  ]
}
장소명은 위 목록의 이름을 그대로 사용하세요.`;

    const msg = await client.messages.create({
      model: MODEL,
      max_tokens: 1500,
      messages: [{ role: "user", content: prompt }],
    });

    const textBlock = msg.content.find((b) => b.type === "text");
    const raw = textBlock && "text" in textBlock ? textBlock.text : "";
    const jsonStr = extractJson(raw);
    const parsed = JSON.parse(jsonStr) as RoutePlan;

    // 좌표 보강 (이름 매칭).
    const byName = new Map(spots.map((s) => [s.name, s]));
    parsed.days?.forEach((d) =>
      d.stops?.forEach((stop) => {
        const match = byName.get(stop.name);
        if (match) {
          stop.lat = match.lat;
          stop.lng = match.lng;
        }
      })
    );

    return NextResponse.json({ ...parsed, mock: false });
  } catch {
    // Claude 실패 시에도 mock 으로 graceful degradation.
    return NextResponse.json(mockPlan(spots, days, transport));
  }
}

function mockPlan(
  spots: Attraction[],
  days: number,
  transport: string
): RoutePlan {
  const perDay = Math.ceil(spots.length / days);
  const dayPlans: RoutePlanDay[] = [];
  for (let d = 0; d < days; d++) {
    const slice = spots.slice(d * perDay, (d + 1) * perDay);
    if (slice.length === 0) continue;
    dayPlans.push({
      day: d + 1,
      title: `${d + 1}일차`,
      stops: slice.map((s, i) => ({
        name: s.name,
        lat: s.lat,
        lng: s.lng,
        time: i === 0 ? "오전" : i === 1 ? "오후" : "저녁",
        note: s.description || s.category,
      })),
    });
  }
  return {
    summary: `선택한 ${spots.length}곳을 ${days}일 동안 ${transport}로 둘러보는 예시 일정입니다. (ANTHROPIC_API_KEY 설정 시 AI 맞춤 일정 생성)`,
    days: dayPlans,
    mock: true,
  };
}

function extractJson(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) return fenced[1].trim();
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start >= 0 && end > start) return text.slice(start, end + 1);
  return text;
}
