import { NextResponse } from "next/server";

const ML_BASE = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

// 탭1: 제주어 감정분석 — FastAPI(/api/emotion) 프록시.
export async function POST(req: Request) {
  let body: { text?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  const text = (body.text ?? "").trim();
  if (!text) {
    return NextResponse.json({ error: "문장을 입력해 주세요." }, { status: 400 });
  }

  try {
    const res = await fetch(`${ML_BASE}/api/emotion`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });

    if (res.status === 503) {
      return NextResponse.json(
        { error: "감정 모델 학습 중입니다. 잠시 후 다시 시도해 주세요.", unavailable: true },
        { status: 503 }
      );
    }
    if (!res.ok) {
      return NextResponse.json({ error: "감정 분석에 실패했습니다." }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      {
        error: "ML 서비스에 연결할 수 없습니다. 모델 학습/기동 중일 수 있습니다.",
        unavailable: true,
      },
      { status: 503 }
    );
  }
}
