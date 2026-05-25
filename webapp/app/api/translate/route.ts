import { NextResponse } from "next/server";

const ML_BASE = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

// 탭1: 제주어 번역 — FastAPI(/api/translate) 프록시.
export async function POST(req: Request) {
  let body: { text?: string; direction?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  const text = (body.text ?? "").trim();
  const direction = body.direction === "s2j" ? "s2j" : "j2s";

  if (!text) {
    return NextResponse.json({ error: "문장을 입력해 주세요." }, { status: 400 });
  }

  try {
    const res = await fetch(`${ML_BASE}/api/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
      // ML 서비스가 느릴 수 있으니 캐시 비활성화.
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });

    if (res.status === 503) {
      return NextResponse.json(
        { error: "모델 학습 중입니다. 잠시 후 다시 시도해 주세요.", unavailable: true },
        { status: 503 }
      );
    }
    if (!res.ok) {
      const detail = await safeDetail(res);
      return NextResponse.json({ error: detail || "번역에 실패했습니다." }, { status: res.status });
    }

    const data = await res.json();
    return NextResponse.json({ translation: data.translation });
  } catch {
    return NextResponse.json(
      {
        error:
          "ML 서비스에 연결할 수 없습니다. 모델 학습/기동 중이거나 서비스가 꺼져 있습니다.",
        unavailable: true,
      },
      { status: 503 }
    );
  }
}

async function safeDetail(res: Response): Promise<string | null> {
  try {
    const j = await res.json();
    return j.detail || j.error || null;
  } catch {
    return null;
  }
}
