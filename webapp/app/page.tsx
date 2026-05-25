"use client";

import { useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import type { EmotionResult, TranslateDirection } from "@/lib/types";

// 감정 라벨 → 색/이모지 매핑 (7종 가정, 미지정 라벨은 기본값).
const EMOTION_STYLE: Record<string, { color: string; emoji: string }> = {
  기쁨: { color: "bg-amber-100 text-amber-800", emoji: "😊" },
  슬픔: { color: "bg-blue-100 text-blue-800", emoji: "😢" },
  분노: { color: "bg-red-100 text-red-800", emoji: "😠" },
  불안: { color: "bg-purple-100 text-purple-800", emoji: "😟" },
  당황: { color: "bg-orange-100 text-orange-800", emoji: "😳" },
  상처: { color: "bg-rose-100 text-rose-800", emoji: "💔" },
  중립: { color: "bg-slate-100 text-slate-700", emoji: "😐" },
};

function emotionStyle(label: string) {
  return EMOTION_STYLE[label] ?? { color: "bg-slate-100 text-slate-700", emoji: "🏷️" };
}

export default function JejuLanguagePage() {
  const [text, setText] = useState("");
  const [direction, setDirection] = useState<TranslateDirection>("j2s");
  const [loading, setLoading] = useState(false);
  const [translation, setTranslation] = useState<string | null>(null);
  const [emotion, setEmotion] = useState<EmotionResult | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function handleSubmit() {
    const t = text.trim();
    if (!t || loading) return;
    setLoading(true);
    setTranslation(null);
    setEmotion(null);
    setNotice(null);

    // 번역
    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t, direction }),
      });
      const data = await res.json();
      if (res.ok) {
        setTranslation(data.translation);
      } else {
        setNotice(data.error || "번역에 실패했습니다.");
      }
    } catch {
      setNotice("네트워크 오류로 번역에 실패했습니다.");
    }

    // 감정분석은 제주어 입력(j2s)일 때만 의미가 있음.
    if (direction === "j2s") {
      try {
        const res = await fetch("/api/emotion", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: t }),
        });
        if (res.ok) {
          setEmotion(await res.json());
        }
        // 감정 실패는 조용히 무시(번역이 핵심).
      } catch {
        /* noop */
      }
    }

    setLoading(false);
  }

  const directionLabel =
    direction === "j2s" ? "제주어 → 표준어" : "표준어 → 제주어";

  return (
    <div>
      <PageHeader title="제주어" subtitle="번역 + 감정분석" />

      <div className="space-y-4 p-4">
        {/* 방향 토글 */}
        <div className="flex items-center justify-center gap-2 rounded-full bg-white p-1 shadow-sm">
          <button
            onClick={() => setDirection("j2s")}
            className={`flex-1 rounded-full py-2 text-sm font-semibold transition ${
              direction === "j2s" ? "bg-jeju text-white" : "text-slate-500"
            }`}
          >
            제주어→표준어
          </button>
          <button
            onClick={() => setDirection("s2j")}
            className={`flex-1 rounded-full py-2 text-sm font-semibold transition ${
              direction === "s2j" ? "bg-jeju text-white" : "text-slate-500"
            }`}
          >
            표준어→제주어
          </button>
        </div>

        {/* 입력창 */}
        <div className="rounded-2xl bg-white p-3 shadow-sm">
          <label className="mb-1 block text-xs font-medium text-slate-400">
            {directionLabel}
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={
              direction === "j2s"
                ? "예) 혼저옵서예, 무사 경 했수광?"
                : "예) 어서 오세요, 왜 그렇게 했어요?"
            }
            rows={3}
            className="w-full resize-none rounded-lg border border-slate-200 p-3 text-sm outline-none focus:border-jeju"
          />
          <button
            onClick={handleSubmit}
            disabled={loading || !text.trim()}
            className="mt-2 w-full rounded-lg bg-jeju py-2.5 text-sm font-semibold text-white disabled:opacity-40"
          >
            {loading ? "분석 중…" : "번역하기"}
          </button>
        </div>

        {/* 안내 메시지 */}
        {notice ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
            {notice}
          </div>
        ) : null}

        {/* 결과 카드 */}
        {translation !== null ? (
          <div className="space-y-3 rounded-2xl bg-white p-4 shadow-sm">
            <div>
              <p className="mb-1 text-xs font-medium text-slate-400">번역 결과</p>
              <p className="text-base font-medium text-slate-800">{translation}</p>
            </div>

            {emotion ? (
              <div className="border-t border-slate-100 pt-3">
                <p className="mb-2 text-xs font-medium text-slate-400">감정 분석</p>
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm font-semibold ${
                      emotionStyle(emotion.label).color
                    }`}
                  >
                    {emotionStyle(emotion.label).emoji} {emotion.label}
                  </span>
                  {typeof emotion.scores?.[emotion.label] === "number" ? (
                    <span className="text-xs text-slate-500">
                      신뢰도 {(emotion.scores[emotion.label] * 100).toFixed(1)}%
                    </span>
                  ) : null}
                </div>

                {/* 점수 분포 막대 */}
                <ul className="mt-3 space-y-1">
                  {Object.entries(emotion.scores ?? {})
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 4)
                    .map(([label, score]) => (
                      <li key={label} className="flex items-center gap-2 text-xs">
                        <span className="w-10 shrink-0 text-slate-500">{label}</span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-jeju"
                            style={{ width: `${Math.round(score * 100)}%` }}
                          />
                        </div>
                        <span className="w-10 text-right text-slate-400">
                          {(score * 100).toFixed(0)}%
                        </span>
                      </li>
                    ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}

        <p className="px-1 text-center text-[11px] leading-relaxed text-slate-400">
          번역·감정 모델은 별도 ML 서비스에서 동작합니다. 모델 학습/기동 전에는
          &ldquo;모델 학습 중&rdquo; 안내가 표시될 수 있습니다.
        </p>
      </div>
    </div>
  );
}
