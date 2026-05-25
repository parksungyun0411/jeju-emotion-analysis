"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { PageHeader } from "@/components/PageHeader";
import { AuthButtons } from "@/components/AuthButtons";

const CATEGORIES = ["질문", "꿀팁", "후기"] as const;

export default function NewPostPage() {
  const router = useRouter();
  const { data: session, status } = useSession();
  const [category, setCategory] = useState<string>("질문");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (status !== "loading" && !session?.user) {
    return (
      <div>
        <PageHeader title="글쓰기" />
        <div className="space-y-4 p-4">
          <p className="text-sm text-slate-500">글을 작성하려면 로그인이 필요합니다.</p>
          <AuthButtons />
        </div>
      </div>
    );
  }

  async function submit() {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/posts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, title, body }),
      });
      const data = await res.json();
      if (res.ok) {
        router.push(`/community/${data.post.id}`);
      } else {
        setError(data.error || "작성에 실패했습니다.");
      }
    } catch {
      setError("네트워크 오류로 작성에 실패했습니다.");
    }
    setSubmitting(false);
  }

  return (
    <div>
      <PageHeader title="글쓰기" />
      <div className="space-y-4 p-4">
        <div className="flex gap-2">
          {CATEGORIES.map((c) => (
            <button
              key={c}
              onClick={() => setCategory(c)}
              className={`flex-1 rounded-lg py-2 text-sm font-semibold ${
                category === c ? "bg-jeju text-white" : "bg-slate-100 text-slate-500"
              }`}
            >
              {c}
            </button>
          ))}
        </div>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="제목"
          className="w-full rounded-lg border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-jeju"
        />
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="내용을 입력하세요"
          rows={8}
          className="w-full resize-none rounded-lg border border-slate-200 p-3 text-sm outline-none focus:border-jeju"
        />
        {error ? (
          <p className="rounded-lg bg-red-50 p-2 text-sm text-red-600">{error}</p>
        ) : null}
        <button
          onClick={submit}
          disabled={submitting || !title.trim() || !body.trim()}
          className="w-full rounded-xl bg-jeju py-3 text-sm font-bold text-white disabled:opacity-40"
        >
          {submitting ? "등록 중…" : "등록하기"}
        </button>
      </div>
    </div>
  );
}
