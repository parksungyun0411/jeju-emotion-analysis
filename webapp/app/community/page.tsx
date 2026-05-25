"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { PageHeader } from "@/components/PageHeader";

const CATEGORIES = ["전체", "질문", "꿀팁", "후기"] as const;

interface PostListItem {
  id: string;
  title: string;
  category: string;
  createdAt: string;
  author: { name: string | null } | null;
  _count: { comments: number; likes: number };
}

export default function CommunityPage() {
  const { data: session } = useSession();
  const [category, setCategory] = useState<string>("전체");
  const [posts, setPosts] = useState<PostListItem[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async (cat: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/posts?category=${encodeURIComponent(cat)}`);
      const data = await res.json();
      setPosts(data.posts ?? []);
    } catch {
      setPosts([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load(category);
  }, [category, load]);

  return (
    <div>
      <PageHeader title="커뮤니티" subtitle="제주 Q&A · 꿀팁 · 후기" />

      <div className="sticky top-[57px] z-20 flex gap-2 overflow-x-auto bg-slate-50 p-3">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setCategory(c)}
            className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-semibold ${
              category === c ? "bg-jeju text-white" : "bg-white text-slate-500"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="p-3">
        {loading ? (
          <p className="py-8 text-center text-sm text-slate-400">불러오는 중…</p>
        ) : posts.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-slate-400">아직 글이 없습니다.</p>
            <p className="mt-1 text-xs text-slate-300">첫 글을 작성해 보세요!</p>
          </div>
        ) : (
          <ul className="space-y-2">
            {posts.map((p) => (
              <li key={p.id}>
                <Link
                  href={`/community/${p.id}`}
                  className="block rounded-xl bg-white p-3 shadow-sm active:bg-slate-50"
                >
                  <div className="flex items-center gap-2">
                    <span className="rounded-full bg-jeju-light px-2 py-0.5 text-[10px] font-medium text-jeju-dark">
                      {p.category}
                    </span>
                    <span className="text-[11px] text-slate-400">
                      {new Date(p.createdAt).toLocaleDateString("ko-KR")}
                    </span>
                  </div>
                  <p className="mt-1 font-semibold text-slate-800">{p.title}</p>
                  <div className="mt-1 flex gap-3 text-xs text-slate-400">
                    <span>{p.author?.name ?? "익명"}</span>
                    <span>💬 {p._count?.comments ?? 0}</span>
                    <span>❤️ {p._count?.likes ?? 0}</span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* 글쓰기 FAB */}
      <div className="fixed bottom-16 left-1/2 z-30 w-full max-w-md -translate-x-1/2 px-3 pb-3">
        {session?.user ? (
          <Link
            href="/community/new"
            className="block w-full rounded-xl bg-jeju py-3 text-center text-sm font-bold text-white shadow-lg"
          >
            ✏️ 글쓰기
          </Link>
        ) : (
          <Link
            href="/mypage"
            className="block w-full rounded-xl border border-jeju bg-white py-3 text-center text-sm font-bold text-jeju shadow-lg"
          >
            로그인하고 글쓰기
          </Link>
        )}
      </div>
    </div>
  );
}
