"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { PageHeader } from "@/components/PageHeader";

interface CommentItem {
  id: string;
  body: string;
  createdAt: string;
  author: { name: string | null } | null;
}
interface PostDetail {
  id: string;
  title: string;
  body: string;
  category: string;
  createdAt: string;
  likeCount: number;
  author: { name: string | null } | null;
  comments: CommentItem[];
  _count: { likes: number };
}

export default function PostDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: session } = useSession();
  const [post, setPost] = useState<PostDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [comment, setComment] = useState("");
  const [likeCount, setLikeCount] = useState(0);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/posts/${id}`);
      const data = await res.json();
      if (res.ok) {
        setPost(data.post);
        setLikeCount(data.post.likeCount ?? data.post._count?.likes ?? 0);
      }
    } catch {
      /* noop */
    }
    setLoading(false);
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  async function submitComment() {
    if (!comment.trim() || busy) return;
    setBusy(true);
    const res = await fetch(`/api/posts/${id}/comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: comment }),
    });
    if (res.ok) {
      setComment("");
      await load();
    } else if (res.status === 401) {
      alert("로그인이 필요합니다.");
    }
    setBusy(false);
  }

  async function toggleLike() {
    const res = await fetch(`/api/posts/${id}/like`, { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      setLikeCount(data.likeCount);
    } else if (res.status === 401) {
      alert("로그인이 필요합니다.");
    }
  }

  if (loading) {
    return (
      <div>
        <PageHeader title="게시글" />
        <p className="py-12 text-center text-sm text-slate-400">불러오는 중…</p>
      </div>
    );
  }

  if (!post) {
    return (
      <div>
        <PageHeader title="게시글" />
        <div className="py-12 text-center">
          <p className="text-sm text-slate-400">글을 찾을 수 없습니다.</p>
          <Link href="/community" className="mt-2 inline-block text-sm text-jeju">
            목록으로
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="게시글" />
      <div className="p-4">
        <div className="flex items-center gap-2">
          <span className="rounded-full bg-jeju-light px-2 py-0.5 text-[10px] font-medium text-jeju-dark">
            {post.category}
          </span>
          <span className="text-[11px] text-slate-400">
            {new Date(post.createdAt).toLocaleString("ko-KR")}
          </span>
        </div>
        <h1 className="mt-2 text-xl font-bold text-slate-800">{post.title}</h1>
        <p className="mt-1 text-xs text-slate-400">{post.author?.name ?? "익명"}</p>
        <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
          {post.body}
        </p>

        <button
          onClick={toggleLike}
          className="mt-5 inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600"
        >
          ❤️ 좋아요 {likeCount}
        </button>
      </div>

      {/* 댓글 */}
      <div className="border-t border-slate-100 p-4">
        <p className="mb-3 text-sm font-semibold text-slate-600">
          댓글 {post.comments.length}
        </p>
        <ul className="space-y-3">
          {post.comments.map((c) => (
            <li key={c.id} className="rounded-lg bg-white p-3 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-700">
                  {c.author?.name ?? "익명"}
                </span>
                <span className="text-[10px] text-slate-400">
                  {new Date(c.createdAt).toLocaleDateString("ko-KR")}
                </span>
              </div>
              <p className="mt-1 text-sm text-slate-700">{c.body}</p>
            </li>
          ))}
          {post.comments.length === 0 ? (
            <li className="text-center text-xs text-slate-400">
              첫 댓글을 남겨보세요.
            </li>
          ) : null}
        </ul>

        {session?.user ? (
          <div className="mt-4 flex gap-2">
            <input
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="댓글 입력"
              className="flex-1 rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-jeju"
              onKeyDown={(e) => e.key === "Enter" && submitComment()}
            />
            <button
              onClick={submitComment}
              disabled={busy}
              className="rounded-lg bg-jeju px-4 text-sm font-semibold text-white disabled:opacity-50"
            >
              등록
            </button>
          </div>
        ) : (
          <Link
            href="/mypage"
            className="mt-4 block rounded-lg border border-jeju py-2.5 text-center text-sm font-semibold text-jeju"
          >
            로그인하고 댓글 달기
          </Link>
        )}
      </div>
    </div>
  );
}
