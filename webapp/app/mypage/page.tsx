"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useSession } from "next-auth/react";
import { PageHeader } from "@/components/PageHeader";
import { AuthButtons } from "@/components/AuthButtons";

interface MyPost {
  id: string;
  title: string;
  category: string;
  createdAt: string;
  likeCount: number;
}

export default function MyPage() {
  const { data: session, status } = useSession();
  const [myPosts, setMyPosts] = useState<MyPost[]>([]);

  useEffect(() => {
    if (!session?.user) return;
    fetch("/api/me/posts")
      .then((r) => r.json())
      .then((d) => setMyPosts(d.posts ?? []))
      .catch(() => setMyPosts([]));
  }, [session]);

  return (
    <div>
      <PageHeader title="마이페이지" />

      <div className="space-y-5 p-4">
        {/* 프로필 / 로그인 */}
        <section className="rounded-2xl bg-white p-4 shadow-sm">
          {status === "loading" ? (
            <p className="text-sm text-slate-400">불러오는 중…</p>
          ) : (
            <AuthButtons />
          )}
        </section>

        {/* 내 글 */}
        {session?.user ? (
          <section>
            <h2 className="mb-2 text-sm font-semibold text-slate-600">내 글</h2>
            {myPosts.length === 0 ? (
              <p className="rounded-xl bg-white p-4 text-center text-sm text-slate-400 shadow-sm">
                아직 작성한 글이 없습니다.
              </p>
            ) : (
              <ul className="space-y-2">
                {myPosts.map((p) => (
                  <li key={p.id}>
                    <Link
                      href={`/community/${p.id}`}
                      className="block rounded-xl bg-white p-3 shadow-sm"
                    >
                      <div className="flex items-center gap-2">
                        <span className="rounded-full bg-jeju-light px-2 py-0.5 text-[10px] font-medium text-jeju-dark">
                          {p.category}
                        </span>
                        <span className="text-[11px] text-slate-400">
                          {new Date(p.createdAt).toLocaleDateString("ko-KR")}
                        </span>
                      </div>
                      <p className="mt-1 font-medium text-slate-800">{p.title}</p>
                      <p className="mt-0.5 text-xs text-slate-400">
                        ❤️ {p.likeCount}
                      </p>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </section>
        ) : null}
      </div>
    </div>
  );
}
