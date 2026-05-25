"use client";

import { signIn, signOut, useSession } from "next-auth/react";

// 키 주입 여부는 빌드 시 알 수 없으므로 클라이언트는 시도 후 안내.
// providers 목록을 서버에서 받아오는 대신, 단순히 로그인 시도 시 NextAuth 가
// 구성된 provider 만 노출한다. 키가 전혀 없으면 안내 문구를 보여준다.

export function AuthButtons() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return <p className="text-sm text-slate-400">불러오는 중…</p>;
  }

  if (session?.user) {
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          {session.user.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={session.user.image}
              alt={session.user.name ?? "프로필"}
              className="h-12 w-12 rounded-full"
            />
          ) : (
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-jeju-light text-xl">
              👤
            </div>
          )}
          <div>
            <p className="font-semibold text-slate-800">
              {session.user.name ?? "사용자"}
            </p>
            <p className="text-xs text-slate-500">{session.user.email}</p>
          </div>
        </div>
        <button
          onClick={() => signOut()}
          className="w-full rounded-lg border border-slate-200 py-2.5 text-sm font-semibold text-slate-600"
        >
          로그아웃
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <button
        onClick={() => signIn("google")}
        className="w-full rounded-lg border border-slate-200 bg-white py-2.5 text-sm font-semibold text-slate-700"
      >
        Google 로 로그인
      </button>
      <button
        onClick={() => signIn("kakao")}
        className="w-full rounded-lg bg-[#FEE500] py-2.5 text-sm font-semibold text-[#191600]"
      >
        Kakao 로 로그인
      </button>
      <p className="pt-1 text-center text-[11px] text-slate-400">
        OAuth 키(GOOGLE/KAKAO)와 NEXTAUTH_SECRET 가 설정되어야 로그인됩니다. 키가
        없으면 로그인 시도 시 오류 페이지로 이동할 수 있습니다.
      </p>
    </div>
  );
}
