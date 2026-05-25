import type { NextAuthOptions } from "next-auth";
import { PrismaAdapter } from "@auth/prisma-adapter";
import GoogleProvider from "next-auth/providers/google";
import KakaoProvider from "next-auth/providers/kakao";
import type { Adapter } from "next-auth/adapters";
import { prisma } from "@/lib/prisma";

// 키가 주입된 provider 만 등록한다. (키 없으면 로그인 버튼은 안내만 표시)
const providers: NextAuthOptions["providers"] = [];

if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
  providers.push(
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    })
  );
}

if (process.env.KAKAO_CLIENT_ID && process.env.KAKAO_CLIENT_SECRET) {
  providers.push(
    KakaoProvider({
      clientId: process.env.KAKAO_CLIENT_ID,
      clientSecret: process.env.KAKAO_CLIENT_SECRET,
    })
  );
}

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma) as Adapter,
  providers,
  session: { strategy: "database" },
  pages: {
    signIn: "/mypage",
  },
  callbacks: {
    async session({ session, user }) {
      if (session.user && user) {
        // user.id 를 세션에 노출.
        (session.user as { id?: string }).id = user.id;
      }
      return session;
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
};

/** 로그인 가능한 provider 가 하나라도 구성되어 있는지. */
export const authConfigured = providers.length > 0;
