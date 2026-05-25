import type { Metadata, Viewport } from "next";
import "./globals.css";
import { BottomNav } from "@/components/BottomNav";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "제주올 — 제주 올인원 앱",
  description:
    "제주어 번역·감정분석, 맛집, 명소·체험, AI 여행경로, 커뮤니티를 한 앱에.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#0d9488",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body className="bg-slate-100">
        <Providers>
          {/* 모바일 프레임 */}
          <div className="relative mx-auto flex min-h-screen w-full max-w-md flex-col bg-slate-50 shadow-xl">
            <main className="flex-1 overflow-y-auto pb-20">{children}</main>
            <BottomNav />
          </div>
        </Providers>
      </body>
    </html>
  );
}
