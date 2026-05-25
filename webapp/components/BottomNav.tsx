"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "제주어", icon: "💬" },
  { href: "/food", label: "맛집", icon: "🍽️" },
  { href: "/spots", label: "명소·체험", icon: "🏝️" },
  { href: "/community", label: "커뮤니티", icon: "💬" },
];

// 명소 탭만 아이콘 중복 방지
const ICONS: Record<string, string> = {
  "/": "🗣️",
  "/food": "🍽️",
  "/spots": "🏝️",
  "/community": "👥",
};

export function BottomNav() {
  const pathname = usePathname();

  function isActive(href: string) {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  }

  return (
    <nav className="fixed bottom-0 left-1/2 z-40 w-full max-w-md -translate-x-1/2 border-t border-slate-200 bg-white/95 backdrop-blur">
      <ul className="flex">
        {TABS.map((t) => {
          const active = isActive(t.href);
          return (
            <li key={t.href} className="flex-1">
              <Link
                href={t.href}
                className={`flex flex-col items-center justify-center gap-0.5 py-2 text-[11px] font-medium transition-colors ${
                  active ? "text-jeju" : "text-slate-400 hover:text-slate-600"
                }`}
              >
                <span className="text-xl leading-none">{ICONS[t.href] ?? t.icon}</span>
                <span>{t.label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
