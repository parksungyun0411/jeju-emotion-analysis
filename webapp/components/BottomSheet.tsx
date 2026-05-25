"use client";

import type { ReactNode } from "react";

export function BottomSheet({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 mx-auto flex max-w-md items-end">
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden
      />
      <div className="relative w-full rounded-t-3xl bg-white p-4 pb-6 shadow-2xl animate-in">
        <div className="mx-auto mb-3 h-1.5 w-10 rounded-full bg-slate-200" />
        {children}
      </div>
    </div>
  );
}
