export function PageHeader({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-jeju px-4 py-3 text-white">
      <h1 className="text-lg font-bold">{title}</h1>
      {subtitle ? <p className="text-xs text-jeju-light">{subtitle}</p> : null}
    </header>
  );
}
