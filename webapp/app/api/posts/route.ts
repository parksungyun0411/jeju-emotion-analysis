import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// GET /api/posts?category= — 목록
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const category = searchParams.get("category");

  try {
    const posts = await prisma.post.findMany({
      where: category && category !== "전체" ? { category } : undefined,
      orderBy: { createdAt: "desc" },
      take: 50,
      include: {
        author: { select: { name: true, image: true } },
        _count: { select: { comments: true, likes: true } },
      },
    });
    return NextResponse.json({ posts });
  } catch {
    // DB 미마이그레이션 등.
    return NextResponse.json({ posts: [], error: "DB 미준비" });
  }
}

// POST /api/posts — 작성 (로그인 가드)
export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  let body: { category?: string; title?: string; body?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  const title = (body.title ?? "").trim();
  const content = (body.body ?? "").trim();
  const category = body.category || "질문";
  if (!title || !content) {
    return NextResponse.json({ error: "제목과 내용을 입력해 주세요." }, { status: 400 });
  }

  const post = await prisma.post.create({
    data: { title, body: content, category, authorId: userId },
  });
  return NextResponse.json({ post }, { status: 201 });
}
