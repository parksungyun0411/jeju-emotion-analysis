import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// GET /api/posts/[id] — 상세 (댓글 포함)
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const post = await prisma.post.findUnique({
      where: { id },
      include: {
        author: { select: { name: true, image: true } },
        comments: {
          orderBy: { createdAt: "asc" },
          include: { author: { select: { name: true, image: true } } },
        },
        _count: { select: { likes: true } },
      },
    });
    if (!post) {
      return NextResponse.json({ error: "글을 찾을 수 없습니다." }, { status: 404 });
    }
    return NextResponse.json({ post });
  } catch {
    return NextResponse.json({ error: "DB 미준비" }, { status: 503 });
  }
}
