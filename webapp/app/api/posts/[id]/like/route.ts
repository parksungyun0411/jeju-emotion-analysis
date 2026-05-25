import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// POST /api/posts/[id]/like — 좋아요 토글 (로그인 가드)
export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { id } = await params;
  const existing = await prisma.like.findUnique({
    where: { userId_postId: { userId, postId: id } },
  });

  let liked: boolean;
  if (existing) {
    await prisma.like.delete({ where: { id: existing.id } });
    await prisma.post.update({
      where: { id },
      data: { likeCount: { decrement: 1 } },
    });
    liked = false;
  } else {
    await prisma.like.create({ data: { userId, postId: id } });
    await prisma.post.update({
      where: { id },
      data: { likeCount: { increment: 1 } },
    });
    liked = true;
  }

  const post = await prisma.post.findUnique({
    where: { id },
    select: { likeCount: true },
  });
  return NextResponse.json({ liked, likeCount: post?.likeCount ?? 0 });
}
