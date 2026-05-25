import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// POST /api/posts/[id]/comments — 댓글 작성 (로그인 가드)
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  const userId = (session?.user as { id?: string } | undefined)?.id;
  if (!userId) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const { id } = await params;
  let body: { body?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "잘못된 요청입니다." }, { status: 400 });
  }

  const content = (body.body ?? "").trim();
  if (!content) {
    return NextResponse.json({ error: "댓글 내용을 입력해 주세요." }, { status: 400 });
  }

  const comment = await prisma.comment.create({
    data: { body: content, postId: id, authorId: userId },
    include: { author: { select: { name: true, image: true } } },
  });
  return NextResponse.json({ comment }, { status: 201 });
}
