/* 제주어 챗봇 프론트엔드 — 빌드 불필요 순수 정적 JS. */
(function () {
  "use strict";

  // 감정별 이모지 (설계 라벨 순서: 중립/슬픔/행복/분노/놀람/공포/혐오).
  const EMOTION_EMOJI = {
    "중립": "😐",
    "슬픔": "😢",
    "행복": "😊",
    "분노": "😠",
    "놀람": "😮",
    "공포": "😱",
    "혐오": "🤢",
  };

  const DIR_LABEL = {
    j2s: "제주어 → 표준어",
    s2j: "표준어 → 제주어",
  };

  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusBar = document.getElementById("status");
  const dirButtons = document.querySelectorAll(".dir-btn");

  let direction = "j2s";

  // ── 방향 토글 ──
  dirButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      dirButtons.forEach(function (b) { b.classList.remove("active"); });
      btn.classList.add("active");
      direction = btn.dataset.dir;
      input.placeholder =
        direction === "j2s" ? "제주어 문장을 입력하세요…" : "표준어 문장을 입력하세요…";
    });
  });

  // ── 메시지 렌더링 ──
  function scrollToBottom() {
    chat.scrollTop = chat.scrollHeight;
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
  }

  function addTyping() {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-bot typing";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = "변환 중…";
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addErrorMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-bot";
    const bubble = document.createElement("div");
    bubble.className = "bubble error";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
  }

  function buildEmotionBlock(emotion) {
    // emotion: {label, label_id, scores:{...}}
    const block = document.createElement("div");
    block.className = "emotion";

    const emoji = EMOTION_EMOJI[emotion.label] || "🙂";
    const conf = emotion.scores && emotion.scores[emotion.label] != null
      ? emotion.scores[emotion.label]
      : 0;

    const badge = document.createElement("span");
    badge.className = "emotion-badge";
    badge.innerHTML =
      '<span class="emotion-emoji">' + emoji + "</span>" +
      "<span>" + emotion.label + "</span>";
    block.appendChild(badge);

    const confWrap = document.createElement("div");
    confWrap.className = "confidence";
    const pct = Math.round(conf * 100);
    confWrap.innerHTML =
      '<div class="conf-label">신뢰도 ' + pct + "%</div>" +
      '<div class="conf-bar"><div class="conf-fill" style="width:' + pct + '%"></div></div>';
    block.appendChild(confWrap);

    return block;
  }

  function addBotMessage(data) {
    const wrap = document.createElement("div");
    wrap.className = "msg msg-bot";
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const tag = document.createElement("div");
    tag.className = "source-tag";
    tag.textContent = DIR_LABEL[direction] || "";
    bubble.appendChild(tag);

    const text = document.createElement("div");
    text.textContent = data.translation;
    bubble.appendChild(text);

    if (data.emotion) {
      bubble.appendChild(buildEmotionBlock(data.emotion));
    }

    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    scrollToBottom();
  }

  // ── 전송 ──
  async function send(text) {
    sendBtn.disabled = true;
    const typing = addTyping();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, direction: direction }),
      });
      typing.remove();
      if (!res.ok) {
        let detail = "요청 처리 중 오류가 발생했습니다.";
        try {
          const err = await res.json();
          if (err && err.detail) detail = err.detail;
        } catch (e) { /* ignore */ }
        if (res.status === 503) {
          detail = "⚠️ 모델이 아직 준비되지 않았습니다.\n" + detail;
        }
        addErrorMessage(detail);
        return;
      }
      const data = await res.json();
      addBotMessage(data);
    } catch (e) {
      typing.remove();
      addErrorMessage("서버에 연결할 수 없습니다: " + e.message);
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    addUserMessage(text);
    input.value = "";
    send(text);
  });

  // ── 헬스 체크 → 상태 바 ──
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const t = data.services.translation.available;
      const em = data.services.emotion.available;
      statusBar.innerHTML =
        "번역" + statusDot(t) +
        "감정" + statusDot(em);
    } catch (e) {
      statusBar.textContent = "상태 확인 실패";
    }
  }

  function statusDot(on) {
    return '<span class="dot ' + (on ? "on" : "off") + '"></span>' +
      (on ? "준비됨 " : "대기 ");
  }

  checkHealth();
})();
