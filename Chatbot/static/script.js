/* ===================================================================
   script.js — SmartBot FAQ Chatbot Frontend Logic
   =================================================================== */

// ── DOM refs ──────────────────────────────────────────────────── //
const chatWindow     = document.getElementById("chatWindow");
const userInput      = document.getElementById("userInput");
const sendBtn        = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const chips          = document.getElementById("chips");

// ── Utility: current time string ─────────────────────────────── //
function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Scroll to bottom ──────────────────────────────────────────── //
function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}

// ── Append a message bubble ───────────────────────────────────── //
function appendMessage(role, text, meta = {}) {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message", role === "user" ? "user-message" : "bot-message");

  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  if (role === "bot") {
    // Avatar
    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.textContent = "🤖";
    wrapper.appendChild(avatar);

    // Text (render line breaks)
    bubble.innerHTML = text
      .split("\n")
      .map(line => `<p>${line}</p>`)
      .join("");

    // Confidence badge
    if (meta.score !== undefined && meta.score > 0) {
      const badge = document.createElement("span");
      badge.classList.add("confidence-badge");
      badge.textContent = `Match confidence: ${Math.round(meta.score * 100)}%`;
      bubble.appendChild(badge);
    }

    // Timestamp
    const ts = document.createElement("span");
    ts.classList.add("timestamp");
    ts.textContent = now();
    bubble.appendChild(ts);
  } else {
    // User bubble
    bubble.textContent = text;
    const ts = document.createElement("span");
    ts.classList.add("timestamp");
    ts.textContent = now();
    bubble.appendChild(ts);
  }

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  scrollToBottom();
}

// ── Show / hide typing indicator ──────────────────────────────── //
function showTyping()  {
  typingIndicator.style.display = "flex";
  scrollToBottom();
}
function hideTyping()  { typingIndicator.style.display = "none"; }

// ── Disable / enable input during fetch ───────────────────────── //
function setInputEnabled(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
  if (enabled) userInput.focus();
}

// ── Core: send message to /chat and display response ─────────── //
async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  // Clear chips (one-time)
  if (chips) chips.remove();

  appendMessage("user", trimmed);
  userInput.value = "";
  autoResize();
  setInputEnabled(false);
  showTyping();

  try {
    const res = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: trimmed }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    hideTyping();
    appendMessage("bot", data.answer, { score: data.score });

  } catch (err) {
    hideTyping();
    appendMessage("bot", "⚠️ Sorry, I couldn't reach the server. Please try again.");
    console.error("Chat error:", err);
  } finally {
    setInputEnabled(true);
  }
}

// ── Auto-resize textarea ──────────────────────────────────────── //
function autoResize() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
}

// ── Event listeners ───────────────────────────────────────────── //

// Send button click
sendBtn.addEventListener("click", () => sendMessage(userInput.value));

// Enter to send (Shift+Enter for newline)
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(userInput.value);
  }
});

// Auto-resize on input
userInput.addEventListener("input", autoResize);

// Suggestion chips
if (chips) {
  chips.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
      sendMessage(e.target.textContent);
    }
  });
}

// ── Focus input on load ───────────────────────────────────────── //
userInput.focus();
