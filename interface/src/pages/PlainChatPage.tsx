import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import * as api from "../api";

type PlainChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string; // ISO
  image_data_url?: string; // optional data URL (from server) or blob URL (optimistic)
};

type PlainChatSession = {
  id: string;
  name: string;
  createdAt: string; // ISO
  lastMessageAt: string; // ISO
  messageCount: number;
};

type SendResponse = {
  session_id: string;
  output: string;
  history: PlainChatMessage[];
};

const DEFAULT_MODEL = "gpt-5.4";
const SESSION_ERROR_MESSAGE = "Failed to load sessions.";
const HISTORY_ERROR_MESSAGE = "Failed to load history.";
const CREATE_SESSION_ERROR_MESSAGE = "Failed to create new session.";
const REQUEST_ERROR_MESSAGE = "Request failed.";
const LOGIN_ERROR_EMPTY = "Please enter a username";
const LOGIN_ERROR_LENGTH = "Username cannot exceed 20 characters";
const LOGIN_ERROR_FORMAT = "Username can only contain letters, numbers, and underscores";

const MODEL_OPTIONS = [
  { value: "gpt-5.4", label: "GPT-5.4" },
  { value: "gpt-5", label: "GPT-5" },
  { value: "gpt-5-mini-2025-08-07", label: "GPT-5 Mini" },
  { value: "gpt-5-nano", label: "GPT-5 Nano" },
] as const;

export default function PlainChatPage() {
  const [username, setUsername] = useState<string>(() => {
    const fromApi = (api.getUsername() || "").trim();
    if (fromApi) return fromApi;
    return (localStorage.getItem("chart_retrieval_username") || "").trim();
  });
  const isLoggedIn = Boolean(username);
  const [sessions, setSessions] = useState<PlainChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => Date.now().toString());
  const [messages, setMessages] = useState<PlainChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [userImage, setUserImage] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [modelName, setModelName] = useState<string>(DEFAULT_MODEL);
  const [error, setError] = useState<string>("");
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null);
  const [loginInput, setLoginInput] = useState<string>(username);
  const [loginError, setLoginError] = useState<string>("");

  const endRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => endRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    const syncUsername = () => {
      const fromApi = (api.getUsername() || "").trim();
      const fromStorage = (localStorage.getItem("chart_retrieval_username") || "").trim();
      const nextUsername = fromApi || fromStorage;
      setUsername((prev) => (prev === nextUsername ? prev : nextUsername));
      setLoginInput((prev) => (prev === nextUsername ? prev : nextUsername));
    };

    syncUsername();
    window.addEventListener("storage", syncUsername);
    window.addEventListener("focus", syncUsername);
    document.addEventListener("visibilitychange", syncUsername);

    return () => {
      window.removeEventListener("storage", syncUsername);
      window.removeEventListener("focus", syncUsername);
      document.removeEventListener("visibilitychange", syncUsername);
    };
  }, []);
    
  useEffect(() => {
    if (!userImage) {
      setPreviewImageUrl(null);
      return;
    }

    const url = URL.createObjectURL(userImage);
    setPreviewImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [userImage]);

  useEffect(() => {
    scrollToBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, loading]);

  async function loadSessions() {
    try {
      const res = await axios.get("/api/plainchat/sessions", { params: { username } });
      const nextSessions = res.data.sessions || [];
      setSessions(nextSessions);
      // if current is missing, pick first
      if (nextSessions.length) {
        const exists = nextSessions.some((s: PlainChatSession) => s.id === currentSessionId);
        if (!exists) {
          setCurrentSessionId(nextSessions[0].id);
        }
      }
    } catch (error) {
      console.error(error);
      setError(SESSION_ERROR_MESSAGE);
    }
  }

  async function loadHistory(sessionId: string) {
    try {
      const res = await axios.get("/api/plainchat/history", { params: { username, session_id: sessionId } });
      setMessages(res.data.history || []);
    } catch (error) {
      console.error(error);
      setError(HISTORY_ERROR_MESSAGE);
      setMessages([]);
    }
  }

  useEffect(() => {
    if (!isLoggedIn) return;
    loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, username]);

  useEffect(() => {
    if (!isLoggedIn) return;
    loadHistory(currentSessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionId, isLoggedIn, username]);

  async function createNewSession() {
    try {
      const res = await axios.post("/api/plainchat/new_session", { username });
      const sid = res.data.session_id as string;
      setCurrentSessionId(sid);
      await loadSessions();
      await loadHistory(sid);
    } catch (error) {
      console.error(error);
      setError(CREATE_SESSION_ERROR_MESSAGE);
    }
  }

  function onPickImage(file: File | null) {
    if (!file) return;
    setUserImage(file);
  }

  async function send() {
    const text = input.trim();
    if ((!text && !userImage) || loading) return;

    setLoading(true);
    setError("");

    const optimisticId = Date.now().toString();
    const optimisticImgUrl = userImage ? URL.createObjectURL(userImage) : undefined;

    // optimistic UI: append user msg
    const userMsg: PlainChatMessage = {
      id: optimisticId,
      role: "user",
      text: text || "(image)",
      timestamp: new Date().toISOString(),
      ...(optimisticImgUrl ? { image_data_url: optimisticImgUrl } : {}),
    };
    setMessages((prev) => [...prev, userMsg]);

    // clear inputs early
    setInput("");
    const imageToSend = userImage;
    setUserImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    try {
      let res;
      if (imageToSend) {
        // Use multipart endpoint when image is present
        const fd = new FormData();
        fd.append("username", username);
        fd.append("session_id", currentSessionId);
        fd.append("user_text", text || "");
        fd.append("model_name", modelName);
        fd.append("user_image", imageToSend);
        res = await axios.post<SendResponse>("/api/plainchat/send_form", fd, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      } else {
        // JSON endpoint for pure text
        res = await axios.post<SendResponse>("/api/plainchat/send", {
          username,
          session_id: currentSessionId,
          user_text: text,
          model_name: modelName,
        });
      }

      // Replace with authoritative history from server (full history for display)
      setMessages(res.data.history || []);
      // refresh sessions list (messageCount, lastMessageAt)
      await loadSessions();
    } catch (error) {
      console.error(error);
      if (axios.isAxiosError(error)) {
        const errorDetail =
          typeof error.response?.data?.detail === "string"
            ? error.response.data.detail
            : REQUEST_ERROR_MESSAGE;
        setError(errorDetail);
      } else {
        setError(REQUEST_ERROR_MESSAGE);
      }
      // keep optimistic user message
    } finally {
      setLoading(false);
      // cleanup blob URL to avoid leaks
      if (optimisticImgUrl) URL.revokeObjectURL(optimisticImgUrl);
    }
  }

  function handleLoginSubmit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = loginInput.trim();
    if (!trimmed) {
      setLoginError(LOGIN_ERROR_EMPTY);
      return;
    }
    if (trimmed.length > 20) {
      setLoginError(LOGIN_ERROR_LENGTH);
      return;
    }
    if (!/^[a-zA-Z0-9_]+$/.test(trimmed)) {
      setLoginError(LOGIN_ERROR_FORMAT);
      return;
    }
    api.setUsername(trimmed);
    setUsername(trimmed);
    setLoginError("");
  }

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-80 bg-gray-900 text-white flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Plain Chat</div>
              <div className="text-lg font-semibold">Standalone</div>
            </div>
            <a
              href="#/"
              className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600"
              title="Back to main app"
            >
              Main App
            </a>
          </div>

          <div className="mt-3 text-xs text-gray-400">
            user: <span className="text-gray-200">{username || "Not logged in"}</span>
          </div>

          <button
            onClick={createNewSession}
            className="w-full mt-3 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-md text-sm font-medium transition-colors"
          >
            + New Session
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {sessions.map((s) => (
            <div
              key={s.id}
              onClick={() => setCurrentSessionId(s.id)}
              className={`p-3 mb-2 rounded-lg cursor-pointer transition-colors ${
                s.id === currentSessionId ? "bg-blue-600 text-white" : "hover:bg-gray-700 text-gray-300"
              }`}
            >
              <div className="text-sm font-medium truncate">{s.name}</div>
              <div className="text-xs opacity-80 mt-1">
                {s.messageCount} msgs · {new Date(s.lastMessageAt).toLocaleString()}
              </div>
            </div>
          ))}
          {!sessions.length && <div className="text-gray-500 text-sm p-3">No sessions yet.</div>}
        </div>

        <div className="p-3 border-t border-gray-700">
          <label className="text-xs text-gray-400">Model</label>
          <select
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="mt-1 w-full text-sm bg-gray-800 border border-gray-700 rounded px-2 py-2"
          >
            {MODEL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {!isLoggedIn ? (
          <div className="h-full flex items-center justify-center bg-white">
            <div className="max-w-md w-full border border-gray-200 rounded-lg p-6 shadow-sm text-center">
              <div className="text-lg font-semibold text-gray-900">Please login first</div>
              <div className="text-sm text-gray-600 mt-2">Plain Chat uses the same username as the main app. You can also login directly here.</div>

              <form onSubmit={handleLoginSubmit} className="mt-4 space-y-3 text-left">
                <input
                  type="text"
                  value={loginInput}
                  onChange={(event) => setLoginInput(event.target.value)}
                  placeholder="Please enter a username"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                {loginError && <div className="text-sm text-red-600">{loginError}</div>}
                <button
                  type="submit"
                  className="w-full px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
                >
                  Log in to Plain Chat
                </button>
              </form>

              <a href="#/" className="inline-block mt-4 text-sm text-blue-600 hover:text-blue-700">
                Or go back to main login
              </a>
            </div>
          </div>
        ) : (
          <>
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="text-xl font-bold text-gray-900">🧪 Plain Chat</div>
          <div className="text-sm text-gray-600">
            Model sees only last 2 turns; UI shows full history; history stored on backend.
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.map((m) => (
            <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-3xl rounded-lg p-4 shadow-sm ${
                  m.role === "user" ? "bg-blue-600 text-white" : "bg-white border border-gray-200 text-gray-800"
                }`}
              >
                <div className={`text-xs mb-2 ${m.role === "user" ? "text-blue-100" : "text-gray-500"}`}>
                  {m.role === "user" ? "👤 user" : "🤖 assistant"} · {new Date(m.timestamp).toLocaleTimeString()}
                </div>

                {m.image_data_url && (
                  <div className="mb-3">
                    <img
                      src={m.image_data_url}
                      alt="uploaded"
                      className="max-w-xs rounded-md border border-white/20"
                    />
                  </div>
                )}

                <pre className="whitespace-pre-wrap font-sans">{m.text}</pre>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-gray-600 text-sm">Thinking…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={endRef} />
        </div>

        <div className="bg-white border-t border-gray-200 px-6 py-4">
          {error && (
            <div className="mb-3 text-sm text-red-600 bg-red-50 border border-red-100 rounded px-3 py-2">
              {error}
            </div>
          )}

          {userImage && (
            <div className="mb-3 flex items-center space-x-3">
              {previewImageUrl && <img src={previewImageUrl} alt="preview" className="w-12 h-12 object-cover rounded border border-gray-200" />}
              <div className="text-sm text-gray-700 truncate max-w-[60%]">{userImage.name}</div>
              <button
                onClick={() => {
                  setUserImage(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="text-sm text-red-600 hover:text-red-700"
              >
                Remove
              </button>
            </div>
          )}

          <div className="flex items-end space-x-3">
            {/* Image upload */}
            <div className="flex-shrink-0">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                id="plain-chat-image-upload"
                onChange={(e) => onPickImage(e.target.files?.[0] || null)}
                disabled={loading}
              />
              <label
                htmlFor="plain-chat-image-upload"
                className={`cursor-pointer flex items-center justify-center w-10 h-10 rounded-full transition-colors ${
                  loading ? "bg-gray-100 text-gray-400 cursor-not-allowed" : "bg-gray-100 hover:bg-gray-200 text-gray-700"
                }`}
                title="Upload image"
              >
                🖼️
              </label>
            </div>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              rows={3}
              style={{ minHeight: "72px", maxHeight: "180px" }}
              disabled={loading}
            />
            <button
              onClick={send}
              disabled={loading || (!input.trim() && !userImage)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            {userImage ? "Image will be sent with your next message." : "You can optionally attach an image."}
          </div>
        </div>
          </>
        )}
      </div>
    </div>
  );
}
