import { useState, useEffect, useRef, useMemo } from "react";
import type { FormEvent } from "react";

type MatchResult = {
  patient_id: string | number;
  age: number;
  score: number;
  decision: string;
  matched_inclusion: string[];
  matched_exclusion: string[];
  age_ok: boolean;
  diagnosis?: string;
  reason?: string;
  inclusion_score?: number;
  exclusion_score?: number;
  _source?: "pre_filter" | "llm" | "rule_based";
};

type ChatMessage = { role: "user" | "bot"; content: string };
type PipelineStats = {
  pre_filtered: number;
  llm_evaluated: number;
  rule_evaluated: number;
};
type MatchResponse = {
  trial_id: string;
  matches: MatchResult[];
  mode?: string;
  pipeline?: PipelineStats;
};


type EvalKPI = {
  total_patients: number;
  processing_time_ms: number;
  rule_based: { age_pass_rate: number };
  logic_based: { eligible_count: number; eligibility_rate: number };
  performance: { precision: number; throughput_patients_per_sec: number };
};

const API_BASE = "http://127.0.0.1:8000";

function MiniBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.07)", borderRadius: 4, height: 6, overflow: "hidden", marginTop: 8 }}>
      <div style={{ width: `${Math.min(pct, 100)}%`, height: "100%", background: color, borderRadius: 4, transition: "width 1s ease" }} />
    </div>
  );
}

function KpiStrip({ trialId }: { trialId: string }) {
  const [kpi, setKpi] = useState<EvalKPI | null>(null);
  const prevId = useRef("");

  useEffect(() => {
    if (!trialId || trialId === prevId.current) return;
    prevId.current = trialId;
    fetch(`${API_BASE}/evaluate/${encodeURIComponent(trialId)}`)
      .then(r => r.json())
      .then(setKpi)
      .catch(() => { });
  }, [trialId]);

  if (!kpi || !kpi.total_patients) return null;

  const rb = kpi.rule_based;
  const lb = kpi.logic_based;
  const pf = kpi.performance;

  const cards = [
    { icon: "👥", label: "Total", value: String(kpi.total_patients), pct: 100, color: "#38bdf8" },
    { icon: "✅", label: "Eligible", value: String(lb.eligible_count), pct: lb.eligibility_rate, color: "#22c55e" },
    { icon: "📈", label: "Match Rate", value: `${lb.eligibility_rate}%`, pct: lb.eligibility_rate, color: "#a855f7" },
    { icon: "🎂", label: "Age Pass", value: `${rb.age_pass_rate}%`, pct: rb.age_pass_rate, color: "#f59e0b" },
    { icon: "🎯", label: "Precision", value: pf.precision.toFixed(2), pct: pf.precision * 100, color: "#6366f1" },
    { icon: "⚡", label: `${kpi.processing_time_ms}ms`, value: "", pct: Math.min(pf.throughput_patients_per_sec / 10, 100), color: "#2dd4bf" },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: "0.75rem", marginTop: "1rem" }}>
      {cards.map((c, i) => (
        <div key={i} style={{
          background: "rgba(10,20,40,0.85)", border: "1px solid rgba(56,130,246,0.2)",
          borderRadius: 14, padding: "1rem 0.75rem", textAlign: "center",
        }}>
          <div style={{ fontSize: "1.5rem", marginBottom: 4 }}>{c.icon}</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 800, color: c.color }}>{c.value}</div>
          <div style={{ fontSize: "0.72rem", color: "#94a3b8", marginTop: 4, fontWeight: 500 }}>{c.label}</div>
          <MiniBar pct={c.pct} color={c.color} />
        </div>
      ))}
    </div>
  );
}

// ── Summary Stats Boxes ───────────────────────────────────────────────────────
function HybridPipelinePanel({ results }: {
  results: MatchResult[];
  stats: PipelineStats | null;
  total: number;
  mode: string;
}) {
  if (!results.length) return null;
  const { total: t, matched, unmatched } = useMemo(() => {
    const len = results.length;
    const mat = results.filter(r => r.decision === "Eligible").length;
    return { total: len, matched: mat, unmatched: len - mat };
  }, [results]);

  const totalCount = t;
  const boxes = [
    {
      icon: "👥",
      label: "Total Patients",
      value: totalCount,
      color: "#38bdf8",
      border: "rgba(56,189,248,0.4)",
      bg: "rgba(56,189,248,0.08)",
      pct: 100,
    },
    {
      icon: "✅",
      label: "Matched",
      value: matched,
      color: "#22c55e",
      border: "rgba(34,197,94,0.4)",
      bg: "rgba(34,197,94,0.08)",
      pct: totalCount ? (matched / totalCount) * 100 : 0,
    },
    {
      icon: "❌",
      label: "Unmatched",
      value: unmatched,
      color: "#f97316",
      border: "rgba(249,115,22,0.4)",
      bg: "rgba(249,115,22,0.08)",
      pct: totalCount ? (unmatched / totalCount) * 100 : 0,
    },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "0.75rem", marginTop: "1rem" }}>
      {boxes.map((b, i) => (
        <div key={i} style={{
          background: b.bg,
          border: `1px solid ${b.border}`,
          borderRadius: 14,
          padding: "1.1rem 1rem",
          textAlign: "center",
        }}>
          <div style={{ fontSize: "1.8rem", marginBottom: 6 }}>{b.icon}</div>
          <div style={{ fontSize: "2rem", fontWeight: 900, color: b.color, lineHeight: 1 }}>{b.value}</div>
          <div style={{ fontSize: "0.78rem", color: "#94a3b8", marginTop: 6, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".05em" }}>{b.label}</div>
          {/* progress bar */}
          <div style={{ marginTop: 10, height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
            <div style={{ width: `${b.pct}%`, height: "100%", background: b.color, borderRadius: 4, transition: "width 1s ease" }} />
          </div>
          <div style={{ fontSize: "0.7rem", color: "#475569", marginTop: 4 }}>{Math.round(b.pct)}% of total</div>
        </div>
      ))}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
function App() {
  const [patientsFile, setPatientsFile] = useState<File | null>(null);
  const [trialsFile, setTrialsFile] = useState<File | null>(null);
  const [trialId, setTrialId] = useState("");
  const [matchedTrialId, setMatchedTrialId] = useState("");
  const [results, setResults] = useState<MatchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [pipelineStats, setPipelineStats] = useState<PipelineStats | null>(null);
  const [resultMode, setResultMode] = useState<"" | "llm">("llm");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: "bot", content: "Hello, Welcome to Clinical Chat. How may I help you?" },
  ]);

  // Auto-clear patients on every page load / refresh
  useEffect(() => {
    fetch(API_BASE + "/admin/reset-patients", { method: "DELETE" })
      .then(r => r.json())
      .then(d => {
        if (d.patients_deleted > 0) {
          setChatMessages(prev => [...prev, {
            role: "bot",
            content: `🗑️ Database cleared on load (${d.patients_deleted} previous records removed). Please upload your patient CSV to begin.`
          }]);
        }
      })
      .catch(() => { }); // silent if backend not ready yet
  }, []);

  async function uploadFile(file: File | null, url: string, label: string, extraParams?: Record<string, string>) {
    if (!file) { alert(`Select a ${label} file first.`); return; }
    const formData = new FormData();
    formData.append("file", file);
    try {
      setLoading(true);
      const query = extraParams ? "?" + new URLSearchParams(extraParams).toString() : "";
      const resp = await fetch(API_BASE + url + query, { method: "POST", body: formData });
      if (!resp.ok) throw new Error(await resp.text() || resp.statusText);
      const json = await resp.json().catch(() => null);
      const totalInDb = json?.total_in_db;
      const mode = json?.mode ?? "";
      if (totalInDb !== undefined) {
        alert(`✅ ${label} uploaded successfully!\n📊 Rows in file: ${json.rows_inserted}\n🗄️ Total patients in database: ${totalInDb}\n⚙️ Mode: ${mode}`);
      } else {
        alert(`${label} uploaded successfully`);
      }
    } catch (e: any) {
      alert(`${label} upload failed: ${e.message || String(e)}`);
    } finally { setLoading(false); }
  }

  async function runMatch() {
    if (!trialId) { alert("Enter a trial ID to match."); return; }
    setResults([]);
    setResultMode("llm");
    setLoading(true);
    setElapsed(0);
    elapsedRef.current = setInterval(() => setElapsed(s => s + 1), 1000);

    try {
      const resp = await fetch(API_BASE + "/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trial_id: trialId }),
      });
      if (!resp.ok) throw new Error(await resp.text() || resp.statusText);
      const data = await resp.json() as MatchResponse;
      setResults(data.matches);
      setMatchedTrialId(trialId);
      if (data.pipeline) setPipelineStats(data.pipeline);
    } catch (e: any) {
      alert("Match failed: " + (e.message || String(e)));
    } finally {
      setLoading(false);
      if (elapsedRef.current) clearInterval(elapsedRef.current);
    }
  }

  function download(kind: "csv" | "pdf") {
    if (!trialId) { alert("Enter a trial ID first."); return; }
    window.open(API_BASE + `/download/${kind}/${encodeURIComponent(trialId)}`, "_blank");
  }

  async function sendChat(e: FormEvent) {
    e.preventDefault();
    const question = chatInput.trim();
    if (!question) return;
    setChatMessages(prev => [...prev, { role: "user", content: question }]);
    setChatInput("");
    try {
      const resp = await fetch(API_BASE + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!resp.ok) throw new Error(await resp.text() || resp.statusText);
      const data = await resp.json() as { role: string; content: string };
      setChatMessages(prev => [...prev, { role: "bot", content: data.content }]);
    } catch (e: any) {
      setChatMessages(prev => [...prev, { role: "bot", content: "Error: " + (e.message || String(e)) }]);
    }
  }

  const S = {
    card: (extra?: React.CSSProperties): React.CSSProperties => ({
      background: "rgba(15,23,42,0.9)",
      border: "1px solid rgba(148,163,184,0.2)",
      borderRadius: 12,
      padding: "0.75rem",
      ...extra,
    }),
  };

  return (
    <div style={{
      minHeight: "100vh", width: "100vw", overflowX: "hidden",
      background: "linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0ea5e9 100%)",
      color: "white", padding: "2rem", boxSizing: "border-box",
      fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }}>
      <div style={{ maxWidth: 1600, margin: "0 auto", background: "rgba(15,23,42,0.9)", borderRadius: 20, padding: "2rem", boxShadow: "0 24px 48px rgba(15,23,42,0.8)", border: "1px solid rgba(148,163,184,0.35)" }}>

        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
          <div>
            <h1 style={{ fontSize: "2rem", margin: 0 }}>🧬 Clinical Match</h1>

          </div>
          <a href="/"
            style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "0.45rem 1.1rem", borderRadius: 999, border: "1px solid rgba(56,189,248,0.35)", background: "rgba(56,189,248,0.08)", color: "#38bdf8", fontWeight: 600, fontSize: "0.85rem", textDecoration: "none", transition: "background 0.2s" }}
            onMouseOver={e => (e.currentTarget.style.background = "rgba(56,189,248,0.18)")}
            onMouseOut={e => (e.currentTarget.style.background = "rgba(56,189,248,0.08)")}
          >
            ← Home
          </a>
        </header>


        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: "1.5rem" }}>

          {/* ── Left Column ── */}
          <div>
            {/* Upload cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1.25rem" }}>
              <div style={{ ...S.card(), border: "1px solid rgba(59,130,246,0.5)", background: "linear-gradient(135deg,rgba(56,189,248,.18),rgba(37,99,235,.12))" }}>
                <h3 style={{ marginTop: 0, fontSize: "1.1rem", marginBottom: "0.75rem" }}>Patients CSV</h3>
                <input type="file" accept=".csv" onChange={e => setPatientsFile(e.target.files?.[0] ?? null)} style={{ fontSize: "0.85rem" }} />
                <button onClick={() => uploadFile(patientsFile, "/upload/patients", "Patients CSV", { replace: "true" })} disabled={loading}
                  style={{ marginTop: "0.75rem", padding: "0.45rem 1rem", borderRadius: 999, border: "none", background: "linear-gradient(135deg,#38bdf8,#2563eb)", color: "white", fontSize: "0.85rem", cursor: "pointer", fontWeight: 600 }}>
                  Upload
                </button>
              </div>
              <div style={{ ...S.card(), border: "1px solid rgba(52,211,153,0.5)", background: "linear-gradient(135deg,rgba(45,212,191,.18),rgba(59,130,246,.12))" }}>
                <h3 style={{ marginTop: 0, fontSize: "1.1rem", marginBottom: "0.75rem" }}>Trials CSV</h3>
                <input type="file" accept=".csv" onChange={e => setTrialsFile(e.target.files?.[0] ?? null)} style={{ fontSize: "0.85rem" }} />
                <button onClick={() => uploadFile(trialsFile, "/upload/trials", "Trials CSV")} disabled={loading}
                  style={{ marginTop: "0.75rem", padding: "0.45rem 1rem", borderRadius: 999, border: "none", background: "linear-gradient(135deg,#22c55e,#16a34a)", color: "white", fontSize: "0.85rem", cursor: "pointer", fontWeight: 600 }}>
                  Upload
                </button>
              </div>
            </div>

            {/* Match controls */}
            <div style={{ ...S.card(), marginBottom: "1rem" }}>
              <h3 style={{ marginTop: 0, fontSize: "1.1rem", marginBottom: "0.75rem" }}>Run Matching</h3>
              <div style={{ display: "flex", gap: "0.75rem", marginBottom: 12 }}>
                <input placeholder="Trial ID (e.g. NCT123456)" value={trialId} onChange={e => setTrialId(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && runMatch()}
                  style={{ flex: 1, padding: "0.55rem 0.9rem", borderRadius: 999, border: "1px solid rgba(148,163,184,0.7)", backgroundColor: "rgba(15,23,42,0.9)", color: "white", fontSize: "0.9rem", fontFamily: "inherit" }} />
                <button onClick={runMatch} disabled={loading}
                  style={{ padding: "0.55rem 1.2rem", borderRadius: 999, border: "none", background: loading ? "#334155" : "linear-gradient(135deg,#6366f1,#0ea5e9)", color: "white", fontSize: "0.9rem", fontWeight: 700, cursor: loading ? "not-allowed" : "pointer", transition: "background 0.3s" }}>
                  {loading ? `⏳ ${elapsed}s` : "Match"}
                </button>
              </div>
              <div style={{ display: "flex", gap: 10 }}>
                <button onClick={() => download("csv")} disabled={!trialId}
                  style={{ padding: "0.4rem 1rem", borderRadius: 999, border: "none", backgroundColor: "#0ea5e9", color: "white", fontSize: "0.82rem", cursor: "pointer", fontWeight: 600 }}>
                  ⬇ CSV
                </button>
                <button onClick={() => download("pdf")} disabled={!trialId}
                  style={{ padding: "0.4rem 1rem", borderRadius: 999, border: "none", backgroundColor: "#38bdf8", color: "white", fontSize: "0.82rem", cursor: "pointer", fontWeight: 600 }}>
                  ⬇ PDF
                </button>
              </div>
            </div>

            {/* Match Results */}
            <div style={{ ...S.card(), maxHeight: 320, overflow: "auto", marginBottom: "0.75rem" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.75rem" }}>
                <h3 style={{ marginTop: 0, fontSize: "1.1rem", margin: 0 }}>Results</h3>
                {results.length > 0 && (
                  <span style={{ fontSize: "0.72rem", fontWeight: 700, padding: "2px 10px", borderRadius: 999, background: "rgba(34,197,94,0.18)", border: "1px solid rgba(34,197,94,0.5)", color: "#4ade80" }}>
                    🤖 AI Results
                  </span>
                )}
              </div>
              {useMemo(() => {
                if (loading && results.length === 0) {
                  return (
                    <div style={{ padding: "1rem", background: "rgba(99,102,241,0.1)", borderRadius: 8, border: "1px solid rgba(99,102,241,0.3)", marginBottom: "0.75rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                        <span style={{ fontSize: "1.4rem" }}>🦙</span>
                        <div>
                          <div style={{ fontWeight: 700, color: "#a5b4fc", fontSize: "0.9rem" }}>Fetching results… {elapsed}s elapsed</div>
                          <div style={{ color: "#64748b", fontSize: "0.78rem", marginTop: 2 }}>LLM evaluating medical terminology · Age rules applied instantly</div>
                        </div>
                      </div>
                      <div style={{ marginTop: "0.75rem", height: 4, background: "rgba(255,255,255,0.07)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ height: "100%", background: "linear-gradient(90deg,#6366f1,#38bdf8)", borderRadius: 4, width: `${Math.min((elapsed / 120) * 100, 95)}%`, transition: "width 1s linear" }} />
                      </div>
                    </div>
                  );
                }
                
                if (!loading && results.length === 0) {
                  return <p style={{ fontSize: "0.85rem", color: "#9ca3af" }}>No results yet. Upload data and run matching.</p>;
                }
                
                if (results.length > 0) {
                  const llmRows  = results.filter(r => r._source !== "pre_filter").sort((a, b) => b.score - a.score);
                  const ageRows  = results.filter(r => r._source === "pre_filter");
                  const cols = ["Patient", "Age", "Score", "Decision", "Matched Keywords"];
                  const renderRow = (r: MatchResult, i: number) => (
                    <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                      <td style={{ padding: "0.4rem 0.6rem", fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: "0.4rem 0.6rem" }}>{r.age}</td>
                      <td style={{ padding: "0.4rem 0.6rem", color: "#38bdf8", fontWeight: 600 }}>{(r.score ?? 0).toFixed(2)}</td>
                      <td style={{ padding: "0.4rem 0.6rem", color: r.decision === "Eligible" ? "#22c55e" : "#f97316", fontWeight: 600 }}>{r.decision}</td>
                      <td style={{ padding: "0.4rem 0.6rem", color: "#94a3b8", fontSize: "0.78rem" }}>{(r.matched_inclusion || []).join(", ") || "—"}</td>
                    </tr>
                  );
                  return (
                    <div>
                      {/* Phase 2: LLM evaluated */}
                      {llmRows.length > 0 && (
                        <div style={{ marginBottom: "0.75rem" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0.4rem 0.7rem", background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.35)", borderRadius: 8, marginBottom: "0.4rem" }}>
                            <span style={{ fontSize: "1rem" }}>🤖</span>
                            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#a5b4fc", textTransform: "uppercase", letterSpacing: ".05em" }}>
                              Phase 2 — LLM Semantic Evaluation
                            </span>
                            <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "#6366f1", fontWeight: 600 }}>{llmRows.length} patients</span>
                          </div>
                          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                            <thead>
                              <tr>{cols.map(h => (
                                <th key={h} style={{ textAlign: "left", padding: "0.35rem 0.6rem", color: "#64748b", borderBottom: "1px solid rgba(99,102,241,0.2)", fontWeight: 600, fontSize: "0.75rem", textTransform: "uppercase" }}>{h}</th>
                              ))}</tr>
                            </thead>
                            <tbody>{llmRows.map(renderRow)}</tbody>
                          </table>
                        </div>
                      )}

                      {/* Phase 1: Age-rejected */}
                      {ageRows.length > 0 && (
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0.4rem 0.7rem", background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 8, marginBottom: "0.4rem" }}>
                            <span style={{ fontSize: "1rem" }}>⚖️</span>
                            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#fbbf24", textTransform: "uppercase", letterSpacing: ".05em" }}>
                              Phase 1 — Rejected by Age Rule
                            </span>
                            <span style={{ marginLeft: "auto", fontSize: "0.72rem", color: "#92400e", fontWeight: 600 }}>{ageRows.length} patients</span>
                          </div>
                          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", opacity: 0.75 }}>
                            <thead>
                              <tr>{cols.map(h => (
                                <th key={h} style={{ textAlign: "left", padding: "0.35rem 0.6rem", color: "#64748b", borderBottom: "1px solid rgba(245,158,11,0.2)", fontWeight: 600, fontSize: "0.75rem", textTransform: "uppercase" }}>{h}</th>
                              ))}</tr>
                            </thead>
                            <tbody>{ageRows.map(renderRow)}</tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              }, [results, loading, elapsed])}
            </div>

            {/* ── Hybrid Pipeline Panel ── */}
            <HybridPipelinePanel results={results} stats={pipelineStats} total={results.length} mode={resultMode} />

            {/* ── KPI Summary Strip ── */}
            {matchedTrialId && <KpiStrip trialId={matchedTrialId} />}

          </div>

          {/* ── Right Column: Chatbot ── */}
          <div style={{ ...S.card(), background: "radial-gradient(circle at top,rgba(96,165,250,.35),transparent 55%),rgba(15,23,42,.95)", border: "1px solid rgba(59,130,246,0.5)", display: "flex", flexDirection: "column" }}>
            <h3 style={{ marginTop: 0, fontSize: "1rem" }}>💬 Clinical Match Chatbot</h3>
            <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem", borderRadius: 8, backgroundColor: "rgba(15,23,42,0.9)", marginBottom: "0.5rem", border: "1px solid rgba(148,163,184,0.5)", minHeight: 200, maxHeight: 500 }}>
              {chatMessages.map((m, i) => (
                <div key={i} style={{ marginBottom: "0.5rem", textAlign: m.role === "user" ? "right" : "left" }}>
                  <div style={{
                    display: "inline-block", padding: "0.4rem 0.6rem", fontSize: "0.8rem", whiteSpace: "pre-line", maxWidth: "90%",
                    borderRadius: m.role === "user" ? "14px 14px 2px 14px" : "14px 14px 14px 2px",
                    backgroundColor: m.role === "user" ? "#1d4ed8" : "rgba(15,23,42,0.9)",
                    border: m.role === "user" ? "1px solid rgba(191,219,254,0.6)" : "1px solid rgba(148,163,184,0.6)",
                  }}>{m.content}</div>
                </div>
              ))}
            </div>
            <form onSubmit={sendChat} style={{ display: "flex", gap: "0.5rem" }}>
              <input value={chatInput} onChange={e => setChatInput(e.target.value)}
                placeholder="e.g. Find patients eligible for trial NCT02827175 | Average age | Top diagnoses"
                style={{ flex: 1, padding: "0.4rem 0.6rem", borderRadius: 999, border: "1px solid rgba(148,163,184,0.7)", backgroundColor: "rgba(15,23,42,0.9)", color: "white", fontSize: "0.8rem", fontFamily: "inherit" }} />
              <button type="submit"
                style={{ padding: "0.4rem 0.9rem", borderRadius: 999, border: "none", background: "linear-gradient(135deg,#38bdf8,#6366f1)", color: "white", fontSize: "0.8rem", cursor: "pointer" }}>
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
