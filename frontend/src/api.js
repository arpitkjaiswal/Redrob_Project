// frontend/src/api.js
// Thin wrapper around the FastAPI backend

const BASE = '/api'

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

// ── Candidates ──────────────────────────────────────────────
export function listCandidates(params = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== '' && q.set(k, v))
  return req(`/candidates?${q}`)
}

export function getCandidate(id) {
  return req(`/candidates/${id}`)
}

export function getCandidateFeatures(id) {
  return req(`/candidates/${id}/features`)
}

// ── Jobs ────────────────────────────────────────────────────
export function listJobs() {
  return req('/jobs')
}

export function getJob(id) {
  return req(`/jobs/${id}`)
}

export function createJob(body) {
  return req('/jobs', { method: 'POST', body: JSON.stringify(body) })
}

// ── Rankings ────────────────────────────────────────────────
export function getRankings(jobId, params = {}) {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== '' && q.set(k, v))
  return req(`/jobs/${jobId}/rankings?${q}`)
}

export function getCandidateRanking(jobId, candidateId) {
  return req(`/jobs/${jobId}/rankings/${candidateId}`)
}

// ── Pipeline ────────────────────────────────────────────────
export function triggerPipeline(body) {
  return req('/pipeline/run', { method: 'POST', body: JSON.stringify(body) })
}

export function listPipelineRuns(jobId) {
  const q = jobId ? `?job_id=${jobId}` : ''
  return req(`/pipeline/runs${q}`)
}

export function getPipelineRun(runId) {
  return req(`/pipeline/runs/${runId}`)
}

// ── Health ──────────────────────────────────────────────────
export async function checkHealth() {
  try {
    const res = await fetch('/health')
    return res.ok
  } catch {
    return false
  }
}
