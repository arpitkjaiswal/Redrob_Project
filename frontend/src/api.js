// frontend/src/api.js

// Backend URL
const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  "https://redrob-project-1.onrender.com";

const BASE = `${API_BASE}/api`;
const HEALTH_URL = `${API_BASE}/health`;

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!res.ok) {
    let err = {};
    try {
      err = await res.json();
    } catch {
      err = { detail: res.statusText };
    }
    throw new Error(err.detail || res.statusText);
  }

  return res.json();
}

// Candidates
export const listCandidates = (params = {}) => {
  const q = new URLSearchParams(params).toString();
  return req(`/candidates${q ? `?${q}` : ""}`);
};

export const getCandidate = (id) => req(`/candidates/${id}`);

export const getCandidateFeatures = (id) =>
  req(`/candidates/${id}/features`);

// Jobs
export const listJobs = () => req("/jobs");

export const getJob = (id) => req(`/jobs/${id}`);

export const createJob = (body) =>
  req("/jobs", {
    method: "POST",
    body: JSON.stringify(body),
  });

// Rankings
export const getRankings = (jobId, params = {}) => {
  const q = new URLSearchParams(params).toString();
  return req(`/jobs/${jobId}/rankings${q ? `?${q}` : ""}`);
};

export const getCandidateRanking = (jobId, candidateId) =>
  req(`/jobs/${jobId}/rankings/${candidateId}`);

// Pipeline
export const triggerPipeline = (body) =>
  req("/pipeline/run", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const listPipelineRuns = (jobId) =>
  req(`/pipeline/runs${jobId ? `?job_id=${jobId}` : ""}`);

export const getPipelineRun = (runId) =>
  req(`/pipeline/runs/${runId}`);

// Health
export async function checkHealth() {
  try {
    const res = await fetch(HEALTH_URL);
    return res.ok;
  } catch {
    return false;
  }
}
