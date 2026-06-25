import { useEffect, useState } from 'react'
import { listJobs, createJob } from '../api'
import { Spinner, Badge, EmptyState } from '../components/UI'
import { Link } from 'react-router-dom'

function CreateJobModal({ onClose, onCreate }) {
  const [form, setForm] = useState({ job_id: '', title: '', description: '' })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.job_id || !form.title || !form.description) return
    setSubmitting(true)
    setError(null)
    try {
      const job = await createJob(form)
      onCreate(job)
      onClose()
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-header">
          <span className="modal-title">Create job posting</span>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="form-group">
            <label className="form-label">Job ID <span style={{ color: 'var(--text3)' }}>(unique slug)</span></label>
            <input
              type="text"
              placeholder="e.g. senior-ai-eng-2024"
              value={form.job_id}
              onChange={e => set('job_id', e.target.value.toLowerCase().replace(/\s+/g, '-'))}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Job title</label>
            <input
              type="text"
              placeholder="e.g. Senior AI Engineer"
              value={form.title}
              onChange={e => set('title', e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Job description</label>
            <textarea
              placeholder="Paste the full job description here. The ML engine will use it to compute semantic similarity scores."
              value={form.description}
              onChange={e => set('description', e.target.value)}
              style={{ minHeight: 120 }}
              required
            />
          </div>

          {error && <div className="error-box">{error}</div>}

          <div className="modal-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? <><Spinner /> Creating…</> : 'Create job'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Jobs() {
  const [jobs, setJobs] = useState(null)
  const [error, setError] = useState(null)
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    listJobs().then(setJobs).catch(e => setError(e.message))
  }, [])

  const statusColor = s => ({ done: 'green', running: 'amber', failed: 'red', pending: 'gray' }[s] || 'gray')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h2>Jobs</h2>
          <p>Each job posting represents a role you're sourcing candidates for</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}>+ New job</button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {jobs === null ? (
        <div className="loading-row"><Spinner /> Loading jobs…</div>
      ) : jobs.length === 0 ? (
        <div className="card">
          <EmptyState icon="📋" message="No jobs yet. Create one to get started — you'll need a job before you can run the ranking pipeline." />
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Title</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(j => (
                <tr key={j.job_id}>
                  <td><span className="td-mono">{j.job_id}</span></td>
                  <td style={{ fontWeight: 600 }}>{j.title}</td>
                  <td><Badge label={j.status} type={statusColor(j.status)} /></td>
                  <td className="td-mono">{j.created_at?.slice(0, 16) ?? '—'}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <Link to={`/rankings?job=${j.job_id}`} className="btn btn-secondary btn-sm">Rankings</Link>
                      <Link to={`/pipeline?job=${j.job_id}`} className="btn btn-secondary btn-sm">Run pipeline</Link>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showModal && (
        <CreateJobModal
          onClose={() => setShowModal(false)}
          onCreate={job => setJobs(prev => [job, ...(prev || [])])}
        />
      )}
    </div>
  )
}
