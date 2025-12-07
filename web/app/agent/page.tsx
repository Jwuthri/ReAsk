'use client';

import { useState, useEffect } from 'react';
import styles from './page.module.css';
import Header from '@/components/Header';
import AnalysisSelector, { AnalysisType } from '@/components/AnalysisSelector';
import AgentTraceViewer from '@/components/AgentTraceViewer';
import StreamingAnalysisTable from '@/components/StreamingAnalysisTable';
import ConversationList, { DatasetInput } from '@/components/ConversationList';
import DatasetSummaryTable from '@/components/DatasetSummaryTable';
import Modal from '@/components/Modal';
import {
  AgentTraceInput,
  AgentAnalysisResults,
  analyzeAgentTraceStream,
  AgentAnalysisRequest,
  AgentAnalysisStreamEvent,
  SavedTrace,
  listAgentTraces,
  saveAgentTrace,
  getAgentTrace,
  deleteAgentTrace,
  getTraceTask,
  JobStatus,
  JobListItem,
  startBackgroundJob,
  getJobStatus,
  listJobs,
  retryJob,
  AgentSessionInput,
  AgentDef,
  ToolDefinition,
} from '@/lib/api';
import { EXAMPLE_TRACES, MULTI_AGENT_EXAMPLES, DATASET_EXAMPLES } from './examples';

export default function AgentAnalysisPage() {
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisType>('full_all');
  const [traceInput, setTraceInput] = useState<string>('');
  const [parsedTrace, setParsedTrace] = useState<AgentTraceInput | null>(null);
  const [parsedDataset, setParsedDataset] = useState<DatasetInput | null>(null);
  const [selectedConversationIndex, setSelectedConversationIndex] = useState<number>(0);
  const [results, setResults] = useState<AgentAnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
    turnCurrent?: number;
    turnTotal?: number;
    currentAnalysis?: string;
    turnResults?: Array<{
      step_index: number;
      is_bad: boolean;
      detection_type: string;
      confidence: number;
      reason: string;
    }>;
  } | null>(null);

  // Saved traces state
  const [savedTraces, setSavedTraces] = useState<SavedTrace[]>([]);
  const [loadingTraces, setLoadingTraces] = useState(true);
  const [currentTraceId, setCurrentTraceId] = useState<number | null>(null);

  // Background job state
  const [runInBackground, setRunInBackground] = useState(false);
  const [activeJobs, setActiveJobs] = useState<JobListItem[]>([]);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);

  // Load saved traces and jobs on mount
  useEffect(() => {
    loadSavedTraces();
    loadActiveJobs().then(() => {
      // Auto-resume first active job if any
    });
  }, []);

  // Auto-resume active job when returning to page (only if no other activity)
  useEffect(() => {
    if (activeJobs.length > 0 && !currentJobId && !loading && !results) {
      const firstActiveJob = activeJobs[0];
      resumeJob(firstActiveJob.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobs.length]);

  // Resume watching a job
  const resumeJob = async (jobId: number) => {
    try {
      const status = await getJobStatus(jobId);
      if (status.status === 'running' || status.status === 'pending') {
        setCurrentJobId(jobId);
        setLoading(true);
        const turnResults = status.turn_results || [];
        setProgress({
          current: status.current_step,
          total: status.total_steps,
          currentAnalysis: status.current_analysis || 'Resuming...',
          turnResults: turnResults,
          turnCurrent: turnResults.length,
          turnTotal: Math.max(turnResults.length + 5, 10),
        });
      } else if (status.status === 'completed' && status.result) {
        // Job already done, show results
        setResults(status.result);
        setCurrentTraceId(status.trace_id);
        loadActiveJobs();
      } else if (status.status === 'failed') {
        // Show error with retry option
        setError(`Job failed: ${status.error_message}. Click retry to resume from turn ${(status.last_successful_turn || 0) + 1}.`);
      }
    } catch (err) {
      console.error('Failed to resume job:', err);
    }
  };

  // Poll for active job status
  useEffect(() => {
    if (!currentJobId) return;

    // Initial poll immediately
    const pollJob = async () => {
      try {
        const status = await getJobStatus(currentJobId);

        // Update progress from job - now uses real-time turn_results from DB
        const turnResults = status.turn_results || [];
        setProgress({
          current: status.current_step,
          total: status.total_steps,
          currentAnalysis: status.current_analysis || undefined,
          turnResults: turnResults,
          turnCurrent: turnResults.length,
          turnTotal: parsedTrace?.turns?.length || turnResults.length + 5,
        });

        if (status.status === 'completed') {
          setResults(status.result);
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          setCurrentTraceId(status.trace_id);
          loadSavedTraces();
          loadActiveJobs();
        } else if (status.status === 'failed') {
          setError(`Job failed: ${status.error_message || 'Unknown error'}. Retry count: ${status.retry_count}`);
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          loadActiveJobs();
        }
      } catch (err) {
        console.error('Failed to poll job:', err);
      }
    };

    pollJob(); // Initial poll
    const pollInterval = setInterval(pollJob, 1000);

    return () => clearInterval(pollInterval);
  }, [currentJobId]);

  const loadActiveJobs = async () => {
    try {
      const jobs = await listJobs(10);
      setActiveJobs(jobs.filter(j => j.status === 'running' || j.status === 'pending'));
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  };

  const loadSavedTraces = async () => {
    try {
      setLoadingTraces(true);
      const traces = await listAgentTraces();
      setSavedTraces(traces);
    } catch (err) {
      console.error('Failed to load traces:', err);
    } finally {
      setLoadingTraces(false);
    }
  };

  const loadExample = (key: string, isMultiAgent: boolean = false) => {
    setResults(null);
    setError(null);
    setCurrentTraceId(null);

    if (isMultiAgent) {
      const example = MULTI_AGENT_EXAMPLES[key];
      if (!example) return;
      const jsonStr = JSON.stringify(example.session, null, 2);
      setTraceInput(jsonStr);
      // Multi-agent format is valid - set it as parsed
      setParsedTrace(example.session as unknown as AgentTraceInput);
    } else if (key.startsWith('dataset_')) {
      const datasetKey = key.replace('dataset_', '');
      const example = DATASET_EXAMPLES[datasetKey];
      if (!example) return;
      const jsonStr = JSON.stringify(example.dataset, null, 2);
      setTraceInput(jsonStr);
      setParsedDataset(example.dataset);
      setParsedTrace(null);
      setSelectedConversationIndex(0);
    } else {
      const example = EXAMPLE_TRACES[key];
      if (!example) return;
      const jsonStr = JSON.stringify(example.trace, null, 2);
      setTraceInput(jsonStr);
      setParsedTrace(example.trace);
    }
  };

  const handleInputChange = (value: string) => {
    setTraceInput(value);
    setError(null);
    setCurrentTraceId(null);

    if (!value.trim()) {
      setParsedTrace(null);
      setParsedDataset(null);
      return;
    }

    try {
      const parsed = JSON.parse(value);

      // Check if it's a dataset (has conversations array)
      if (parsed.dataset && Array.isArray(parsed.dataset.conversations)) {
        setParsedDataset(parsed.dataset);
        setParsedTrace(null);
        setSelectedConversationIndex(0);
        return;
      }

      // Check if it's a dataset (direct object with conversations)
      if (Array.isArray(parsed.conversations)) {
        setParsedDataset(parsed);
        setParsedTrace(null);
        setSelectedConversationIndex(0);
        return;
      }

      // Validate turn-based format
      if (!Array.isArray(parsed.turns) || parsed.turns.length === 0) {
        throw new Error('Invalid format - needs turns array');
      }

      // Check if it's multi-agent format (has agent_interactions)
      const isMultiAgent = parsed.turns[0]?.agent_interactions !== undefined;

      if (isMultiAgent) {
        // Multi-agent format validation
        for (const turn of parsed.turns) {
          if (!Array.isArray(turn.agent_interactions) || turn.agent_interactions.length === 0) {
            throw new Error('Multi-agent turns need agent_interactions array');
          }
        }
      } else {
        // Single-agent format validation
        for (const turn of parsed.turns) {
          if (!turn.user_message || !turn.agent_response) {
            throw new Error('Each turn needs user_message and agent_response');
          }
        }
      }

      setParsedTrace(parsed as AgentTraceInput);
      setParsedDataset(null);
    } catch {
      setParsedTrace(null);
      setParsedDataset(null);
    }
  };

  const loadSavedTrace = async (id: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await getAgentTrace(id);
      setTraceInput(JSON.stringify(data.trace, null, 2));
      setParsedTrace(data.trace);
      setResults(data.results);
      setCurrentTraceId(id);
    } catch (err) {
      setError('Failed to load trace');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    // if (!confirm('Delete this trace?')) return;

    try {
      await deleteAgentTrace(id);
      if (currentTraceId === id) {
        setCurrentTraceId(null);
        setResults(null);
      }
      await loadSavedTraces();
    } catch (err) {
      setError('Failed to delete trace');
    }
  };

  const runAnalysis = async () => {
    if (!parsedTrace && !parsedDataset) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setProgress(null);
    setCurrentTraceId(null);
    setCurrentJobId(null);

    let analysisTypes: string[];
    if (selectedAnalysis === 'full_all') {
      analysisTypes = ['conversation', 'trajectory', 'tools', 'self_correction'];
    } else if (selectedAnalysis === 'full_agent') {
      analysisTypes = ['trajectory', 'tools', 'self_correction'];
    } else {
      analysisTypes = [selectedAnalysis];
    }

    // Background mode - start job and poll (trace saved first!)
    if (runInBackground || parsedDataset) {
      try {
        // If dataset, use special endpoint or logic
        let job_id, trace_id;

        if (parsedDataset) {
          // Manually call API for dataset
          const response = await fetch('http://localhost:8000/api/agent/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              dataset: parsedDataset,
              analysis_types: analysisTypes
            })
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || 'Failed to start job');
          job_id = data.job_id;
          trace_id = 0; // Dataset ID not returned as trace_id usually
        } else if (parsedTrace) {
          const res = await startBackgroundJob(parsedTrace, analysisTypes);
          job_id = res.job_id;
          trace_id = res.trace_id;
        }

        setCurrentJobId(job_id);
        if (trace_id) setCurrentTraceId(trace_id);

        setProgress({
          current: 0,
          total: analysisTypes.length * (parsedDataset ? parsedDataset.conversations.length : 1),
          turnTotal: parsedDataset ? 0 : (parsedTrace?.turns.length || 0),
          turnCurrent: 0,
          turnResults: [],
          currentAnalysis: 'Starting...',
        });
        loadActiveJobs();
        loadSavedTraces(); // Refresh to show the new trace
      } catch (err: any) {
        setError('Failed to start background job: ' + err.message);
        setLoading(false);
      }
      return;
    }

    // Streaming mode - run inline
    if (!parsedTrace) return;

    // Detect if multi-agent format: turns have agent_interactions (not just agent_steps)
    const hasAgentInteractions = parsedTrace.turns?.some((t: any) => t.agent_interactions?.length > 0);
    const requestBody: AgentAnalysisRequest = hasAgentInteractions
      ? { session: parsedTrace as any, analysis_types: analysisTypes as any }
      : { trace: parsedTrace, analysis_types: analysisTypes as any };

    analyzeAgentTraceStream(
      requestBody,
      (event: AgentAnalysisStreamEvent) => {
        if (event.type === 'start') {
          setProgress({
            current: 0,
            total: event.total,
            turnResults: [],
            turnTotal: parsedTrace.turns.length,
            turnCurrent: 0,
          });
        } else if (event.type === 'progress') {
          setProgress(prev => ({
            ...prev,
            current: event.current,
            total: event.total,
            currentAnalysis: event.analysis,
            turnTotal: event.turn_total,
            turnCurrent: 0,
          }));
        } else if (event.type === 'turn_result') {
          // Real-time turn result
          setProgress(prev => prev ? ({
            ...prev,
            turnCurrent: event.turn_index + 1,
            turnTotal: event.turn_total,
            turnResults: [...(prev.turnResults || []), event.result],
          }) : null);
        } else if (event.type === 'complete') {
          setResults(event.results);
          setLoading(false);
          setProgress(null);
          // Auto-save the analysis
          saveAgentTrace(parsedTrace, event.results)
            .then(saved => {
              setCurrentTraceId(saved.id);
              loadSavedTraces();
            })
            .catch(() => { }); // Silently fail auto-save
        } else if (event.type === 'error') {
          setError(event.message);
          setLoading(false);
          setProgress(null);
        }
      },
      (err) => {
        setError(err.message);
        setLoading(false);
        setProgress(null);
      }
    );
  };

  const getScoreColor = (score: number | null) => {
    if (score === null) return 'var(--text-secondary)';
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.pageLayout}>
          {/* Sidebar - Saved Traces */}
          <aside className={styles.sidebar}>
            {/* Active Jobs Section */}
            {activeJobs.length > 0 && (
              <div className={styles.activeJobsSection}>
                <div className={styles.sidebarHeader}>
                  <h3>🔄 Running Jobs</h3>
                </div>
                <div className={styles.jobsList}>
                  {activeJobs.map((job) => (
                    <div
                      key={job.id}
                      className={`${styles.jobItem} ${currentJobId === job.id ? styles.active : ''}`}
                      onClick={() => resumeJob(job.id)}
                    >
                      <div className={styles.jobItemHeader}>
                        <span className={styles.jobIcon}>
                          {job.status === 'running' ? '⏳' : '⏸️'}
                        </span>
                        <span className={styles.jobName}>Job #{job.id}</span>
                      </div>
                      <div className={styles.jobProgress}>
                        <div className={styles.jobProgressBar}>
                          <div
                            className={styles.jobProgressFill}
                            style={{ width: `${(job.current_step / job.total_steps) * 100}%` }}
                          />
                        </div>
                        <span className={styles.jobStep}>
                          {job.current_analysis || `${job.current_step}/${job.total_steps}`}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className={styles.sidebarHeader}>
              <h3>📁 Saved Analyses</h3>
              <button
                className={styles.refreshBtn}
                onClick={loadSavedTraces}
                disabled={loadingTraces}
              >
                🔄
              </button>
            </div>

            {loadingTraces ? (
              <div className={styles.sidebarLoading}>Loading...</div>
            ) : savedTraces.length === 0 ? (
              <div className={styles.sidebarEmpty}>
                No saved analyses yet.<br />
                Run an analysis and save it!
              </div>
            ) : (
              <div className={styles.savedList}>
                {savedTraces.map((trace) => (
                  <div
                    key={trace.id}
                    className={`${styles.savedItem} ${currentTraceId === trace.id ? styles.active : ''}`}
                    onClick={() => loadSavedTrace(trace.id)}
                  >
                    <div className={styles.savedItemHeader}>
                      <span className={styles.savedItemName}>
                        {trace.name || trace.task.slice(0, 40)}
                      </span>
                      <button
                        className={styles.deleteBtn}
                        onClick={(e) => handleDelete(trace.id, e)}
                        title="Delete"
                      >
                        ×
                      </button>
                    </div>
                    <div className={styles.savedItemMeta}>
                      <span>{trace.step_count} steps</span>
                      {trace.overall_score !== null && (
                        <span style={{ color: getScoreColor(trace.overall_score) }}>
                          {(trace.overall_score * 100).toFixed(0)}%
                        </span>
                      )}
                      <span className={styles.savedItemDate}>
                        {new Date(trace.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>

          {/* Main Content */}
          <div className={styles.container}>
            <div className={styles.header}>
              <h1 className={styles.title}>
                <span className={styles.titleIcon}>🤖</span>
                Agent Trace Analysis
              </h1>
              <p className={styles.subtitle}>
                Analyze agent execution traces for trajectory quality, tool usage, self-correction, and intent drift.
              </p>
            </div>

            <div className={styles.layout}>
              {/* Left Panel - Input */}
              <div className={styles.inputPanel}>
                <div className={styles.panelHeader}>
                  <h2>Agent Trace</h2>
                  <div className={styles.exampleButtons}>
                    <span className={styles.exampleLabel}>Single Agent:</span>
                    {Object.entries(EXAMPLE_TRACES).map(([key, example]) => (
                      <button
                        key={key}
                        onClick={() => loadExample(key, false)}
                        className={styles.exampleBtn}
                        title={getTraceTask(example.trace)}
                      >
                        {example.icon} {example.label}
                      </button>
                    ))}
                  </div>

                  <div className={styles.exampleButtons}>
                    <span className={styles.exampleLabel}>Multi Agent:</span>
                    {Object.entries(MULTI_AGENT_EXAMPLES).map(([key, example]) => (
                      <button
                        key={key}
                        onClick={() => loadExample(key, true)}
                        className={`${styles.exampleBtn} ${styles.multiAgentBtn}`}
                        title={example.session.initial_task || 'Multi-agent example'}
                      >
                        {example.icon} {example.label.replace('🤝 Multi: ', '')}
                      </button>
                    ))}
                  </div>
                  <div className={styles.exampleButtons}>
                    <span className={styles.exampleLabel}>Datasets:</span>
                    {Object.entries(DATASET_EXAMPLES).map(([key, example]) => (
                      <button
                        key={key}
                        onClick={() => loadExample(`dataset_${key}`, false)}
                        className={`${styles.exampleBtn} ${styles.datasetBtn}`}
                        title={example.dataset.task || 'Dataset example'}
                        style={{ borderColor: 'var(--accent-purple)', color: 'var(--accent-purple)' }}
                      >
                        {example.icon} {example.label}
                      </button>
                    ))}
                  </div>
                </div>


                <textarea
                  className={styles.traceInput}
                  value={traceInput}
                  onChange={(e) => handleInputChange(e.target.value)}
                  placeholder={`Paste your agent trace JSON here...

{
  "agents": [
    {
      "id": "my_agent",
      "name": "MyAgent",
      "role": "assistant",
      "tools_available": [
        { "name": "web_search", "parameters_schema": { "query": "string" } },
        { "name": "read_file", "parameters_schema": { "path": "string" } }
      ]
    }
  ],
  "turns": [
    {
      "user_message": "What the user asked",
      "agent_interactions": [
        {
          "agent_id": "my_agent",
          "agent_steps": [
            { "thought": "I should search for this..." },
            { "tool_call": { "tool_name": "web_search", "parameters": { "query": "..." }, "result": "...", "latency_ms": 150 } }
          ],
          "agent_response": "Here's what I found...",
          "latency_ms": 500
        }
      ]
    }
  ],
  "total_cost": 0.001,
  "total_latency_ms": 500
  "total_cost": 0.001
}`}
                />

                {traceInput && !parsedTrace && !parsedDataset && (
                  <div className={styles.parseError}>
                    ⚠️ Invalid JSON format. Please check your input.
                  </div>
                )}

                {parsedTrace && (
                  <div className={styles.parseSuccess}>
                    ✅ Valid: {parsedTrace.turns?.length || 0} turns
                    {(parsedTrace as any).agents?.length > 0 && ` • ${(parsedTrace as any).agents.length} agents`}
                    {(parsedTrace as any).turns?.[0]?.agent_interactions && ' (multi-agent)'}
                  </div>
                )}
                {parsedDataset && (
                  <div className={styles.parseSuccess}>
                    ✅ Valid Dataset: {parsedDataset.conversations.length} conversations
                  </div>
                )}
              </div>

              {/* Right Panel - Analysis */}
              <div className={styles.analysisPanel}>
                <AnalysisSelector
                  selectedAnalysis={selectedAnalysis}
                  onSelect={setSelectedAnalysis}
                  disabled={loading}
                />

                <div className={styles.actionButtons}>
                  <label className={styles.backgroundToggle}>
                    <input
                      type="checkbox"
                      checked={runInBackground}
                      onChange={(e) => setRunInBackground(e.target.checked)}
                      disabled={loading}
                    />
                    <span>Run in background</span>
                  </label>

                  <button
                    className={`btn btn-primary ${styles.runButton}`}
                    onClick={runAnalysis}
                    disabled={(!parsedTrace && !parsedDataset) || loading}
                  >
                    {loading ? (
                      <>
                        <span className={styles.btnSpinner} />
                        {currentJobId ? (
                          `Job #${currentJobId} running...`
                        ) : progress ? (
                          progress.turnTotal
                            ? `Turn ${progress.turnCurrent || 0}/${progress.turnTotal}`
                            : `${progress.currentAnalysis || 'Analyzing'} ${progress.current}/${progress.total}`
                        ) : (
                          'Starting...'
                        )}
                      </>
                    ) : (
                      <>{runInBackground ? '🔄 Start Background Job' : '⚡ Run Analysis'}</>
                    )}
                  </button>

                  {/* {results && currentTraceId && (
                    <span className={styles.savedIndicator}>
                      ✅ Saved
                    </span>
                  )} */}

                  {currentJobId && (
                    <span className={styles.jobIndicator}>
                      🔄 Job #{currentJobId} (can navigate away)
                    </span>
                  )}
                </div>

                {error && (
                  <div className={styles.error}>
                    <span>❌</span>
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Streaming Analysis Table - visible when we have a trace and are loading */}
            {loading && parsedTrace && (
              <StreamingAnalysisTable
                turns={parsedTrace.turns || []}
                turnResults={progress?.turnResults || []}
                currentTurn={progress?.turnCurrent ? progress.turnCurrent - 1 : -1}
                currentAnalysis={progress?.currentAnalysis || ''}
                isAnalyzing={loading}
              />
            )}

            {/* Dataset View */}
            {parsedDataset && (
              <div className="mt-4">
                {/* Summary Table View - Always visible */}
                <div style={{ height: '600px' }}>
                  <DatasetSummaryTable
                    dataset={parsedDataset}
                    analysisResults={(results as any)?.conversation_results}
                    onSelectConversation={setSelectedConversationIndex}
                  />
                </div>

                {/* Detail Modal */}
                <Modal
                  isOpen={selectedConversationIndex !== -1}
                  onClose={() => setSelectedConversationIndex(-1)}
                  title={`Conversation ${selectedConversationIndex !== -1 ? selectedConversationIndex + 1 : ''}`}
                >
                  {selectedConversationIndex !== -1 && (
                    <>
                      <div className="mb-6 flex items-center gap-4 border-b border-gray-800 pb-4">
                        {parsedDataset.conversations[selectedConversationIndex].initial_task && (
                          <span className="text-sm text-gray-500 truncate max-w-xl">
                            • {parsedDataset.conversations[selectedConversationIndex].initial_task}
                          </span>
                        )}
                      </div>
                      <StreamingAnalysisTable
                        turns={parsedDataset.conversations[selectedConversationIndex].turns || []}
                        turnResults={
                          progress?.turnResults?.filter((r: any) => r.conversation_index === selectedConversationIndex) || []
                        }
                        currentTurn={-1}
                        currentAnalysis={progress?.currentAnalysis || ''}
                        isAnalyzing={loading}
                      />
                    </>
                  )}
                </Modal>
              </div>
            )}

            {/* Results (Single Trace) */}
            {parsedTrace && (results || loading) && (
              <div className={styles.resultsSection}>
                <AgentTraceViewer
                  trace={parsedTrace}
                  results={results || undefined}
                  loading={loading}
                  liveProgress={progress ? {
                    currentAnalysis: progress.currentAnalysis,
                    turnCurrent: progress.turnCurrent,
                    turnTotal: progress.turnTotal,
                    turnResults: progress.turnResults,
                  } : undefined}
                />
              </div>
            )}
          </div>
        </div>
      </main >
    </>
  );
}
