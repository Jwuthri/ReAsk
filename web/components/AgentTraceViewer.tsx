'use client';

import { useState } from 'react';
import styles from './AgentTraceViewer.module.css';
import AgentTree, { TreeSelection } from './AgentTree';
import DetailPanel from './DetailPanel';
import StreamingAnalysisTable from './StreamingAnalysisTable';
import {
  AgentTraceInput,
  AgentAnalysisResults,
} from '@/lib/api';

interface LiveProgress {
  currentAnalysis?: string;
  turnCurrent?: number;
  turnTotal?: number;
  turnResults?: Array<{
    step_index: number;
    is_bad: boolean;
    detection_type: string;
    confidence: number;
    reason: string;
  }>;
}

interface AgentTraceViewerProps {
  trace: AgentTraceInput;
  results?: AgentAnalysisResults;
  loading?: boolean;
  liveProgress?: LiveProgress;
}

export default function AgentTraceViewer({ trace, results, loading, liveProgress }: AgentTraceViewerProps) {
  const [selection, setSelection] = useState<TreeSelection>({ type: 'global' });

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  const getScoreGrade = (score: number) => {
    if (score >= 0.9) return 'A+';
    if (score >= 0.8) return 'A';
    if (score >= 0.7) return 'B';
    if (score >= 0.6) return 'C';
    if (score >= 0.5) return 'D';
    return 'F';
  };

  // Task description
  const taskDescription = trace.initial_task || trace.turns?.[0]?.user_message || 'Agent Trace';

  // Calculate summary stats
  const agentCount = trace.agents?.length || 1;
  const turnCount = trace.turns?.length || 0;
  const goodResponses = results?.conversation?.good_responses || 0;
  const badResponses = results?.conversation?.bad_responses || 0;

  return (
    <div className={styles.container}>
      {/* Overview Hero Section */}
      {results && (
        <div className={styles.overviewHero}>
          {/* Left: Score Ring */}
          <div className={styles.scoreRing}>
            <svg viewBox="0 0 120 120" className={styles.scoreCircle}>
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="var(--bg-secondary)"
                strokeWidth="8"
              />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke={getScoreColor(results.overall_score)}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${results.overall_score * 327} 327`}
                transform="rotate(-90 60 60)"
                className={styles.scoreProgress}
              />
            </svg>
            <div className={styles.scoreCenter}>
              <span className={styles.scoreGrade} style={{ color: getScoreColor(results.overall_score) }}>
                {getScoreGrade(results.overall_score)}
              </span>
              <span className={styles.scorePercent}>
                {(results.overall_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Middle: Task & Meta */}
          <div className={styles.overviewMeta}>
            <h2 className={styles.overviewTitle}>Analysis Complete</h2>
            <p className={styles.overviewTask}>
              {taskDescription.slice(0, 100)}{taskDescription.length > 100 ? '...' : ''}
            </p>
            <div className={styles.overviewTags}>
              <span className={styles.overviewTag}>
                <span className={styles.tagIcon}>👥</span>
                {agentCount} Agent{agentCount !== 1 ? 's' : ''}
              </span>
              <span className={styles.overviewTag}>
                <span className={styles.tagIcon}>💬</span>
                {turnCount} Turn{turnCount !== 1 ? 's' : ''}
              </span>
              {results.tools && (
                <span className={styles.overviewTag}>
                  <span className={styles.tagIcon}>🔧</span>
                  {results.tools.total_calls} Tool Call{results.tools.total_calls !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Right: Metric Cards */}
          <div className={styles.overviewMetrics}>
            {/* Responses */}
            <div className={styles.metricBox}>
              <div className={styles.metricBoxHeader}>
                <span className={styles.metricBoxIcon}>💬</span>
                <span className={styles.metricBoxLabel}>Responses</span>
              </div>
              <div className={styles.metricBoxValue}>
                <span className={styles.metricGood}>{goodResponses}</span>
                <span className={styles.metricDivider}>/</span>
                <span className={badResponses > 0 ? styles.metricBad : styles.metricMuted}>
                  {badResponses}
                </span>
              </div>
              <span className={styles.metricBoxSublabel}>Good / Issues</span>
            </div>

            {/* Efficiency */}
            {results.trajectory && (
              <div className={styles.metricBox}>
                <div className={styles.metricBoxHeader}>
                  <span className={styles.metricBoxIcon}>⚡</span>
                  <span className={styles.metricBoxLabel}>Efficiency</span>
                </div>
                <div className={styles.metricBoxValue}>
                  <span style={{ color: getScoreColor(results.trajectory.efficiency_score) }}>
                    {(results.trajectory.efficiency_score * 100).toFixed(0)}%
                  </span>
                </div>
                <span className={styles.metricBoxSublabel}>{results.trajectory.signal}</span>
              </div>
            )}

            {/* Coordination (if multi-agent) */}
            {results.coordination_score != null && (
              <div className={styles.metricBox}>
                <div className={styles.metricBoxHeader}>
                  <span className={styles.metricBoxIcon}>🤝</span>
                  <span className={styles.metricBoxLabel}>Coordination</span>
                </div>
                <div className={styles.metricBoxValue}>
                  <span style={{ color: getScoreColor(results.coordination_score) }}>
                    {(results.coordination_score * 100).toFixed(0)}%
                  </span>
                </div>
                <span className={styles.metricBoxSublabel}>Multi-agent</span>
              </div>
            )}

            {/* Tool Usage */}
            {results.tools && (
              <div className={styles.metricBox}>
                <div className={styles.metricBoxHeader}>
                  <span className={styles.metricBoxIcon}>🔧</span>
                  <span className={styles.metricBoxLabel}>Tools</span>
                </div>
                <div className={styles.metricBoxValue}>
                  <span style={{ color: getScoreColor(results.tools.efficiency) }}>
                    {(results.tools.efficiency * 100).toFixed(0)}%
                  </span>
                </div>
                <span className={styles.metricBoxSublabel}>Accuracy</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Split View - Navigation & Details */}
      <div className={styles.splitView}>
        {/* Left Panel - Navigation */}
        <div className={styles.navPanel}>
          <div className={styles.navHeader}>
            <span className={styles.navIcon}>🗂️</span>
            <span className={styles.navTitle}>Navigation</span>
          </div>
          <AgentTree
            trace={trace}
            results={results}
            selection={selection}
            onSelect={setSelection}
          />
        </div>

        {/* Right Panel - Detail View */}
        <div className={styles.detailPanel}>
          <DetailPanel
            trace={trace}
            results={results}
            selection={selection}
          />
        </div>
      </div>

      {/* Conversation Analysis Table */}
      {trace.turns && trace.turns.length > 0 && (
        <StreamingAnalysisTable
          turns={trace.turns}
          turnResults={results?.conversation?.results || liveProgress?.turnResults || []}
          currentTurn={loading && liveProgress?.turnCurrent ? liveProgress.turnCurrent - 1 : -1}
          currentAnalysis={loading ? liveProgress?.currentAnalysis || 'Analyzing...' : 'Complete'}
          isAnalyzing={loading || false}
        />
      )}

      {loading && (
        <div className={styles.loadingOverlay}>
          <div className={styles.spinner} />
          <span>Analyzing trace...</span>
        </div>
      )}
    </div>
  );
}
