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


  // Task description
  const taskDescription = trace.initial_task || trace.turns?.[0]?.user_message || 'Agent Trace';

  return (
    <div className={styles.container}>
      {/* Global Score Hero */}
      {results && (
        <div className={styles.scoreHero}>
          <div className={styles.heroLeft}>
            <div className={styles.heroScoreDisplay}>
              <span className={styles.heroScoreValue} style={{ color: getScoreColor(results.overall_score) }}>
                {(results.overall_score * 100).toFixed(0)}
              </span>
              <div className={styles.heroScoreMeta}>
                <span className={styles.heroScorePercent}>%</span>
                <span className={styles.heroScoreLabel}>Score</span>
              </div>
            </div>
            <div className={styles.heroInfo}>
              <h2 className={styles.heroTitle}>Overall Score</h2>
              <p className={styles.heroTask}>{taskDescription.slice(0, 80)}{taskDescription.length > 80 ? '...' : ''}</p>
            </div>
          </div>

          <div className={styles.heroMetrics}>
            {results.conversation && (
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricValue} style={{
                  color: results.conversation.bad_responses > 0 ? 'var(--accent-red)' : 'var(--accent-green)'
                }}>
                  {results.conversation.good_responses}/{results.conversation.total_responses}
                </span>
                <span className={styles.heroMetricLabel}>Good Responses</span>
              </div>
            )}
            {results.trajectory && (
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricValue} style={{ color: getScoreColor(results.trajectory.efficiency_score) }}>
                  {(results.trajectory.efficiency_score * 100).toFixed(0)}%
                </span>
                <span className={styles.heroMetricLabel}>Efficiency</span>
              </div>
            )}
            {results.tools && (
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricValue} style={{ color: getScoreColor(results.tools.efficiency) }}>
                  {(results.tools.efficiency * 100).toFixed(0)}%
                </span>
                <span className={styles.heroMetricLabel}>Tool Usage</span>
              </div>
            )}
            {results.coordination_score != null && (
              <div className={styles.heroMetric}>
                <span className={styles.heroMetricValue} style={{ color: getScoreColor(results.coordination_score) }}>
                  {(results.coordination_score * 100).toFixed(0)}%
                </span>
                <span className={styles.heroMetricLabel}>Coordination</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Live Progress Panel */}
      {loading && liveProgress && liveProgress.turnResults && liveProgress.turnResults.length > 0 && (
        <div className={styles.liveProgress}>
          <div className={styles.liveHeader}>
            <span className={styles.liveIcon}>⚡</span>
            <span className={styles.liveTitle}>
              Analyzing Turn {liveProgress.turnCurrent}/{liveProgress.turnTotal}
            </span>
            <div className={styles.liveBar}>
              <div
                className={styles.liveBarFill}
                style={{ width: `${((liveProgress.turnCurrent || 0) / (liveProgress.turnTotal || 1)) * 100}%` }}
              />
            </div>
          </div>
          <div className={styles.liveTurns}>
            {liveProgress.turnResults.map((result, idx) => (
              <div
                key={idx}
                className={`${styles.liveTurn} ${result.is_bad ? styles.liveBad : styles.liveGood}`}
              >
                <span className={styles.liveTurnIcon}>
                  {result.is_bad ? '❌' : '✅'}
                </span>
                <span className={styles.liveTurnIndex}>Turn {result.step_index + 1}</span>
                <span className={styles.liveTurnType}>
                  {result.detection_type.toUpperCase()}
                </span>
                <span className={styles.liveTurnConfidence}>
                  {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Split View */}
      <div className={styles.splitView}>
        {/* Left Panel - Agent Tree */}
        <div className={styles.treePanel}>
          <div className={styles.treePanelHeader}>
            <span className={styles.treePanelTitle}>🗂️ Navigation</span>
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

      {/* Conversation Analysis Table (Bottom) */}
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
