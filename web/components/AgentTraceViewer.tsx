'use client';

import { useState } from 'react';
import styles from './AgentTraceViewer.module.css';
import {
  AgentTraceInput,
  AgentStepInput,
  AgentTurn,
  AgentAnalysisResults,
  TrajectoryResult,
  ToolsResult,
  SelfCorrectionResult,
  IntentDriftResult,
  ConversationAnalysisResult,
  PerAgentScore,
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
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]));

  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const getDriftColor = (drift: number) => {
    if (drift < 0.35) return 'var(--accent-green)';
    if (drift < 0.6) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  // Get task from initial_task or first turn
  const taskDescription = trace.initial_task || trace.turns?.[0]?.user_message || 'Agent Trace';

  return (
    <div className={styles.container}>
      {/* Task Header */}
      <div className={styles.taskHeader}>
        <div className={styles.taskInfo}>
          <span className={styles.taskIcon}>🎯</span>
          <div>
            <h3 className={styles.taskTitle}>Task</h3>
            <p className={styles.taskContent}>{taskDescription}</p>
          </div>
        </div>
        <div className={styles.taskMeta}>
          <span className={styles.metaBadge}>
            {trace.turns?.length || 0} turns
          </span>
          {trace.agents && trace.agents.length > 0 && (
            <span className={styles.metaBadge}>
              🤖 {trace.agents.length} agent{trace.agents.length > 1 ? 's' : ''}
            </span>
          )}
          {trace.total_latency_ms !== undefined && (
            <span className={styles.metaBadge}>
              ⏱️ {trace.total_latency_ms}ms
            </span>
          )}
          {trace.total_cost !== undefined && (
            <span className={styles.metaBadge}>
              💰 ${trace.total_cost.toFixed(4)}
            </span>
          )}
        </div>
      </div>

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

      {/* Conversation Analysis Section (full width) */}
      {results?.conversation && (
        <ConversationCard result={results.conversation} />
      )}

      {/* Agent Analysis Results Grid */}
      {results && (results.trajectory || results.tools || results.self_correction || results.intent_drift) && (
        <div className={styles.resultsGrid}>
          {results.trajectory && (
            <TrajectoryCard result={results.trajectory} />
          )}
          {results.tools && (
            <ToolsCard result={results.tools} />
          )}
          {results.self_correction && (
            <SelfCorrectionCard result={results.self_correction} />
          )}
          {results.intent_drift && (
            <IntentDriftCard result={results.intent_drift} />
          )}
        </div>
      )}

      {/* Overall Score */}
      {results && (
        <div className={styles.overallScore}>
          <span className={styles.overallLabel}>Overall Score</span>
          <div className={styles.scoreBar}>
            <div 
              className={styles.scoreFill}
              style={{ 
                width: `${results.overall_score * 100}%`,
                background: results.overall_score >= 0.7 ? 'var(--accent-green)' : 
                           results.overall_score >= 0.4 ? 'var(--accent-orange)' : 'var(--accent-red)'
              }}
            />
          </div>
          <span className={styles.scoreValue}>{(results.overall_score * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Per-Agent Scores */}
      {results?.per_agent_scores && Object.keys(results.per_agent_scores).length > 0 && (
        <div className={styles.agentScoresSection}>
          <h4 className={styles.sectionTitle}>🤖 Per-Agent Scores</h4>
          <div className={styles.agentScoresGrid}>
            {Object.entries(results.per_agent_scores).map(([agentId, scores]) => {
              const agentDef = trace.agents?.find(a => a.id === agentId);
              const agentName = agentDef?.name || agentId;
              const scoreColor = scores.overall >= 0.7 ? 'var(--accent-green)' : 
                                scores.overall >= 0.4 ? 'var(--accent-orange)' : 'var(--accent-red)';
              return (
                <div key={agentId} className={styles.agentScoreCard}>
                  <div className={styles.agentScoreHeader}>
                    <span className={styles.agentName}>{agentName}</span>
                    <span className={styles.agentRole}>{agentDef?.role || 'agent'}</span>
                  </div>
                  <div className={styles.agentMainScore}>
                    <div className={styles.scoreBar}>
                      <div 
                        className={styles.scoreFill}
                        style={{ width: `${scores.overall * 100}%`, background: scoreColor }}
                      />
                    </div>
                    <span className={styles.scoreValue} style={{ color: scoreColor }}>
                      {(scores.overall * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className={styles.agentMetrics}>
                    {scores.tool_use != null && (
                      <div className={styles.agentMetric}>
                        <span className={styles.metricLabel}>🔧 Tool Use</span>
                        <span className={styles.metricValue}>{(scores.tool_use * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {scores.reasoning != null && (
                      <div className={styles.agentMetric}>
                        <span className={styles.metricLabel}>🧠 Reasoning</span>
                        <span className={styles.metricValue}>{(scores.reasoning * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    {scores.handoff != null && (
                      <div className={styles.agentMetric}>
                        <span className={styles.metricLabel}>🔄 Handoff</span>
                        <span className={styles.metricValue}>{(scores.handoff * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    <div className={styles.agentMetric}>
                      <span className={styles.metricLabel}>💬 Interactions</span>
                      <span className={styles.metricValue}>{scores.interactions_count}</span>
                    </div>
                  </div>
                  {scores.issues.length > 0 && (
                    <div className={styles.agentIssues}>
                      {scores.issues.map((issue, i) => (
                        <div key={i} className={styles.issueItem}>⚠️ {issue}</div>
                      ))}
                    </div>
                  )}
                  {scores.recommendations.length > 0 && (
                    <div className={styles.agentRecommendations}>
                      {scores.recommendations.map((rec, i) => (
                        <div key={i} className={styles.recommendationItem}>💡 {rec}</div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Coordination Score */}
      {results?.coordination_score != null && (
        <div className={styles.coordinationScore}>
          <span className={styles.coordinationLabel}>🤝 Agent Coordination</span>
          <div className={styles.scoreBar}>
            <div 
              className={styles.scoreFill}
              style={{ 
                width: `${results.coordination_score * 100}%`,
                background: results.coordination_score >= 0.7 ? 'var(--accent-green)' : 
                           results.coordination_score >= 0.4 ? 'var(--accent-orange)' : 'var(--accent-red)'
              }}
            />
          </div>
          <span className={styles.scoreValue}>{(results.coordination_score * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Conversation Timeline */}
      <div className={styles.timeline}>
        <h4 className={styles.timelineTitle}>Conversation Timeline</h4>
        {trace.turns?.map((turn, index) => (
          <TurnCard
            key={index}
            turn={turn}
            index={index}
            expanded={expandedSteps.has(index)}
            onToggle={() => toggleStep(index)}
            driftScore={results?.intent_drift?.drift_history?.[index]}
            conversationResult={results?.conversation?.results?.find(r => r.step_index === index)}
          />
        ))}
      </div>

      {loading && (
        <div className={styles.loadingOverlay}>
          <div className={styles.spinner} />
          <span>Analyzing trace...</span>
        </div>
      )}
    </div>
  );
}

// Helper to normalize turn data (handles both simple and multi-agent formats)
function normalizeTurn(turn: any): { 
  userMessage: string; 
  agentSteps: AgentStepInput[]; 
  agentResponse: string;
  agentId?: string;
  interactions?: any[];
} {
  // Multi-agent format: has agent_interactions
  if (turn.agent_interactions && turn.agent_interactions.length > 0) {
    const interaction = turn.agent_interactions[0];
    return {
      userMessage: turn.user_message || '',
      agentSteps: interaction.agent_steps || [],
      agentResponse: interaction.agent_response || turn.final_response || '',
      agentId: interaction.agent_id,
      interactions: turn.agent_interactions,
    };
  }
  
  // Simple format: direct agent_steps and agent_response
  return {
    userMessage: turn.user_message || '',
    agentSteps: turn.agent_steps || [],
    agentResponse: turn.agent_response || '',
  };
}

// Turn-based conversation card
function TurnCard({ 
  turn, 
  index, 
  expanded, 
  onToggle,
  driftScore,
  conversationResult,
}: { 
  turn: AgentTurn; 
  index: number; 
  expanded: boolean;
  onToggle: () => void;
  driftScore?: number;
  conversationResult?: { is_bad: boolean; detection_type: string; confidence: number; reason?: string };
}) {
  const getDriftColor = (drift: number) => {
    if (drift < 0.35) return 'var(--accent-green)';
    if (drift < 0.6) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  const getDetectionBadge = (type: string) => {
    const colors: Record<string, string> = {
      ccm: 'var(--accent-cyan)',
      rdm: 'var(--accent-purple)',
      llm_judge: 'var(--accent-orange)',
      hallucination: 'var(--accent-red)',
    };
    return colors[type] || 'var(--text-secondary)';
  };

  // Normalize turn to handle both formats
  const normalized = normalizeTurn(turn);
  const hasSteps = normalized.agentSteps.length > 0;

  // Get score color based on confidence
  const getScoreColor = (isBad: boolean, confidence: number) => {
    if (isBad) return 'var(--accent-red)';
    if (confidence >= 0.9) return 'var(--accent-green)';
    if (confidence >= 0.7) return 'var(--accent-cyan)';
    return 'var(--accent-orange)';
  };

  return (
    <div className={`${styles.turn} ${expanded ? styles.expanded : ''} ${conversationResult?.is_bad ? styles.turnBad : styles.turnGood}`}>
      <button className={styles.turnHeader} onClick={onToggle}>
        <div className={styles.turnIndex}>
          <span className={styles.turnNumber}>{index + 1}</span>
        </div>
        <div className={styles.turnPreview}>
          <span className={styles.turnUser}>👤 {normalized.userMessage.slice(0, 50)}{normalized.userMessage.length > 50 ? '...' : ''}</span>
        </div>
        {/* Always show score badge when we have results */}
        {conversationResult && (
          <span 
            className={styles.scoreBadge}
            style={{ 
              background: getScoreColor(conversationResult.is_bad, conversationResult.confidence),
              opacity: conversationResult.is_bad ? 1 : 0.8
            }}
          >
            {conversationResult.is_bad ? '❌' : '✓'} {conversationResult.detection_type.toUpperCase()} ({(conversationResult.confidence * 100).toFixed(0)}%)
          </span>
        )}
        {driftScore !== undefined && (
          <div 
            className={styles.driftIndicator}
            style={{ background: getDriftColor(driftScore) }}
            title={`Drift: ${(driftScore * 100).toFixed(0)}%`}
          >
            {(driftScore * 100).toFixed(0)}%
          </div>
        )}
        <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
      </button>

      {expanded && (
        <div className={styles.turnContent}>
          {/* User Message */}
          <div className={styles.messageBlock}>
            <div className={styles.messageHeader}>
              <span className={styles.messageRole}>👤 USER</span>
            </div>
            <p className={styles.messageContent}>{normalized.userMessage}</p>
          </div>

          {/* Agent Steps (if any) */}
          {hasSteps && (
            <div className={styles.agentSteps}>
              <span className={styles.stepsLabel}>🧠 Agent Reasoning ({normalized.agentSteps.length} steps)</span>
              {normalized.agentSteps.map((step: any, stepIndex: number) => (
                <div key={stepIndex} className={styles.agentStep}>
                  {step.thought && (
                    <div className={styles.stepItem}>
                      <span className={styles.stepItemIcon}>💭</span>
                      <span>{step.thought}</span>
                    </div>
                  )}
                  {step.action && (
                    <div className={styles.stepItem}>
                      <span className={styles.stepItemIcon}>⚡</span>
                      <span>{step.action}</span>
                    </div>
                  )}
                  {step.observation && (
                    <div className={styles.stepItem}>
                      <span className={styles.stepItemIcon}>👁️</span>
                      <span className={styles.observation}>{step.observation}</span>
                    </div>
                  )}
                  {step.tool_call && (
                    <div className={styles.toolCallCompact}>
                      <span className={styles.stepItemIcon}>🔧</span>
                      <code>{step.tool_call.tool_name || step.tool_call.name}</code>
                      {step.tool_call.result && <span className={styles.toolSuccess}>✓ {String(step.tool_call.result).slice(0, 100)}</span>}
                      {step.tool_call.error && <span className={styles.toolFailure}>✗ {step.tool_call.error}</span>}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Agent Response */}
          <div className={`${styles.messageBlock} ${styles.agentMessage} ${conversationResult?.is_bad ? styles.badResponse : conversationResult ? styles.goodResponse : ''}`}>
            <div className={styles.messageHeader}>
              <span className={styles.messageRole}>🤖 AGENT{normalized.agentId ? ` (${normalized.agentId})` : ''}</span>
              {conversationResult && (
                <span 
                  className={conversationResult.is_bad ? styles.badBadge : styles.goodBadge}
                  style={{ background: getScoreColor(conversationResult.is_bad, conversationResult.confidence) }}
                >
                  {conversationResult.is_bad ? '❌' : '✓'} {conversationResult.detection_type.toUpperCase()} ({(conversationResult.confidence * 100).toFixed(0)}%)
                </span>
              )}
            </div>
            {normalized.agentResponse ? (
              <p className={styles.messageContent}>{normalized.agentResponse}</p>
            ) : (
              <p className={styles.messageContentEmpty}>No response recorded</p>
            )}
            {conversationResult?.reason && (
              <p className={styles.detectionReason}>💡 {conversationResult.reason}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TrajectoryCard({ result }: { result: TrajectoryResult }) {
  const signalColors: Record<string, string> = {
    optimal: 'var(--accent-green)',
    circular: 'var(--accent-orange)',
    regression: 'var(--accent-red)',
    stall: 'var(--accent-orange)',
    recovery: 'var(--accent-cyan)',
    drift: 'var(--accent-purple)',
  };

  return (
    <div className={styles.resultCard}>
      <div className={styles.cardHeader}>
        <span className={styles.cardIcon}>🔄</span>
        <span className={styles.cardTitle}>Trajectory</span>
      </div>
      <div className={styles.cardContent}>
        <div className={styles.signalBadge} style={{ background: signalColors[result.signal] || 'var(--text-secondary)' }}>
          {result.signal.toUpperCase()}
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Efficiency</span>
          <span className={styles.metricValue}>{(result.efficiency_score * 100).toFixed(0)}%</span>
        </div>
        {result.circular_count > 0 && (
          <div className={styles.warning}>
            ⚠️ {result.circular_count} circular pattern(s)
          </div>
        )}
        {result.regression_count > 0 && (
          <div className={styles.warning}>
            ⚠️ {result.regression_count} regression(s)
          </div>
        )}
        <p className={styles.reason}>{result.reason}</p>
      </div>
    </div>
  );
}

function ToolsCard({ result }: { result: ToolsResult }) {
  const correctCount = result.correct_count ?? result.results.filter(r => r.signal === 'correct').length;
  const accuracy = result.total_calls > 0 ? correctCount / result.total_calls : 1;

  return (
    <div className={styles.resultCard}>
      <div className={styles.cardHeader}>
        <span className={styles.cardIcon}>🔧</span>
        <span className={styles.cardTitle}>Tool Usage</span>
      </div>
      <div className={styles.cardContent}>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Accuracy</span>
          <span className={styles.metricValue}>{(accuracy * 100).toFixed(0)}%</span>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Efficiency</span>
          <span className={styles.metricValue}>{(result.efficiency * 100).toFixed(0)}%</span>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Tool Calls</span>
          <span className={styles.metricValue}>{result.total_calls}</span>
        </div>
        {result.results.filter(r => r.signal !== 'correct').map((r, i) => (
          <div key={i} className={styles.warning}>
            ⚠️ {r.tool_name}: {r.signal.replace('_', ' ')}
          </div>
        ))}
      </div>
    </div>
  );
}

function SelfCorrectionCard({ result }: { result: SelfCorrectionResult }) {
  return (
    <div className={styles.resultCard}>
      <div className={styles.cardHeader}>
        <span className={styles.cardIcon}>🔁</span>
        <span className={styles.cardTitle}>Self-Correction</span>
      </div>
      <div className={styles.cardContent}>
        <div className={styles.checkList}>
          <div className={styles.checkItem}>
            <span className={result.detected_error ? styles.checkYes : styles.checkNo}>
              {result.detected_error ? '✅' : '➖'}
            </span>
            <span>Error Detected</span>
          </div>
          <div className={styles.checkItem}>
            <span className={result.correction_attempt ? styles.checkYes : styles.checkNo}>
              {result.correction_attempt ? '✅' : '➖'}
            </span>
            <span>Correction Attempted</span>
          </div>
          <div className={styles.checkItem}>
            <span className={result.correction_success ? styles.checkYes : styles.checkNo}>
              {result.correction_success ? '✅' : '❌'}
            </span>
            <span>Correction Successful</span>
          </div>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Self-Awareness</span>
          <span className={styles.metricValue}>{(result.self_awareness_score * 100).toFixed(0)}%</span>
        </div>
        {result.loops_before_fix > 0 && (
          <div className={styles.warning}>
            ⚠️ {result.loops_before_fix} loop(s) before fix
          </div>
        )}
        <p className={styles.reason}>{result.reason}</p>
      </div>
    </div>
  );
}

function IntentDriftCard({ result }: { result: IntentDriftResult }) {
  const driftColor = result.drift_score < 0.35 ? 'var(--accent-green)' : 
                     result.drift_score < 0.6 ? 'var(--accent-orange)' : 'var(--accent-red)';

  return (
    <div className={styles.resultCard}>
      <div className={styles.cardHeader}>
        <span className={styles.cardIcon}>🎯</span>
        <span className={styles.cardTitle}>Intent Drift</span>
      </div>
      <div className={styles.cardContent}>
        <div className={styles.driftScore} style={{ color: driftColor }}>
          {(result.drift_score * 100).toFixed(0)}%
        </div>
        <div className={styles.driftBar}>
          {result.drift_history.map((drift, i) => (
            <div
              key={i}
              className={styles.driftBarSegment}
              style={{
                height: `${drift * 100}%`,
                background: drift < 0.35 ? 'var(--accent-green)' : 
                           drift < 0.6 ? 'var(--accent-orange)' : 'var(--accent-red)',
              }}
              title={`Step ${i + 1}: ${(drift * 100).toFixed(0)}%`}
            />
          ))}
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Max Drift Step</span>
          <span className={styles.metricValue}>#{result.step_index + 1}</span>
        </div>
        <div className={styles.checkItem}>
          <span className={result.is_legitimate ? styles.checkYes : styles.checkNo}>
            {result.is_legitimate ? '✅' : '⚠️'}
          </span>
          <span>{result.is_legitimate ? 'Legitimate deviation' : 'Unintended drift'}</span>
        </div>
        <p className={styles.reason}>{result.reason}</p>
      </div>
    </div>
  );
}

function ConversationCard({ result }: { result: ConversationAnalysisResult }) {
  const total = result.total_responses;
  const bad = result.bad_responses;
  const good = result.good_responses;
  
  // Calculate avg confidence from results
  const avgConfidence = result.results.length > 0
    ? result.results.reduce((sum, r) => sum + r.confidence, 0) / result.results.length
    : 0;

  // Get max detection count for bar scaling
  const maxDetections = Math.max(
    result.ccm_detections,
    result.rdm_detections,
    result.llm_judge_detections,
    result.hallucination_detections,
    1
  );

  return (
    <div className={styles.conversationSection}>
      {/* Stats Cards */}
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📊</div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{total}</span>
            <span className={styles.statLabel}>TOTAL RESPONSES</span>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>✅</div>
          <div className={styles.statContent}>
            <span className={styles.statValue} style={{ color: 'var(--accent-green)' }}>{good}</span>
            <span className={styles.statLabel}>GOOD RESPONSES</span>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>❌</div>
          <div className={styles.statContent}>
            <span className={styles.statValue} style={{ color: bad > 0 ? 'var(--accent-red)' : 'var(--text-secondary)' }}>{bad}</span>
            <span className={styles.statLabel}>BAD RESPONSES</span>
          </div>
        </div>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>🎯</div>
          <div className={styles.statContent}>
            <span className={styles.statValue}>{(avgConfidence * 100).toFixed(0)}%</span>
            <span className={styles.statLabel}>AVG CONFIDENCE</span>
          </div>
        </div>
      </div>

      {/* Detection Methods */}
      <div className={styles.detectionMethods}>
        <h4 className={styles.detectionTitle}>DETECTION METHODS</h4>
        
        <div className={styles.detectionRow}>
          <span className={styles.detectionBadgeLarge} style={{ background: 'var(--accent-cyan)' }}>CCM</span>
          <div className={styles.detectionBarContainer}>
            <div 
              className={styles.detectionBar}
              style={{ 
                width: `${(result.ccm_detections / maxDetections) * 100}%`,
                background: 'var(--accent-cyan)'
              }}
            />
          </div>
          <span className={styles.detectionCount}>{result.ccm_detections}</span>
        </div>

        <div className={styles.detectionRow}>
          <span className={styles.detectionBadgeLarge} style={{ background: 'var(--accent-purple)' }}>RDM</span>
          <div className={styles.detectionBarContainer}>
            <div 
              className={styles.detectionBar}
              style={{ 
                width: `${(result.rdm_detections / maxDetections) * 100}%`,
                background: 'var(--accent-purple)'
              }}
            />
          </div>
          <span className={styles.detectionCount}>{result.rdm_detections}</span>
        </div>

        <div className={styles.detectionRow}>
          <span className={styles.detectionBadgeLarge} style={{ background: 'var(--accent-orange)' }}>LLM JUDGE</span>
          <div className={styles.detectionBarContainer}>
            <div 
              className={styles.detectionBar}
              style={{ 
                width: `${(result.llm_judge_detections / maxDetections) * 100}%`,
                background: 'var(--accent-orange)'
              }}
            />
          </div>
          <span className={styles.detectionCount}>{result.llm_judge_detections}</span>
        </div>

        <div className={styles.detectionRow}>
          <span className={styles.detectionBadgeLarge} style={{ background: 'var(--accent-red)' }}>HALLUCINATION</span>
          <div className={styles.detectionBarContainer}>
            <div 
              className={styles.detectionBar}
              style={{ 
                width: `${(result.hallucination_detections / maxDetections) * 100}%`,
                background: 'var(--accent-red)'
              }}
            />
          </div>
          <span className={styles.detectionCount}>{result.hallucination_detections}</span>
        </div>
      </div>
    </div>
  );
}

