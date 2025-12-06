'use client';

import { useState } from 'react';
import styles from './DetailPanel.module.css';
import { TreeSelection } from './AgentTree';
import {
  AgentTraceInput,
  AgentTurn,
  AgentAnalysisResults,
  PerAgentScore,
  AgentDef,
  ConversationStepResult,
  TrajectoryResult,
  ToolsResult,
  SelfCorrectionResult,
  ConversationAnalysisResult,
  AgentInteractionInput,
  AgentStepInput,
} from '@/lib/api';

// Local interface for new reasoning data structure
interface ReasoningThought {
  turn_index: number;
  thought_preview: string;
  is_clear: boolean;
  is_structured: boolean;
  score: number;
}

interface ReasoningData {
  quality_score: number;
  reasoning_depth: number;
  avg_thought_quality: number;
  total_thoughts: number;
  clear_thoughts: number;
  structured_thoughts: number;
  assessment: string;
  thoughts: ReasoningThought[];
}

interface DetailPanelProps {
  trace: AgentTraceInput;
  results?: AgentAnalysisResults;
  selection: TreeSelection;
}

export default function DetailPanel({ trace, results, selection }: DetailPanelProps) {
  if (selection.type === 'global') {
    return <GlobalView trace={trace} results={results} />;
  }

  if (selection.type === 'agent') {
    return <AgentView trace={trace} results={results} agentId={selection.agentId} />;
  }

  if (selection.type === 'turn') {
    return <TurnView trace={trace} results={results} turnIndex={selection.turnIndex} agentId={selection.agentId} />;
  }

  return null;
}

// ============================================
// GLOBAL VIEW
// ============================================
function GlobalView({ trace, results }: { trace: AgentTraceInput; results?: AgentAnalysisResults }) {
  if (!results) {
    return (
      <div className={styles.emptyState}>
        <span className={styles.emptyIcon}>📊</span>
        <h3>No Analysis Results</h3>
        <p>Run an analysis to see detailed metrics</p>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h2 className={styles.title}>📊 Overall Analysis</h2>
        <p className={styles.subtitle}>
          {trace.initial_task || trace.turns?.[0]?.user_message || 'Agent Session'}
        </p>
      </div>

      {/* Main Score */}
      <div className={styles.scoreHero}>
        <div className={styles.scoreDisplay}>
          <span className={styles.scoreValue}>{(results.overall_score * 100).toFixed(0)}</span>
          <div className={styles.scoreMeta}>
            <span className={styles.scorePercent}>%</span>
            <span className={styles.scoreLabel}>Overall</span>
          </div>
        </div>
        <div className={styles.scoreBar}>
          <div
            className={styles.scoreBarFill}
            style={{
              width: `${results.overall_score * 100}%`,
              background: getScoreColor(results.overall_score)
            }}
          />
          <div className={styles.scoreBarMarkers}>
            <span style={{ left: '40%' }} />
            <span style={{ left: '70%' }} />
          </div>
        </div>
        <div className={styles.scoreScale}>
          <span>Poor</span>
          <span>Good</span>
          <span>Excellent</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className={styles.metricsGrid}>
        {results.conversation && (
          <MetricCard
            icon="💬"
            label="Responses"
            value={`${results.conversation.good_responses}/${results.conversation.total_responses}`}
            sublabel="Good/Total"
            color={results.conversation.bad_responses > 0 ? 'var(--accent-red)' : 'var(--accent-green)'}
          />
        )}
        {results.trajectory && (
          <MetricCard
            icon="🔄"
            label="Efficiency"
            value={`${(results.trajectory.efficiency_score * 100).toFixed(0)}%`}
            sublabel={results.trajectory.signal}
            color={getScoreColor(results.trajectory.efficiency_score)}
          />
        )}
        {results.tools && (
          <MetricCard
            icon="🔧"
            label="Tool Usage"
            value={`${(results.tools.efficiency * 100).toFixed(0)}%`}
            sublabel={`${results.tools.total_calls} calls`}
            color={getScoreColor(results.tools.efficiency)}
          />
        )}
        {results.self_correction && (
          <MetricCard
            icon="🔁"
            label="Self-Correction"
            value={`${(results.self_correction.self_awareness_score * 100).toFixed(0)}%`}
            sublabel={results.self_correction.correction_success ? 'Success' : 'No correction'}
            color={getScoreColor(results.self_correction.self_awareness_score)}
          />
        )}
        {results.coordination_score != null && (
          <MetricCard
            icon="🤝"
            label="Coordination"
            value={`${(results.coordination_score * 100).toFixed(0)}%`}
            sublabel="Multi-agent"
            color={getScoreColor(results.coordination_score)}
          />
        )}
      </div>

      {/* Detection Methods Breakdown */}
      {results.conversation && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Detection Methods</h3>
          <DetectionBreakdown result={results.conversation} />
        </div>
      )}

      {/* All Agents Summary */}
      {results.per_agent_scores && Object.keys(results.per_agent_scores).length > 0 && (
        <div className={styles.section}>
          <h3 className={styles.sectionTitle}>Agent Summary</h3>
          <div className={styles.agentSummaryList}>
            {Object.entries(results.per_agent_scores).map(([agentId, scores]) => {
              const agentDef = trace.agents?.find(a => a.id === agentId);
              return (
                <div key={agentId} className={styles.agentSummaryItem}>
                  <div className={styles.agentSummaryInfo}>
                    <span className={styles.agentSummaryName}>{agentDef?.name || agentId}</span>
                    <span className={styles.agentSummaryRole}>{agentDef?.role || 'agent'}</span>
                  </div>
                  <div className={styles.agentSummaryScore}>
                    <div className={styles.miniBar}>
                      <div
                        className={styles.miniBarFill}
                        style={{ width: `${scores.overall * 100}%`, background: getScoreColor(scores.overall) }}
                      />
                    </div>
                    <span style={{ color: getScoreColor(scores.overall) }}>
                      {(scores.overall * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================
// AGENT VIEW
// ============================================
function AgentView({ trace, results, agentId }: { trace: AgentTraceInput; results?: AgentAnalysisResults; agentId: string }) {
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);
  const agentDef = trace.agents?.find(a => a.id === agentId);
  const agentScore = results?.per_agent_scores?.[agentId];

  const getScoreColor = (score?: number | null) => {
    if (score == null) return 'var(--text-muted)';
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  // Count turns for this agent
  const turnCount = (trace.turns || []).filter(turn => {
    if (turn.agent_interactions) {
      return turn.agent_interactions.some(i => i.agent_id === agentId);
    }
    return agentId === 'agent';
  }).length;

  const toggleMetric = (metric: string) => {
    setExpandedMetric(prev => prev === metric ? null : metric);
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.agentHeader}>
          <span className={styles.agentIcon}>🤖</span>
          <div>
            <h2 className={styles.title}>{agentDef?.name || agentId}</h2>
            {agentDef?.role && <span className={styles.roleTag}>{agentDef.role}</span>}
          </div>
        </div>
        {agentDef?.description && (
          <p className={styles.subtitle}>{agentDef.description}</p>
        )}
      </div>

      {/* Agent Score */}
      {agentScore && (
        <div className={styles.agentScoreSection}>
          <div className={styles.agentScoreHeader}>
            <div className={styles.agentScoreDisplay}>
              <span className={styles.agentScoreValue} style={{ color: getScoreColor(agentScore.overall) }}>
                {(agentScore.overall * 100).toFixed(0)}
              </span>
              <span className={styles.agentScorePercent}>%</span>
            </div>
            <div className={styles.agentScoreBar}>
              <div
                className={styles.agentScoreBarFill}
                style={{
                  width: `${agentScore.overall * 100}%`,
                  background: getScoreColor(agentScore.overall)
                }}
              />
            </div>
          </div>

          {/* Metric Cards with Full Breakdown */}
          <div className={styles.metricCardsGrid}>
            {/* Tool Use */}
            {agentScore.tool_use && (
              <div
                className={`${styles.metricCardExpand} ${expandedMetric === 'tool_use' ? styles.expanded : ''}`}
                onClick={() => toggleMetric('tool_use')}
              >
                <div className={styles.metricCardHeader}>
                  <span className={styles.metricCardIcon}>🔧</span>
                  <span className={styles.metricCardTitle}>Tool Use</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(agentScore.tool_use.efficiency) }}
                  >
                    {(agentScore.tool_use.efficiency * 100).toFixed(0)}%
                  </span>
                  <span className={styles.expandArrow}>{expandedMetric === 'tool_use' ? '▼' : '▶'}</span>
                </div>
                {expandedMetric === 'tool_use' && (
                  <div className={styles.metricCardBody}>
                    <div className={styles.metricStats}>
                      <div className={styles.metricStat}>
                        <span>Total Calls</span>
                        <span>{agentScore.tool_use.total_calls}</span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Correct</span>
                        <span className={styles.good}>{agentScore.tool_use.correct_count}</span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Errors</span>
                        <span className={styles.bad}>{agentScore.tool_use.error_count}</span>
                      </div>
                    </div>
                    {agentScore.tool_use.results && agentScore.tool_use.results.length > 0 && (
                      <div className={styles.metricResults}>
                        {agentScore.tool_use.results.map((r, i) => (
                          <div key={i} className={styles.metricResultItem}>
                            <span className={styles.toolIcon}>🔧</span>
                            <span className={styles.toolName}>{r.tool_name}</span>
                            <span className={`${styles.signal} ${r.signal === 'correct' ? styles.good : styles.bad}`}>
                              {r.signal || 'unknown'}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Self Correction */}
            {agentScore.self_correction && (
              <div
                className={`${styles.metricCardExpand} ${expandedMetric === 'self_correction' ? styles.expanded : ''}`}
                onClick={() => toggleMetric('self_correction')}
              >
                <div className={styles.metricCardHeader}>
                  <span className={styles.metricCardIcon}>🔁</span>
                  <span className={styles.metricCardTitle}>Self-Correction</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(agentScore.self_correction.self_awareness_score) }}
                  >
                    {(agentScore.self_correction.self_awareness_score * 100).toFixed(0)}%
                  </span>
                  <span className={styles.expandArrow}>{expandedMetric === 'self_correction' ? '▼' : '▶'}</span>
                </div>
                {expandedMetric === 'self_correction' && (
                  <div className={styles.metricCardBody}>
                    <div className={styles.metricStats}>
                      <div className={styles.metricStat}>
                        <span>Error Detected</span>
                        <span className={agentScore.self_correction.detected_error ? styles.good : ''}>
                          {agentScore.self_correction.detected_error ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Correction Attempted</span>
                        <span className={agentScore.self_correction.correction_attempt ? styles.good : ''}>
                          {agentScore.self_correction.correction_attempt ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Reasoning Steps</span>
                        <span>{agentScore.self_correction.reasoning_steps}</span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Efficiency</span>
                        <span>{(agentScore.self_correction.correction_efficiency * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}


            {/* Response Quality */}
            {agentScore.response_quality && (
              <div
                className={`${styles.metricCardExpand} ${expandedMetric === 'response_quality' ? styles.expanded : ''}`}
                onClick={() => toggleMetric('response_quality')}
              >
                <div className={styles.metricCardHeader}>
                  <span className={styles.metricCardIcon}>💬</span>
                  <span className={styles.metricCardTitle}>Response Quality</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(agentScore.response_quality.quality_score) }}
                  >
                    {(agentScore.response_quality.quality_score * 100).toFixed(0)}%
                  </span>
                  <span className={styles.expandArrow}>{expandedMetric === 'response_quality' ? '▼' : '▶'}</span>
                </div>
                {expandedMetric === 'response_quality' && (
                  <div className={styles.metricCardBody}>
                    <div className={styles.metricStats}>
                      <div className={styles.metricStat}>
                        <span>Good</span>
                        <span className={styles.good}>{agentScore.response_quality.good_count}</span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Bad</span>
                        <span className={styles.bad}>{agentScore.response_quality.bad_count}</span>
                      </div>
                      <div className={styles.metricStat}>
                        <span>Total</span>
                        <span>{agentScore.response_quality.total_responses}</span>
                      </div>
                    </div>
                    {agentScore.response_quality.results && agentScore.response_quality.results.length > 0 && (
                      <div className={styles.metricResults}>
                        {agentScore.response_quality.results.map((r, i) => (
                          <div key={i} className={styles.metricResultItem}>
                            <span className={styles.turnLabel}>Turn {r.turn_index + 1}</span>
                            <span className={`${styles.signal} ${r.is_bad ? styles.bad : styles.good}`}>
                              {r.is_bad ? '❌' : '✓'} {r.detection_type}
                            </span>
                            <span className={styles.confidence}>{(r.confidence * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Reasoning */}
            {agentScore.reasoning && (
              <div
                className={`${styles.metricCardExpand} ${expandedMetric === 'reasoning' ? styles.expanded : ''}`}
                onClick={() => toggleMetric('reasoning')}
              >
                {/* Cast to unknown first then ReasoningData because API types say it's a number */}
                {(() => {
                  const reasoningData = agentScore.reasoning as unknown as ReasoningData;
                  // Handle case where it might still be a number (backward compatibility)
                  if (typeof agentScore.reasoning === 'number') {
                    return (
                      <div className={styles.metricCardSimple}>
                        <span className={styles.metricCardIcon}>🧠</span>
                        <span className={styles.metricCardTitle}>Reasoning</span>
                        <span
                          className={styles.metricCardScore}
                          style={{ color: getScoreColor(agentScore.reasoning) }}
                        >
                          {(agentScore.reasoning * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  }

                  return (
                    <>
                      <div className={styles.metricCardHeader}>
                        <span className={styles.metricCardIcon}>🧠</span>
                        <span className={styles.metricCardTitle}>Reasoning</span>
                        <span
                          className={styles.metricCardScore}
                          style={{ color: getScoreColor(reasoningData.quality_score) }}
                        >
                          {(reasoningData.quality_score * 100).toFixed(0)}%
                        </span>
                        <span className={styles.expandArrow}>{expandedMetric === 'reasoning' ? '▼' : '▶'}</span>
                      </div>
                      {expandedMetric === 'reasoning' && (
                        <div className={styles.metricCardBody}>
                          {/* Assessment */}
                          <div className={styles.assessmentBox}>
                            {reasoningData.assessment}
                          </div>
                          <div className={styles.metricStats}>
                            <div className={styles.metricStat}>
                              <span>Total Thoughts</span>
                              <span>{reasoningData.total_thoughts}</span>
                            </div>
                            <div className={styles.metricStat}>
                              <span>Reasoning Depth</span>
                              <span className={styles.good}>{(reasoningData.reasoning_depth * 100).toFixed(0)}%</span>
                            </div>
                            <div className={styles.metricStat}>
                              <span>Clear</span>
                              <span className={styles.good}>{reasoningData.clear_thoughts}</span>
                            </div>
                            <div className={styles.metricStat}>
                              <span>Structured</span>
                              <span className={styles.good}>{reasoningData.structured_thoughts}</span>
                            </div>
                          </div>
                          {reasoningData.thoughts && reasoningData.thoughts.length > 0 && (
                            <div className={styles.metricResults}>
                              <div className={styles.thoughtsHeader}>Thought Samples:</div>
                              {reasoningData.thoughts.map((t, i) => (
                                <div key={i} className={styles.thoughtResultItem}>
                                  <div className={styles.thoughtHeader}>
                                    <span className={styles.turnLabel}>Turn {t.turn_index + 1}</span>
                                    <span className={`${styles.signal} ${t.score >= 0.6 ? styles.good : styles.bad}`}>
                                      {(t.score * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                  <p className={styles.thoughtPreview}>{t.thought_preview}</p>
                                  <div className={styles.thoughtTags}>
                                    {t.is_clear && <span className={styles.thoughtTag}>✓ Clear</span>}
                                    {t.is_structured && <span className={styles.thoughtTag}>✓ Structured</span>}
                                    {!t.is_clear && <span className={`${styles.thoughtTag} ${styles.thoughtTagBad}`}>Brief</span>}
                                    {!t.is_structured && <span className={`${styles.thoughtTag} ${styles.thoughtTagBad}`}>Unstructured</span>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  );
                })()}
              </div>
            )}

            {/* Handoff */}
            {agentScore.handoff && (
              <div
                className={`${styles.metricCardExpand} ${expandedMetric === 'handoff' ? styles.expanded : ''}`}
                onClick={() => toggleMetric('handoff')}
              >
                <div className={styles.metricCardHeader}>
                  <span className={styles.metricCardIcon}>🔄</span>
                  <span className={styles.metricCardTitle}>Handoff</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(agentScore.handoff.quality_score) }}
                  >
                    {(agentScore.handoff.quality_score * 100).toFixed(0)}%
                  </span>
                  <span className={styles.expandArrow}>{expandedMetric === 'handoff' ? '▼' : '▶'}</span>
                </div>
                {expandedMetric === 'handoff' && (
                  <div className={styles.metricCardBody}>
                    <div className={styles.metricStats}>
                      <div className={styles.metricStat}>
                        <span>Total Handoffs</span>
                        <span>{agentScore.handoff.total_handoffs}</span>
                      </div>
                    </div>
                    {agentScore.handoff.targets && agentScore.handoff.targets.length > 0 && (
                      <div className={styles.metricResults}>
                        {agentScore.handoff.targets.map((t, i) => (
                          <div key={i} className={styles.metricResultItem}>
                            <span className={styles.turnLabel}>Turn {t.turn_index + 1}</span>
                            <span>→ {t.target_agent}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Interactions count */}
          <div className={styles.interactionsCount}>
            <span className={styles.interactionsLabel}>💬 {agentScore.interactions_count} Interactions</span>
          </div>
        </div>
      )}

      {/* Fallback: Show global metrics if no per-agent scores */}
      {!agentScore && results && (
        <div className={styles.agentScoreSection}>
          <div className={styles.fallbackMetrics}>
            <p className={styles.fallbackNote}>Showing global metrics (per-agent breakdown not available)</p>
            <div className={styles.metricCardsGrid}>
              {results.tools && (
                <div className={styles.metricCardSimple}>
                  <span className={styles.metricCardIcon}>🔧</span>
                  <span className={styles.metricCardTitle}>Tool Use</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(results.tools.efficiency) }}
                  >
                    {(results.tools.efficiency * 100).toFixed(0)}%
                  </span>
                </div>
              )}
              {results.self_correction && (
                <div className={styles.metricCardSimple}>
                  <span className={styles.metricCardIcon}>🔁</span>
                  <span className={styles.metricCardTitle}>Self-Correction</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(results.self_correction.self_awareness_score) }}
                  >
                    {(results.self_correction.self_awareness_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}
              {results.conversation && (
                <div className={styles.metricCardSimple}>
                  <span className={styles.metricCardIcon}>💬</span>
                  <span className={styles.metricCardTitle}>Response Quality</span>
                  <span
                    className={styles.metricCardScore}
                    style={{ color: getScoreColor(results.conversation.good_responses / results.conversation.total_responses) }}
                  >
                    {results.conversation.good_responses}/{results.conversation.total_responses}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Agent Info */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Agent Info</h3>
        <div className={styles.infoGrid}>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Turns</span>
            <span className={styles.infoValue}>{turnCount}</span>
          </div>
          {agentDef?.capabilities && (
            <div className={styles.infoItem}>
              <span className={styles.infoLabel}>Capabilities</span>
              <div className={styles.tagList}>
                {agentDef.capabilities.map((cap, i) => (
                  <span key={i} className={styles.capTag}>{cap}</span>
                ))}
              </div>
            </div>
          )}
          {agentDef?.tools_available && agentDef.tools_available.length > 0 && (
            <div className={styles.infoItem}>
              <span className={styles.infoLabel}>Tools</span>
              <div className={styles.tagList}>
                {agentDef.tools_available.map((tool, i) => (
                  <span key={i} className={styles.toolTag}>{tool.name}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Issues & Recommendations */}
      {agentScore && (agentScore.issues.length > 0 || agentScore.recommendations.length > 0) && (
        <div className={styles.section}>
          {agentScore.issues.length > 0 && (
            <>
              <h3 className={styles.sectionTitle}>⚠️ Issues</h3>
              <div className={styles.issuesList}>
                {agentScore.issues.map((issue, i) => (
                  <div key={i} className={styles.issueItem}>{issue}</div>
                ))}
              </div>
            </>
          )}
          {agentScore.recommendations.length > 0 && (
            <>
              <h3 className={styles.sectionTitle}>💡 Recommendations</h3>
              <div className={styles.recommendationsList}>
                {agentScore.recommendations.map((rec, i) => (
                  <div key={i} className={styles.recommendationItem}>{rec}</div>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================
// TURN VIEW
// ============================================
function TurnView({ trace, results, turnIndex, agentId }: {
  trace: AgentTraceInput;
  results?: AgentAnalysisResults;
  turnIndex: number;
  agentId?: string;
}) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]));
  const turn = trace.turns?.[turnIndex];
  if (!turn) return null;

  const turnResult = results?.conversation?.results?.find(r => r.step_index === turnIndex);

  const toggleStep = (idx: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  const getScoreColor = (score: number, isBad?: boolean) => {
    if (isBad) return 'var(--accent-red)';
    if (score >= 0.8) return 'var(--accent-green)';
    if (score >= 0.5) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };


  // Get interactions for this turn
  const interactions: AgentInteractionInput[] = turn.agent_interactions || (turn.agent_response ? [{
    agent_id: 'agent',
    agent_steps: turn.agent_steps,
    agent_response: turn.agent_response,
  }] : []);

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.turnHeader}>
          <span className={styles.turnNumber}>Turn {turnIndex + 1}</span>
          {turnResult && (
            <span
              className={styles.detectionBadge}
              style={{ background: getScoreColor(turnResult.confidence, turnResult.is_bad) }}
            >
              {turnResult.is_bad ? '❌' : '✓'} {turnResult.detection_type.toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {/* Score Breakdown */}
      {turnResult && (
        <div className={styles.scoreBreakdown}>
          <div className={styles.breakdownHeader}>
            <span className={styles.breakdownLabel}>Confidence</span>
            <span
              className={styles.breakdownValue}
              style={{ color: getScoreColor(turnResult.confidence, turnResult.is_bad) }}
            >
              {(turnResult.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className={styles.breakdownBar}>
            <div
              className={styles.breakdownFill}
              style={{
                width: `${turnResult.confidence * 100}%`,
                background: getScoreColor(turnResult.confidence, turnResult.is_bad)
              }}
            />
          </div>
          {turnResult.reason && (
            <p className={styles.breakdownReason}>{turnResult.reason}</p>
          )}
        </div>
      )}

      {/* User Message */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>👤 User Message</h3>
        <div className={styles.messageBox}>
          {turn.user_message || '(No user message)'}
        </div>
      </div>

      {/* Agent Interactions */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>🤖 Agent Response{interactions.length > 1 ? 's' : ''}</h3>

        {interactions.map((interaction, idx) => {
          const agentDef = trace.agents?.find(a => a.id === interaction.agent_id);
          const isFiltered = agentId && interaction.agent_id !== agentId;

          return (
            <div
              key={idx}
              className={`${styles.interactionBlock} ${isFiltered ? styles.dimmed : ''}`}
            >
              {interactions.length > 1 && (
                <div className={styles.interactionHeader}>
                  <span className={styles.interactionAgent}>
                    {agentDef?.name || interaction.agent_id}
                  </span>
                  {interaction.latency_ms && (
                    <span className={styles.latencyBadge}>{interaction.latency_ms}ms</span>
                  )}
                </div>
              )}

              {/* Steps */}
              {interaction.agent_steps && interaction.agent_steps.length > 0 && (
                <div className={styles.stepsSection}>
                  <div className={styles.stepsHeader}>
                    <span>🧠 Reasoning ({interaction.agent_steps.length} steps)</span>
                  </div>
                  {interaction.agent_steps.map((step, stepIdx) => (
                    <StepDetail
                      key={stepIdx}
                      step={step}
                      index={stepIdx}
                      expanded={expandedSteps.has(stepIdx)}
                      onToggle={() => toggleStep(stepIdx)}
                    />
                  ))}
                </div>
              )}

              {/* Response */}
              {interaction.agent_response && (
                <div className={styles.responseBox}>
                  {interaction.agent_response}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================
// HELPER COMPONENTS
// ============================================
function MetricCard({ icon, label, value, sublabel, color }: {
  icon: string;
  label: string;
  value: string;
  sublabel?: string;
  color?: string;
}) {
  return (
    <div className={styles.metricCard}>
      <span className={styles.metricIcon}>{icon}</span>
      <div className={styles.metricContent}>
        <span className={styles.metricValue} style={{ color }}>{value}</span>
        <span className={styles.metricLabel}>{label}</span>
        {sublabel && <span className={styles.metricSublabel}>{sublabel}</span>}
      </div>
    </div>
  );
}

function DetectionBreakdown({ result }: { result: ConversationAnalysisResult }) {
  const max = Math.max(
    result.ccm_detections,
    result.rdm_detections,
    result.llm_judge_detections,
    result.hallucination_detections,
    1
  );

  const methods = [
    { key: 'ccm', label: 'CCM', count: result.ccm_detections, color: 'var(--accent-cyan)' },
    { key: 'rdm', label: 'RDM', count: result.rdm_detections, color: 'var(--accent-purple)' },
    { key: 'llm_judge', label: 'LLM Judge', count: result.llm_judge_detections, color: 'var(--accent-orange)' },
    { key: 'hallucination', label: 'Hallu', count: result.hallucination_detections, color: 'var(--accent-red)' },
  ];

  return (
    <div className={styles.detectionBreakdown}>
      {methods.map(method => (
        <div key={method.key} className={styles.detectionRow}>
          <span className={styles.detectionLabel} style={{ background: method.color }}>
            {method.label}
          </span>
          <div className={styles.detectionBarContainer}>
            <div
              className={styles.detectionBar}
              style={{ width: `${(method.count / max) * 100}%`, background: method.color }}
            />
          </div>
          <span className={styles.detectionCount}>{method.count}</span>
        </div>
      ))}
    </div>
  );
}

function StepDetail({ step, index, expanded, onToggle }: {
  step: AgentStepInput | any;
  index: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className={`${styles.stepCard} ${expanded ? styles.expanded : ''}`}>
      <button className={styles.stepHeader} onClick={onToggle}>
        <span className={styles.stepIndex}>Step {index + 1}</span>
        {step.tool_call && (
          <span className={styles.toolBadge}>
            🔧 {step.tool_call.tool_name || step.tool_call.name}
          </span>
        )}
        {step.thought && !step.tool_call && (
          <span className={styles.thoughtPreview}>
            💭 {step.thought.slice(0, 40)}...
          </span>
        )}
        <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
      </button>

      {expanded && (
        <div className={styles.stepContent}>
          {step.thought && (
            <div className={styles.stepItem}>
              <span className={styles.stepItemLabel}>💭 Thought</span>
              <p className={styles.stepItemContent}>{step.thought}</p>
            </div>
          )}
          {step.action && (
            <div className={styles.stepItem}>
              <span className={styles.stepItemLabel}>⚡ Action</span>
              <p className={styles.stepItemContent}>{step.action}</p>
            </div>
          )}
          {step.tool_call && (
            <div className={styles.stepItem}>
              <span className={styles.stepItemLabel}>🔧 Tool Call</span>
              <div className={styles.toolCallDetail}>
                <code className={styles.toolName}>{step.tool_call.tool_name || step.tool_call.name}</code>
                {step.tool_call.parameters && (
                  <pre className={styles.toolParams}>
                    {JSON.stringify(step.tool_call.parameters, null, 2)}
                  </pre>
                )}
                {step.tool_call.result && (
                  <div className={styles.toolResult}>
                    <span className={styles.resultLabel}>✓ Result:</span>
                    <span>{String(step.tool_call.result).slice(0, 200)}</span>
                  </div>
                )}
                {step.tool_call.error && (
                  <div className={styles.toolError}>
                    <span className={styles.errorLabel}>✗ Error:</span>
                    <span>{step.tool_call.error}</span>
                  </div>
                )}
                {step.tool_call.latency_ms && (
                  <span className={styles.toolLatency}>{step.tool_call.latency_ms}ms</span>
                )}
              </div>
            </div>
          )}
          {step.observation && (
            <div className={styles.stepItem}>
              <span className={styles.stepItemLabel}>👁️ Observation</span>
              <p className={styles.stepItemContent}>{step.observation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

