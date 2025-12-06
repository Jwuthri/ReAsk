'use client';

import { useState, useMemo } from 'react';
import styles from './StreamingAnalysisTable.module.css';

interface TurnResult {
    step_index: number;
    is_bad: boolean;
    detection_type: string;
    confidence: number;
    reason?: string;
}

interface ToolCall {
    tool_name?: string;
    parameters?: Record<string, unknown>;
    result?: string;
    error?: string;
    latency_ms?: number;
}

interface Step {
    thought?: string;
    tool_call?: ToolCall;
    action?: string;
    observation?: string;
}

interface AgentInteraction {
    agent_id: string;
    agent_response?: string;
    agent_steps?: Step[];
    latency_ms?: number;
}

interface Turn {
    turn_index?: number;
    user_message?: string;
    agent_response?: string;
    agent_interactions?: AgentInteraction[];
    agent_steps?: Step[];
}

interface StreamingAnalysisTableProps {
    turns: Turn[];
    turnResults: TurnResult[];
    currentTurn?: number;
    currentAnalysis?: string;
    isAnalyzing: boolean;
}

// Flatten all steps from a turn into a single array with agent info
interface FlatStep {
    agent_id: string;
    step_index: number;
    type: 'thought' | 'tool_call' | 'response';
    content: string;
    tool_name?: string;
    tool_params?: Record<string, unknown>;
    tool_result?: string;
    tool_error?: string;
    latency_ms?: number;
}

export default function StreamingAnalysisTable({
    turns,
    turnResults,
    currentTurn = -1,
    currentAnalysis = '',
    isAnalyzing,
}: StreamingAnalysisTableProps) {
    const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

    // Build a map of turn index to result
    const resultsMap = useMemo(() => {
        const map: Record<number, TurnResult> = {};
        turnResults.forEach((r) => {
            map[r.step_index] = r;
        });
        return map;
    }, [turnResults]);

    const toggleRow = (index: number) => {
        setExpandedRows(prev => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    // Flatten all steps from a turn
    const getFlatSteps = (turn: Turn): FlatStep[] => {
        const steps: FlatStep[] = [];
        let stepIdx = 0;

        const interactions = turn.agent_interactions && turn.agent_interactions.length > 0
            ? turn.agent_interactions
            : [{ agent_id: 'agent', agent_response: turn.agent_response, agent_steps: turn.agent_steps }];

        for (const interaction of interactions) {
            // Add thoughts and tool calls
            if (interaction.agent_steps) {
                for (const step of interaction.agent_steps) {
                    if (step.thought) {
                        steps.push({
                            agent_id: interaction.agent_id,
                            step_index: stepIdx++,
                            type: 'thought',
                            content: step.thought,
                        });
                    }
                    if (step.tool_call) {
                        steps.push({
                            agent_id: interaction.agent_id,
                            step_index: stepIdx++,
                            type: 'tool_call',
                            content: step.tool_call.tool_name || 'unknown',
                            tool_name: step.tool_call.tool_name,
                            tool_params: step.tool_call.parameters,
                            tool_result: step.tool_call.result,
                            tool_error: step.tool_call.error,
                            latency_ms: step.tool_call.latency_ms,
                        });
                    }
                }
            }
            // Add response
            if (interaction.agent_response) {
                steps.push({
                    agent_id: interaction.agent_id,
                    step_index: stepIdx++,
                    type: 'response',
                    content: interaction.agent_response,
                    latency_ms: interaction.latency_ms,
                });
            }
        }
        return steps;
    };

    // Get agents involved
    const getAgents = (turn: Turn): string[] => {
        if (turn.agent_interactions && turn.agent_interactions.length > 0) {
            return turn.agent_interactions.map(i => i.agent_id);
        }
        return ['agent'];
    };

    const getStatusInfo = (turnIndex: number) => {
        const result = resultsMap[turnIndex];
        if (result) {
            return {
                status: result.is_bad ? 'bad' : 'good',
                icon: result.is_bad ? '✗' : '✓',
                label: result.detection_type.toUpperCase(),
                confidence: result.confidence,
                reason: result.reason,
            };
        }
        if (currentTurn === turnIndex) {
            return { status: 'analyzing', icon: '', label: 'Analyzing...', confidence: 0, reason: '' };
        }
        return { status: 'pending', icon: '○', label: 'Pending', confidence: 0, reason: '' };
    };

    const completedCount = turnResults.length;
    const badCount = turnResults.filter(r => r.is_bad).length;
    const goodCount = completedCount - badCount;
    const progressPercent = turns.length > 0 ? (completedCount / turns.length) * 100 : 0;

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'thought': return '💭';
            case 'tool_call': return '🔧';
            case 'response': return '💬';
            default: return '•';
        }
    };

    const getTypeLabel = (type: string) => {
        switch (type) {
            case 'thought': return 'Thought';
            case 'tool_call': return 'Tool Call';
            case 'response': return 'Response';
            default: return type;
        }
    };

    return (
        <div className={styles.container}>
            {/* Header */}
            <div className={styles.header}>
                <div className={styles.headerLeft}>
                    <h3 className={styles.title}>
                        <span className={styles.titleIcon}>📊</span>
                        Conversation Analysis
                    </h3>
                    <span className={styles.subtitle}>
                        {turns.length} messages • {currentAnalysis || 'Ready'}
                    </span>
                </div>

                <div className={styles.headerRight}>
                    <div className={styles.stats}>
                        <div className={`${styles.stat} ${styles.statGood}`}>
                            <span className={styles.statValue}>{goodCount}</span>
                            <span className={styles.statLabel}>Good</span>
                        </div>
                        <div className={`${styles.stat} ${styles.statBad}`}>
                            <span className={styles.statValue}>{badCount}</span>
                            <span className={styles.statLabel}>Issues</span>
                        </div>
                    </div>
                    <div className={styles.progressWrapper}>
                        <div className={styles.progressBar}>
                            <div
                                className={styles.progressFill}
                                style={{ width: `${progressPercent}%` }}
                            />
                        </div>
                        <span className={styles.progressText}>
                            {completedCount}/{turns.length}
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Table */}
            <div className={styles.tableWrapper}>
                <table className={styles.mainTable}>
                    <thead>
                        <tr>
                            <th className={styles.colExpand}></th>
                            <th className={styles.colIndex}>#</th>
                            <th className={styles.colStatus}>Status</th>
                            <th className={styles.colMessage}>Message</th>
                            <th className={styles.colAgents}>Agents</th>
                            <th className={styles.colSteps}>Steps</th>
                            <th className={styles.colDetection}>Detection</th>
                            <th className={styles.colConfidence}>Conf.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {turns.map((turn, index) => {
                            const isExpanded = expandedRows.has(index);
                            const flatSteps = getFlatSteps(turn);
                            const agents = getAgents(turn);
                            const statusInfo = getStatusInfo(index);
                            const isCurrentlyAnalyzing = currentTurn === index;

                            return (
                                <>
                                    {/* Main Row */}
                                    <tr
                                        key={`row-${index}`}
                                        className={`${styles.mainRow} ${styles[statusInfo.status]} ${isExpanded ? styles.expanded : ''}`}
                                        onClick={() => toggleRow(index)}
                                    >
                                        {/* Expand button */}
                                        <td className={styles.cellExpand}>
                                            <button className={`${styles.expandBtn} ${isExpanded ? styles.expandedBtn : ''}`}>
                                                <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
                                                    <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
                                                </svg>
                                            </button>
                                        </td>

                                        {/* Index */}
                                        <td className={styles.cellIndex}>
                                            <span className={styles.indexNum}>{index + 1}</span>
                                        </td>

                                        {/* Status */}
                                        <td className={styles.cellStatus}>
                                            <div className={`${styles.statusBadge} ${styles[`badge${statusInfo.status.charAt(0).toUpperCase() + statusInfo.status.slice(1)}`]}`}>
                                                {statusInfo.status === 'analyzing' ? (
                                                    <span className={styles.spinner} />
                                                ) : (
                                                    <span>{statusInfo.icon}</span>
                                                )}
                                            </div>
                                        </td>

                                        {/* Message */}
                                        <td className={styles.cellMessage}>
                                            <div className={styles.messageContent}>
                                                <span className={styles.userTag}>USER</span>
                                                <span className={styles.messageText}>
                                                    {turn.user_message?.slice(0, 80) || '(No message)'}
                                                    {(turn.user_message?.length || 0) > 80 ? '...' : ''}
                                                </span>
                                            </div>
                                        </td>

                                        {/* Agents */}
                                        <td className={styles.cellAgents}>
                                            <div className={styles.agentChips}>
                                                {agents.slice(0, 2).map((a, i) => (
                                                    <span key={i} className={styles.agentChip}>{a}</span>
                                                ))}
                                                {agents.length > 2 && (
                                                    <span className={styles.moreChip}>+{agents.length - 2}</span>
                                                )}
                                            </div>
                                        </td>

                                        {/* Steps count */}
                                        <td className={styles.cellSteps}>
                                            <span className={styles.stepCount}>{flatSteps.length}</span>
                                        </td>

                                        {/* Detection */}
                                        <td className={styles.cellDetection}>
                                            {statusInfo.status === 'pending' ? (
                                                <span className={styles.pending}>—</span>
                                            ) : statusInfo.status === 'analyzing' ? (
                                                <span className={styles.analyzing}>...</span>
                                            ) : (
                                                <span className={`${styles.detectionBadge} ${styles[statusInfo.status]}`}>
                                                    {statusInfo.label}
                                                </span>
                                            )}
                                        </td>

                                        {/* Confidence */}
                                        <td className={styles.cellConfidence}>
                                            {statusInfo.confidence > 0 ? (
                                                <span className={styles.confValue}>
                                                    {(statusInfo.confidence * 100).toFixed(0)}%
                                                </span>
                                            ) : (
                                                <span className={styles.pending}>—</span>
                                            )}
                                        </td>
                                    </tr>

                                    {/* Expanded Steps Table */}
                                    {isExpanded && (
                                        <tr key={`expanded-${index}`} className={styles.expandedRow}>
                                            <td colSpan={8} className={styles.expandedCell}>
                                                <div className={styles.stepsContainer}>
                                                    <table className={styles.stepsTable}>
                                                        <thead>
                                                            <tr>
                                                                <th className={styles.stepColIndex}>#</th>
                                                                <th className={styles.stepColAgent}>Agent</th>
                                                                <th className={styles.stepColType}>Type</th>
                                                                <th className={styles.stepColContent}>Content</th>
                                                                <th className={styles.stepColResult}>Result / Output</th>
                                                                <th className={styles.stepColLatency}>Latency</th>
                                                                <th className={styles.stepColStatus}>Status</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {flatSteps.map((step, sIdx) => (
                                                                <tr key={sIdx} className={`${styles.stepRow} ${step.tool_error ? styles.stepError : ''}`}>
                                                                    {/* Step Index */}
                                                                    <td className={styles.stepCellIndex}>
                                                                        <span className={styles.stepNum}>{sIdx + 1}</span>
                                                                    </td>

                                                                    {/* Agent */}
                                                                    <td className={styles.stepCellAgent}>
                                                                        <span className={styles.stepAgentChip}>{step.agent_id}</span>
                                                                    </td>

                                                                    {/* Type */}
                                                                    <td className={styles.stepCellType}>
                                                                        <span className={`${styles.typeTag} ${styles[`type${step.type.charAt(0).toUpperCase() + step.type.slice(1).replace('_', '')}`]}`}>
                                                                            {getTypeIcon(step.type)} {getTypeLabel(step.type)}
                                                                        </span>
                                                                    </td>

                                                                    {/* Content */}
                                                                    <td className={styles.stepCellContent}>
                                                                        {step.type === 'tool_call' ? (
                                                                            <div className={styles.toolContent}>
                                                                                <span className={styles.toolName}>{step.tool_name}</span>
                                                                                {step.tool_params && (
                                                                                    <div className={styles.toolParams}>
                                                                                        {Object.entries(step.tool_params).slice(0, 2).map(([k, v]) => (
                                                                                            <span key={k} className={styles.param}>
                                                                                                {k}: <code>{typeof v === 'string' ? v.slice(0, 30) : JSON.stringify(v).slice(0, 30)}</code>
                                                                                            </span>
                                                                                        ))}
                                                                                    </div>
                                                                                )}
                                                                            </div>
                                                                        ) : (
                                                                            <span className={styles.contentText}>
                                                                                {step.content.slice(0, 100)}
                                                                                {step.content.length > 100 ? '...' : ''}
                                                                            </span>
                                                                        )}
                                                                    </td>

                                                                    {/* Result */}
                                                                    <td className={styles.stepCellResult}>
                                                                        {step.tool_result ? (
                                                                            <span className={styles.resultSuccess}>
                                                                                {step.tool_result.slice(0, 60)}
                                                                                {step.tool_result.length > 60 ? '...' : ''}
                                                                            </span>
                                                                        ) : step.tool_error ? (
                                                                            <span className={styles.resultError}>
                                                                                ✗ {step.tool_error.slice(0, 50)}
                                                                            </span>
                                                                        ) : (
                                                                            <span className={styles.resultNa}>—</span>
                                                                        )}
                                                                    </td>

                                                                    {/* Latency */}
                                                                    <td className={styles.stepCellLatency}>
                                                                        {step.latency_ms ? (
                                                                            <span className={styles.latencyValue}>{step.latency_ms}ms</span>
                                                                        ) : (
                                                                            <span className={styles.resultNa}>—</span>
                                                                        )}
                                                                    </td>

                                                                    {/* Status (placeholder for future step-level analysis) */}
                                                                    <td className={styles.stepCellStatus}>
                                                                        {step.tool_error ? (
                                                                            <span className={styles.stepStatusBad}>Error</span>
                                                                        ) : step.tool_result ? (
                                                                            <span className={styles.stepStatusGood}>OK</span>
                                                                        ) : (
                                                                            <span className={styles.stepStatusNeutral}>—</span>
                                                                        )}
                                                                    </td>
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>

                                                    {/* Analysis Reason */}
                                                    {statusInfo.reason && (
                                                        <div className={`${styles.analysisBox} ${styles[statusInfo.status]}`}>
                                                            <div className={styles.analysisHeader}>
                                                                <span className={styles.analysisIcon}>
                                                                    {statusInfo.status === 'good' ? '✓' : '⚠'}
                                                                </span>
                                                                <span className={styles.analysisLabel}>{statusInfo.label}</span>
                                                                <span className={styles.analysisConf}>{(statusInfo.confidence * 100).toFixed(0)}%</span>
                                                            </div>
                                                            <p className={styles.analysisReason}>{statusInfo.reason}</p>
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Loading indicator */}
            {isAnalyzing && (
                <div className={styles.loadingBar}>
                    <div className={styles.loadingFill} />
                </div>
            )}
        </div>
    );
}
