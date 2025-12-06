'use client';

import { useMemo } from 'react';
import styles from './StreamingAnalysisTable.module.css';

interface TurnResult {
    step_index: number;
    is_bad: boolean;
    detection_type: string;
    confidence: number;
    reason: string;
}

interface Turn {
    turn_index?: number;
    user_message?: string;
    agent_response?: string;
    agent_interactions?: Array<{
        agent_id: string;
        agent_response?: string;
        agent_steps?: Array<{
            thought?: string;
            tool_call?: {
                tool_name?: string;
                result?: string;
                error?: string;
            };
        }>;
    }>;
}

interface StreamingAnalysisTableProps {
    turns: Turn[];
    turnResults: TurnResult[];
    currentTurn?: number;
    currentAnalysis?: string;
    isAnalyzing: boolean;
}

export default function StreamingAnalysisTable({
    turns,
    turnResults,
    currentTurn = -1,
    currentAnalysis = '',
    isAnalyzing,
}: StreamingAnalysisTableProps) {
    // Build a map of turn index to result for quick lookup
    const resultsMap = useMemo(() => {
        const map: Record<number, TurnResult> = {};
        turnResults.forEach((r) => {
            map[r.step_index] = r;
        });
        return map;
    }, [turnResults]);

    // Get agents involved in a turn
    const getAgents = (turn: Turn): string[] => {
        if (turn.agent_interactions) {
            return turn.agent_interactions.map((i) => i.agent_id);
        }
        return ['agent'];
    };

    // Get tool calls from a turn
    const getToolCalls = (turn: Turn): string[] => {
        const tools: string[] = [];
        if (turn.agent_interactions) {
            turn.agent_interactions.forEach((i) => {
                i.agent_steps?.forEach((s) => {
                    if (s.tool_call?.tool_name) {
                        tools.push(s.tool_call.tool_name);
                    }
                });
            });
        }
        return tools;
    };

    // Get response preview
    const getResponse = (turn: Turn): string => {
        if (turn.agent_response) return turn.agent_response;
        if (turn.agent_interactions) {
            const lastInteraction = turn.agent_interactions[turn.agent_interactions.length - 1];
            return lastInteraction?.agent_response || '';
        }
        return '';
    };

    const getStatusClass = (turnIndex: number): string => {
        const result = resultsMap[turnIndex];
        if (result) {
            return result.is_bad ? styles.bad : styles.good;
        }
        if (currentTurn === turnIndex) {
            return styles.analyzing;
        }
        return styles.pending;
    };

    const getDetectionBadge = (result: TurnResult | undefined) => {
        if (!result) return null;

        const typeColors: Record<string, string> = {
            'none': 'var(--accent-green)',
            'ccm': 'var(--accent-orange)',
            'rdm': 'var(--accent-red)',
            'hallucination': 'var(--accent-purple)',
            'llm_judge': 'var(--accent-orange)',
        };

        return (
            <span
                className={styles.detectionBadge}
                style={{
                    background: `${typeColors[result.detection_type] || 'var(--text-secondary)'}20`,
                    borderColor: typeColors[result.detection_type] || 'var(--text-secondary)',
                }}
            >
                {result.detection_type.toUpperCase()}
            </span>
        );
    };

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <div className={styles.headerTitle}>
                    <span className={styles.headerIcon}>📊</span>
                    <h3>Real-Time Analysis</h3>
                </div>
                {isAnalyzing && (
                    <div className={styles.headerStatus}>
                        <span className={styles.statusDot} />
                        <span>{currentAnalysis || 'Analyzing...'}</span>
                    </div>
                )}
                <div className={styles.headerProgress}>
                    <span className={styles.progressText}>
                        {turnResults.length} / {turns.length} turns
                    </span>
                    <div className={styles.progressBar}>
                        <div
                            className={styles.progressFill}
                            style={{ width: `${(turnResults.length / Math.max(turns.length, 1)) * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            <div className={styles.tableWrapper}>
                <table className={styles.table}>
                    <thead>
                        <tr>
                            <th className={styles.colIndex}>#</th>
                            <th className={styles.colStatus}>Status</th>
                            <th className={styles.colUser}>User Message</th>
                            <th className={styles.colAgents}>Agents</th>
                            <th className={styles.colTools}>Tools</th>
                            <th className={styles.colResponse}>Response</th>
                            <th className={styles.colDetection}>Detection</th>
                            <th className={styles.colConfidence}>Conf.</th>
                            <th className={styles.colReason}>Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {turns.map((turn, index) => {
                            const result = resultsMap[index];
                            const isCurrentlyAnalyzing = currentTurn === index;
                            const agents = getAgents(turn);
                            const tools = getToolCalls(turn);
                            const response = getResponse(turn);

                            return (
                                <tr
                                    key={index}
                                    className={`${styles.row} ${getStatusClass(index)}`}
                                >
                                    {/* Index */}
                                    <td className={styles.cellIndex}>
                                        <span className={styles.turnNumber}>{index + 1}</span>
                                    </td>

                                    {/* Status */}
                                    <td className={styles.cellStatus}>
                                        {result ? (
                                            result.is_bad ? (
                                                <span className={styles.statusBad}>❌</span>
                                            ) : (
                                                <span className={styles.statusGood}>✅</span>
                                            )
                                        ) : isCurrentlyAnalyzing ? (
                                            <span className={styles.statusAnalyzing}>
                                                <span className={styles.spinner} />
                                            </span>
                                        ) : (
                                            <span className={styles.statusPending}>⏳</span>
                                        )}
                                    </td>

                                    {/* User Message */}
                                    <td className={styles.cellUser}>
                                        <div className={styles.cellContent}>
                                            {turn.user_message?.slice(0, 60)}
                                            {(turn.user_message?.length || 0) > 60 ? '...' : ''}
                                        </div>
                                    </td>

                                    {/* Agents */}
                                    <td className={styles.cellAgents}>
                                        <div className={styles.agentList}>
                                            {agents.map((agent, i) => (
                                                <span key={i} className={styles.agentChip}>{agent}</span>
                                            ))}
                                        </div>
                                    </td>

                                    {/* Tools */}
                                    <td className={styles.cellTools}>
                                        {tools.length > 0 ? (
                                            <div className={styles.toolList}>
                                                {tools.slice(0, 2).map((tool, i) => (
                                                    <span key={i} className={styles.toolChip}>{tool}</span>
                                                ))}
                                                {tools.length > 2 && (
                                                    <span className={styles.moreChip}>+{tools.length - 2}</span>
                                                )}
                                            </div>
                                        ) : (
                                            <span className={styles.emptyCell}>-</span>
                                        )}
                                    </td>

                                    {/* Response */}
                                    <td className={styles.cellResponse}>
                                        <div className={styles.cellContent}>
                                            {response.slice(0, 50)}
                                            {response.length > 50 ? '...' : ''}
                                        </div>
                                    </td>

                                    {/* Detection Type */}
                                    <td className={styles.cellDetection}>
                                        {result ? (
                                            getDetectionBadge(result)
                                        ) : isCurrentlyAnalyzing ? (
                                            <span className={styles.analyzing}>...</span>
                                        ) : (
                                            <span className={styles.emptyCell}>-</span>
                                        )}
                                    </td>

                                    {/* Confidence */}
                                    <td className={styles.cellConfidence}>
                                        {result ? (
                                            <span
                                                className={styles.confidenceValue}
                                                style={{
                                                    color: result.confidence >= 0.8
                                                        ? 'var(--accent-green)'
                                                        : result.confidence >= 0.5
                                                            ? 'var(--accent-orange)'
                                                            : 'var(--accent-red)'
                                                }}
                                            >
                                                {(result.confidence * 100).toFixed(0)}%
                                            </span>
                                        ) : isCurrentlyAnalyzing ? (
                                            <span className={styles.analyzing}>...</span>
                                        ) : (
                                            <span className={styles.emptyCell}>-</span>
                                        )}
                                    </td>

                                    {/* Reason */}
                                    <td className={styles.cellReason}>
                                        {result ? (
                                            <div className={styles.reasonText} title={result.reason}>
                                                {result.reason.slice(0, 40)}
                                                {result.reason.length > 40 ? '...' : ''}
                                            </div>
                                        ) : isCurrentlyAnalyzing ? (
                                            <span className={styles.analyzing}>Analyzing...</span>
                                        ) : (
                                            <span className={styles.emptyCell}>Pending</span>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Summary Stats */}
            {turnResults.length > 0 && (
                <div className={styles.summary}>
                    <div className={styles.summaryItem}>
                        <span className={styles.summaryIcon}>✅</span>
                        <span className={styles.summaryValue}>
                            {turnResults.filter(r => !r.is_bad).length}
                        </span>
                        <span className={styles.summaryLabel}>Good</span>
                    </div>
                    <div className={styles.summaryItem}>
                        <span className={styles.summaryIcon}>❌</span>
                        <span className={styles.summaryValue}>
                            {turnResults.filter(r => r.is_bad).length}
                        </span>
                        <span className={styles.summaryLabel}>Issues</span>
                    </div>
                    <div className={styles.summaryItem}>
                        <span className={styles.summaryIcon}>🔁</span>
                        <span className={styles.summaryValue}>
                            {turnResults.filter(r => r.detection_type === 'ccm').length}
                        </span>
                        <span className={styles.summaryLabel}>CCM</span>
                    </div>
                    <div className={styles.summaryItem}>
                        <span className={styles.summaryIcon}>❓</span>
                        <span className={styles.summaryValue}>
                            {turnResults.filter(r => r.detection_type === 'rdm').length}
                        </span>
                        <span className={styles.summaryLabel}>RDM</span>
                    </div>
                    <div className={styles.summaryItem}>
                        <span className={styles.summaryIcon}>👻</span>
                        <span className={styles.summaryValue}>
                            {turnResults.filter(r => r.detection_type === 'hallucination').length}
                        </span>
                        <span className={styles.summaryLabel}>Halluc.</span>
                    </div>
                </div>
            )}
        </div>
    );
}
