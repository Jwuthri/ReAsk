import React from 'react';
import { AgentSessionInput } from '@/lib/api';
import styles from './DatasetSummaryTable.module.css';

export interface DatasetInput {
    name?: string;
    task?: string;
    conversations: AgentSessionInput[];
    total_cost?: number;
}

interface DatasetSummaryTableProps {
    dataset: DatasetInput;
    analysisResults?: any;
    onSelectConversation: (index: number) => void;
}

export default function DatasetSummaryTable({ dataset, analysisResults, onSelectConversation }: DatasetSummaryTableProps) {

    const getScoreColor = (score: number | null) => {
        if (score === null || score === undefined) return 'var(--text-secondary)';
        if (score >= 0.7) return 'var(--accent-green)';
        if (score >= 0.4) return 'var(--accent-orange)';
        return 'var(--accent-red)';
    };

    const handleDownload = () => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
            dataset: dataset,
            results: analysisResults
        }, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${dataset.name || "dataset"}_analysis.json`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    };

    // Extract all unique agent IDs across the dataset to create columns
    const allAgentIds = Array.from(new Set(
        dataset.conversations.flatMap(conv =>
            conv.agents?.map(a => a.id) ||
            conv.turns?.flatMap(t => t.agent_interactions?.map(ai => ai.agent_id)) ||
            []
        ).filter(Boolean)
    ));

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <div>
                    <h2 className={styles.title}>{dataset.name || "Dataset Analysis"}</h2>
                    <p className={styles.subtitle}>
                        {dataset.conversations.length} conversations • {dataset.task || "No task description"}
                    </p>
                </div>
                <div className={styles.actions}>
                    <button className={styles.downloadBtn} onClick={handleDownload}>
                        ⬇️ Download Results
                    </button>
                </div>
            </div>

            <div className={styles.tableContainer}>
                <table className={styles.table}>
                    <thead>
                        <tr>
                            <th>Conversation / Task</th>
                            <th>Status</th>
                            <th>Overall Score</th>
                            {allAgentIds.map(agentId => (
                                <th key={agentId}>Agent: {agentId}</th>
                            ))}
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {dataset.conversations.map((conv, idx) => {
                            const result = analysisResults?.find((r: any) => r.conversation_index === idx);
                            const isBad = result?.is_bad;
                            const score = result?.score; // Assuming result has overall score

                            return (
                                <tr key={idx}>
                                    <td className={styles.taskCell}>
                                        <div className={styles.taskText}>
                                            {conv.initial_task || conv.turns?.[0]?.user_message || `Conversation ${idx + 1}`}
                                        </div>
                                        <div className={styles.taskMeta}>
                                            ID: {idx + 1} • {conv.turns?.length || 0} turns
                                        </div>
                                    </td>
                                    <td>
                                        {result ? (
                                            <span className={`${styles.statusBadge} ${isBad ? styles.statusIssue : styles.statusOk}`}>
                                                {isBad ? 'Issue Detected' : 'Passed'}
                                            </span>
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>Pending...</span>
                                        )}
                                    </td>
                                    <td className={styles.scoreCell}>
                                        <span style={{ color: getScoreColor(score) }}>
                                            {score !== undefined ? `${(score * 100).toFixed(0)}%` : '-'}
                                        </span>
                                    </td>
                                    {allAgentIds.map(agentId => {
                                        // Try to find agent specific score if available in result
                                        // This depends on the result structure. Assuming result.agent_scores = { agentId: score }
                                        const agentScore = result?.agent_scores?.[agentId];
                                        return (
                                            <td key={agentId} className={styles.scoreCell}>
                                                <span style={{ color: getScoreColor(agentScore) }}>
                                                    {agentScore !== undefined ? `${(agentScore * 100).toFixed(0)}%` : '-'}
                                                </span>
                                            </td>
                                        );
                                    })}
                                    <td>
                                        <button
                                            className={styles.viewBtn}
                                            onClick={() => onSelectConversation(idx)}
                                        >
                                            View Details
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
