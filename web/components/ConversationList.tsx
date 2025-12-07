import React from 'react';
import { AgentSessionInput } from '@/lib/api';
import styles from './ConversationList.module.css';

export interface DatasetInput {
    name?: string;
    task?: string;
    conversations: AgentSessionInput[];
    total_cost?: number;
    metadata?: Record<string, string>;
}

interface ConversationListProps {
    dataset: DatasetInput;
    selectedIndex: number;
    onSelect: (index: number) => void;
    analysisResults?: any; // To show status icons
}

export default function ConversationList({ dataset, selectedIndex, onSelect, analysisResults }: ConversationListProps) {
    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h2 className={styles.title}>{dataset.name || "Dataset"}</h2>
                <p className={styles.subtitle}>{dataset.conversations.length} conversations</p>
            </div>
            <div className={styles.list}>
                {dataset.conversations.map((conv, idx) => {
                    // Check status if available
                    const result = analysisResults?.find((r: any) => r.conversation_index === idx);
                    const isBad = result?.is_bad;

                    return (
                        <div
                            key={idx}
                            onClick={() => onSelect(idx)}
                            className={`${styles.item} ${selectedIndex === idx ? styles.selected : ''}`}
                        >
                            <div className={styles.itemHeader}>
                                <span className={styles.itemTask}>
                                    {conv.initial_task || `Conversation ${idx + 1}`}
                                </span>
                                {result && (
                                    <span className={`${styles.statusBadge} ${isBad ? styles.statusIssue : styles.statusOk}`}>
                                        {isBad ? 'Issue' : 'OK'}
                                    </span>
                                )}
                            </div>
                            <div className="text-xs text-gray-500 truncate">
                                {conv.turns.length} turns • {(conv.total_cost || 0).toFixed(4)} cost
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
