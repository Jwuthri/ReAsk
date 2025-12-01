'use client';

import Link from 'next/link';
import styles from './DatasetCard.module.css';
import { Dataset } from '@/lib/api';

interface DatasetCardProps {
  dataset: Dataset;
  onDelete: (id: number) => void;
}

export default function DatasetCard({ dataset, onDelete }: DatasetCardProps) {
  const formattedDate = new Date(dataset.uploaded_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className={styles.card}>
      <Link href={`/datasets/${dataset.id}`} className={styles.link}>
        <div className={styles.header}>
          <div className={styles.icon}>
            {dataset.file_type === 'csv' ? '📊' : '📋'}
          </div>
          <div className={styles.titleArea}>
            <h3 className={styles.title}>{dataset.name}</h3>
            <span className={styles.date}>{formattedDate}</span>
          </div>
        </div>

        <div className={styles.stats}>
          <div className={styles.stat}>
            <span className={styles.statValue}>{dataset.conversation_count}</span>
            <span className={styles.statLabel}>Conversations</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statValue}>{dataset.message_count}</span>
            <span className={styles.statLabel}>Messages</span>
          </div>
          <div className={styles.stat}>
            <span className={`${styles.statValue} ${dataset.evaluated ? styles.evaluated : styles.pending}`}>
              {dataset.evaluated ? '✓' : '○'}
            </span>
            <span className={styles.statLabel}>
              {dataset.evaluated ? 'Evaluated' : 'Pending'}
            </span>
          </div>
        </div>

        <div className={styles.footer}>
          <span className={`badge ${dataset.file_type === 'csv' ? 'badge-info' : 'badge-purple'}`}>
            {dataset.file_type.toUpperCase()}
          </span>
        </div>
      </Link>

      <button
        className={styles.deleteBtn}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onDelete(dataset.id);
        }}
        title="Delete dataset"
      >
        ×
      </button>
    </div>
  );
}

