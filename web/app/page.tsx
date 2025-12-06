'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import styles from './page.module.css';
import Header from '@/components/Header';
import { Dataset, fetchDatasets } from '@/lib/api';

export default function Home() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await fetchDatasets();
      setDatasets(data);
    } catch (err) {
      console.error(err);
      setError('Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'var(--accent-green, #10b981)';
    if (score >= 0.4) return 'var(--accent-yellow, #f59e0b)';
    return 'var(--accent-red, #ef4444)';
  };

  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.container}>
          <section className={styles.hero}>
            <h1 className={styles.title}>
              <span className={styles.titleGradient}>Evaluate</span> Your LLM Conversations
            </h1>
            <p className={styles.subtitle}>
              Upload conversation datasets and detect bad responses through re-ask pattern analysis.
              Powered by CCM, RDM, and LLM Judge detection methods.
            </p>
          </section>

          {loading ? (
            <div className={styles.grid}>
              {[1, 2, 3].map((i) => (
                <div key={i} className={styles.skeletonCard}>
                  <div className={styles.skeletonHeader}>
                    <div className={styles.skeletonIcon} style={{ background: 'var(--bg-tertiary)' }} />
                    <div className={styles.skeletonTitleArea}>
                      <div className={styles.skeletonTitle} style={{ background: 'var(--bg-tertiary)' }} />
                      <div className={styles.skeletonDate} style={{ background: 'var(--bg-tertiary)' }} />
                    </div>
                  </div>
                  <div className={styles.skeletonStats}>
                    <div className={styles.skeletonStat} style={{ background: 'var(--bg-tertiary)' }} />
                    <div className={styles.skeletonStat} style={{ background: 'var(--bg-tertiary)' }} />
                  </div>
                </div>
              ))}
            </div>
          ) : error ? (
            <div className={styles.error}>
              <span className={styles.errorIcon}>⚠️</span>
              <span>{error}</span>
              <button className="btn btn-primary" onClick={loadDatasets}>Retry</button>
            </div>
          ) : datasets.length === 0 ? (
            <div className={styles.empty}>
              <span className={styles.emptyIcon}>📂</span>
              <h3>No datasets yet</h3>
              <p>Upload a dataset to get started</p>
            </div>
          ) : (
            <div className={styles.grid}>
              {datasets.map((dataset) => (
                <Link key={dataset.id} href={`/datasets/${dataset.id}`} className={styles.card}>
                  <div className={styles.cardHeader}>
                    <div className={styles.cardIcon}>
                      {dataset.file_type === 'csv' ? '📊' : '📄'}
                    </div>
                    {dataset.evaluated && (
                      <span className={styles.badge}>Evaluated</span>
                    )}
                  </div>

                  <h3 className={styles.cardTitle}>{dataset.name}</h3>
                  <div className={styles.cardMeta}>
                    <span>{new Date(dataset.uploaded_at).toLocaleDateString()}</span>
                    <span>•</span>
                    <span>{dataset.conversation_count} convs</span>
                  </div>

                  {dataset.overall_score !== undefined && dataset.overall_score !== null && (
                    <div className={styles.scoreSection}>
                      <div className={styles.scoreHeader}>
                        <span className={styles.scoreLabel}>Overall Score</span>
                        <span className={styles.scoreValue} style={{ color: getScoreColor(dataset.overall_score) }}>
                          {(dataset.overall_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className={styles.scoreBar}>
                        <div
                          className={styles.scoreFill}
                          style={{
                            width: `${dataset.overall_score * 100}%`,
                            background: getScoreColor(dataset.overall_score)
                          }}
                        />
                      </div>

                      {dataset.agent_scores && Object.keys(dataset.agent_scores).length > 0 && (
                        <div className={styles.agentScores}>
                          {Object.entries(dataset.agent_scores).slice(0, 3).map(([agentId, scoreData]: [string, any]) => (
                            <div key={agentId} className={styles.agentScoreItem}>
                              <span className={styles.agentName}>
                                🤖 {agentId}
                              </span>
                              <span className={styles.agentScoreValue} style={{ color: getScoreColor(scoreData.overall) }}>
                                {(scoreData.overall * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                          {Object.keys(dataset.agent_scores).length > 3 && (
                            <div className={styles.agentScoreItem}>
                              <span className={styles.agentName} style={{ fontSize: '11px' }}>
                                +{Object.keys(dataset.agent_scores).length - 3} more agents
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
