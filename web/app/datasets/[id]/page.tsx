'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import styles from './page.module.css';
import Header from '@/components/Header';
import { 
  DatasetWithStats, 
  fetchDataset, 
  deleteDataset,
  evaluateDatasetStream,
  EvalStreamEvent 
} from '@/lib/api';

export default function DatasetDetail() {
  const params = useParams();
  const router = useRouter();
  const id = Number(params.id);
  
  const [dataset, setDataset] = useState<DatasetWithStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [evalProgress, setEvalProgress] = useState<{ current: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedConvs, setExpandedConvs] = useState<Set<number>>(new Set());

  const loadDataset = async () => {
    try {
      const data = await fetchDataset(id);
      setDataset(data);
      setError(null);
    } catch (err) {
      setError('Failed to load dataset');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) loadDataset();
  }, [id]);

  const handleEvaluate = async () => {
    setEvaluating(true);
    setEvalProgress(null);
    
    const cleanup = evaluateDatasetStream(
      id,
      (event: EvalStreamEvent) => {
        if (event.type === 'start') {
          setEvalProgress({ current: 0, total: event.total });
        } else if (event.type === 'progress') {
          setEvalProgress({ current: event.current, total: event.total });
          
          // Update dataset with new eval results in real-time
          setDataset((prev) => {
            if (!prev) return prev;
            
            const updatedConversations = prev.conversations.map((conv) => {
              if (conv.id === event.conversation_db_id) {
                const updatedMessages = conv.messages.map((msg) => {
                  const result = event.results.find((r) => r.message_id === msg.id);
                  if (result) {
                    return {
                      ...msg,
                      eval_result: {
                        id: 0,
                        message_id: msg.id,
                        is_bad: result.is_bad,
                        detection_type: result.detection_type,
                        confidence: result.confidence,
                        reason: result.reason,
                      },
                    };
                  }
                  return msg;
                });
                return { ...conv, messages: updatedMessages };
              }
              return conv;
            });
            
            return { ...prev, conversations: updatedConversations };
          });
        } else if (event.type === 'complete') {
          // Update stats and mark as evaluated
          setDataset((prev) => {
            if (!prev) return prev;
            return { 
              ...prev, 
              evaluated: true, 
              stats: event.stats || prev.stats 
            };
          });
          setEvaluating(false);
          setEvalProgress(null);
        } else if (event.type === 'error') {
          alert(`Evaluation failed: ${event.message}`);
          setEvaluating(false);
          setEvalProgress(null);
        }
      },
      (error) => {
        alert('Evaluation failed. Check API server.');
        setEvaluating(false);
        setEvalProgress(null);
      }
    );
    
    // Cleanup will be called when evaluation completes or errors
  };

  const handleDelete = async () => {
    if (!confirm('Delete this dataset and all its data?')) return;
    try {
      await deleteDataset(id);
      router.push('/');
    } catch (err) {
      alert('Failed to delete');
    }
  };

  const toggleConversation = (convId: number) => {
    setExpandedConvs((prev) => {
      const next = new Set(prev);
      if (next.has(convId)) {
        next.delete(convId);
      } else {
        next.add(convId);
      }
      return next;
    });
  };

  const expandAll = () => {
    if (!dataset) return;
    setExpandedConvs(new Set(dataset.conversations.map((c) => c.id)));
  };

  const collapseAll = () => {
    setExpandedConvs(new Set());
  };

  if (loading) {
    return (
      <>
        <Header />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.loadingState}>
              <div className={styles.spinner} />
              <span>Loading dataset...</span>
            </div>
          </div>
        </main>
      </>
    );
  }

  if (error || !dataset) {
    return (
      <>
        <Header />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.errorState}>
              <span>⚠️</span>
              <h2>Dataset not found</h2>
              <Link href="/" className="btn btn-primary">Back to Dashboard</Link>
            </div>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.container}>
          <Link href="/" className={styles.backLink}>
            ← Back to Dashboard
          </Link>

          <div className={styles.header}>
            <div className={styles.headerInfo}>
              <h1 className={styles.title}>{dataset.name}</h1>
              <div className={styles.meta}>
                <span className={`badge ${dataset.file_type === 'csv' ? 'badge-info' : 'badge-purple'}`}>
                  {dataset.file_type.toUpperCase()}
                </span>
                <span className={styles.metaText}>
                  {dataset.conversation_count} conversations · {dataset.message_count} messages
                </span>
              </div>
            </div>

            <div className={styles.actions}>
              {!dataset.evaluated ? (
                <button
                  className="btn btn-primary"
                  onClick={handleEvaluate}
                  disabled={evaluating}
                >
                  {evaluating ? (
                    <>
                      <span className={styles.btnSpinner} />
                      {evalProgress 
                        ? `Evaluating ${evalProgress.current}/${evalProgress.total}...`
                        : 'Starting...'
                      }
                    </>
                  ) : (
                    <>⚡ Run Evaluation</>
                  )}
                </button>
              ) : (
                <span className={`badge badge-success ${styles.evalBadge}`}>
                  ✓ Evaluated
                </span>
              )}
              <button className="btn btn-danger" onClick={handleDelete}>
                Delete
              </button>
            </div>
          </div>

          {dataset.stats && (
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <span className={styles.statIcon}>📊</span>
                <div className={styles.statContent}>
                  <span className={styles.statValue}>{dataset.stats.total_responses}</span>
                  <span className={styles.statLabel}>Total Responses</span>
                </div>
              </div>
              <div className={`${styles.statCard} ${styles.statGood}`}>
                <span className={styles.statIcon}>✅</span>
                <div className={styles.statContent}>
                  <span className={styles.statValue}>{dataset.stats.good_responses}</span>
                  <span className={styles.statLabel}>Good Responses</span>
                </div>
              </div>
              <div className={`${styles.statCard} ${styles.statBad}`}>
                <span className={styles.statIcon}>❌</span>
                <div className={styles.statContent}>
                  <span className={styles.statValue}>{dataset.stats.bad_responses}</span>
                  <span className={styles.statLabel}>Bad Responses</span>
                </div>
              </div>
              <div className={styles.statCard}>
                <span className={styles.statIcon}>🎯</span>
                <div className={styles.statContent}>
                  <span className={styles.statValue}>
                    {(dataset.stats.avg_confidence * 100).toFixed(0)}%
                  </span>
                  <span className={styles.statLabel}>Avg Confidence</span>
                </div>
              </div>
            </div>
          )}

          {dataset.stats && (
            <div className={styles.detectionBreakdown}>
              <h3>Detection Methods</h3>
              <div className={styles.detectionBars}>
                <div className={styles.detectionItem}>
                  <div className={styles.detectionHeader}>
                    <span className="badge badge-info">CCM</span>
                    <span>{dataset.stats.ccm_detections}</span>
                  </div>
                  <div className={styles.detectionBar}>
                    <div
                      className={styles.detectionFill}
                      style={{
                        width: `${(dataset.stats.ccm_detections / Math.max(dataset.stats.total_responses, 1)) * 100}%`,
                        background: 'var(--accent-cyan)',
                      }}
                    />
                  </div>
                </div>
                <div className={styles.detectionItem}>
                  <div className={styles.detectionHeader}>
                    <span className="badge badge-purple">RDM</span>
                    <span>{dataset.stats.rdm_detections}</span>
                  </div>
                  <div className={styles.detectionBar}>
                    <div
                      className={styles.detectionFill}
                      style={{
                        width: `${(dataset.stats.rdm_detections / Math.max(dataset.stats.total_responses, 1)) * 100}%`,
                        background: 'var(--accent-purple)',
                      }}
                    />
                  </div>
                </div>
                <div className={styles.detectionItem}>
                  <div className={styles.detectionHeader}>
                    <span className="badge badge-warning">LLM Judge</span>
                    <span>{dataset.stats.llm_judge_detections}</span>
                  </div>
                  <div className={styles.detectionBar}>
                    <div
                      className={styles.detectionFill}
                      style={{
                        width: `${(dataset.stats.llm_judge_detections / Math.max(dataset.stats.total_responses, 1)) * 100}%`,
                        background: 'var(--accent-orange)',
                      }}
                    />
                  </div>
                </div>
                <div className={styles.detectionItem}>
                  <div className={styles.detectionHeader}>
                    <span className="badge badge-danger">Hallucination</span>
                    <span>{dataset.stats.hallucination_detections}</span>
                  </div>
                  <div className={styles.detectionBar}>
                    <div
                      className={styles.detectionFill}
                      style={{
                        width: `${(dataset.stats.hallucination_detections / Math.max(dataset.stats.total_responses, 1)) * 100}%`,
                        background: 'var(--accent-red)',
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className={styles.conversationsSection}>
            <div className={styles.sectionHeader}>
              <h2>Conversations</h2>
              <div className={styles.expandControls}>
                <button className={styles.expandBtn} onClick={expandAll}>Expand All</button>
                <button className={styles.expandBtn} onClick={collapseAll}>Collapse All</button>
              </div>
            </div>

            <div className={styles.conversationsList}>
              {dataset.conversations.map((conv, index) => {
                const isExpanded = expandedConvs.has(conv.id);
                const hasIssues = conv.messages.some((m) => m.eval_result?.is_bad);
                
                return (
                  <div
                    key={conv.id}
                    className={`${styles.conversationCard} ${hasIssues ? styles.hasIssues : ''}`}
                  >
                    <button
                      className={styles.conversationHeader}
                      onClick={() => toggleConversation(conv.id)}
                    >
                      <div className={styles.convInfo}>
                        <span className={styles.convIndex}>#{index + 1}</span>
                        <span className={styles.convId}>{conv.conversation_id}</span>
                        <span className={styles.convMsgCount}>
                          {conv.messages.length} messages
                        </span>
                      </div>
                      <div className={styles.convStatus}>
                        {hasIssues && (
                          <span className="badge badge-danger">Issues Found</span>
                        )}
                        <span className={styles.expandIcon}>{isExpanded ? '▼' : '▶'}</span>
                      </div>
                    </button>

                    {isExpanded && (
                      <div className={styles.messagesContainer}>
                        {conv.messages.map((msg) => (
                          <div
                            key={msg.id}
                            className={`${styles.message} ${styles[msg.role]}`}
                          >
                            <div className={styles.messageHeader}>
                              <span className={styles.messageRole}>
                                {msg.role === 'user' ? '👤 User' : '🤖 Assistant'}
                              </span>
                              {msg.eval_result && (
                                <div className={styles.evalInfo}>
                                  <span className={`badge ${msg.eval_result.is_bad ? 'badge-danger' : 'badge-success'}`}>
                                    {msg.eval_result.is_bad ? '❌ Bad' : '✅ Good'}
                                  </span>
                                  <span className={`badge ${
                                    msg.eval_result.detection_type === 'ccm' ? 'badge-info' :
                                    msg.eval_result.detection_type === 'rdm' ? 'badge-purple' :
                                    msg.eval_result.detection_type === 'hallucination' ? 'badge-danger' :
                                    'badge-warning'
                                  }`}>
                                    {msg.eval_result.detection_type.toUpperCase()}
                                  </span>
                                  <span className={styles.confidence}>
                                    {(msg.eval_result.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              )}
                            </div>
                            <div className={styles.messageContent}>
                              {msg.content}
                            </div>
                            {msg.knowledge && (
                              <div className={styles.knowledgeBox}>
                                <span className={styles.knowledgeLabel}>📚 Knowledge Context</span>
                                <div className={styles.knowledgeContent}>{msg.knowledge}</div>
                              </div>
                            )}
                            {msg.eval_result?.reason && (
                              <div className={styles.evalReason}>
                                💬 {msg.eval_result.reason}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

