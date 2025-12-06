'use client';

import { useState, useMemo } from 'react';
import styles from './AgentTree.module.css';
import {
  AgentTraceInput,
  AgentTurn,
  AgentAnalysisResults,
  PerAgentScore,
  AgentDef,
  ConversationStepResult,
} from '@/lib/api';

export type TreeSelection = 
  | { type: 'global' }
  | { type: 'agent'; agentId: string }
  | { type: 'turn'; turnIndex: number; agentId?: string };

interface AgentTreeProps {
  trace: AgentTraceInput;
  results?: AgentAnalysisResults;
  selection: TreeSelection;
  onSelect: (selection: TreeSelection) => void;
}

interface AgentTurnInfo {
  turnIndex: number;
  agentId: string;
  userMessage?: string;
  hasResponse: boolean;
  score?: number;
  isBad?: boolean;
}

export default function AgentTree({ trace, results, selection, onSelect }: AgentTreeProps) {
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set(['global']));

  const toggleAgent = (agentId: string) => {
    setExpandedAgents(prev => {
      const next = new Set(prev);
      if (next.has(agentId)) {
        next.delete(agentId);
      } else {
        next.add(agentId);
      }
      return next;
    });
  };

  // Get agents - from trace or infer from turns
  const agents = useMemo((): AgentDef[] => {
    if (trace.agents && trace.agents.length > 0) {
      return trace.agents;
    }
    // Infer from turns
    const agentIds = new Set<string>();
    for (const turn of trace.turns || []) {
      if (turn.agent_interactions) {
        for (const interaction of turn.agent_interactions) {
          agentIds.add(interaction.agent_id);
        }
      } else {
        agentIds.add('agent');
      }
    }
    return Array.from(agentIds).map(id => ({ id, name: id === 'agent' ? 'Agent' : id }));
  }, [trace]);

  // Build turn info per agent
  const agentTurns = useMemo((): Map<string, AgentTurnInfo[]> => {
    const map = new Map<string, AgentTurnInfo[]>();
    
    for (const agent of agents) {
      map.set(agent.id, []);
    }

    for (let turnIdx = 0; turnIdx < (trace.turns?.length || 0); turnIdx++) {
      const turn = trace.turns![turnIdx];
      const turnResult = results?.conversation?.results?.find(r => r.step_index === turnIdx);
      
      if (turn.agent_interactions) {
        // Multi-agent format
        for (const interaction of turn.agent_interactions) {
          const agentTurnList = map.get(interaction.agent_id) || [];
          agentTurnList.push({
            turnIndex: turnIdx,
            agentId: interaction.agent_id,
            userMessage: turn.user_message,
            hasResponse: !!interaction.agent_response,
            score: turnResult?.confidence,
            isBad: turnResult?.is_bad,
          });
          map.set(interaction.agent_id, agentTurnList);
        }
      } else {
        // Simple format - assign to default agent
        const agentTurnList = map.get('agent') || [];
        agentTurnList.push({
          turnIndex: turnIdx,
          agentId: 'agent',
          userMessage: turn.user_message,
          hasResponse: !!turn.agent_response,
          score: turnResult?.confidence,
          isBad: turnResult?.is_bad,
        });
        map.set('agent', agentTurnList);
      }
    }

    return map;
  }, [trace, results, agents]);

  const getScoreColor = (score?: number, isBad?: boolean) => {
    if (isBad) return 'var(--accent-red)';
    if (score === undefined) return 'var(--text-muted)';
    if (score >= 0.8) return 'var(--accent-green)';
    if (score >= 0.5) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  const getAgentScoreColor = (score?: number) => {
    if (score === undefined) return 'var(--text-muted)';
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  const isSelected = (sel: TreeSelection) => {
    if (selection.type !== sel.type) return false;
    if (sel.type === 'global') return true;
    if (sel.type === 'agent') return (selection as any).agentId === sel.agentId;
    if (sel.type === 'turn') {
      return (selection as any).turnIndex === sel.turnIndex && 
             (selection as any).agentId === sel.agentId;
    }
    return false;
  };

  return (
    <div className={styles.tree}>
      {/* Global Node */}
      <div
        className={`${styles.node} ${styles.globalNode} ${isSelected({ type: 'global' }) ? styles.selected : ''}`}
        onClick={() => onSelect({ type: 'global' })}
      >
        <div className={styles.nodeContent}>
          <span className={styles.nodeIcon}>📊</span>
          <span className={styles.nodeName}>Overview</span>
        </div>
        {results && (
          <span 
            className={styles.scoreBadge}
            style={{ background: getAgentScoreColor(results.overall_score) }}
          >
            {(results.overall_score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Agent Nodes */}
      <div className={styles.agentList}>
        {agents.map(agent => {
          const agentScore = results?.per_agent_scores?.[agent.id];
          const turns = agentTurns.get(agent.id) || [];
          const isExpanded = expandedAgents.has(agent.id);

          return (
            <div key={agent.id} className={styles.agentSection}>
              {/* Agent Header */}
              <div
                className={`${styles.node} ${styles.agentNode} ${isSelected({ type: 'agent', agentId: agent.id }) ? styles.selected : ''}`}
              >
                <button 
                  className={styles.expandBtn}
                  onClick={(e) => { e.stopPropagation(); toggleAgent(agent.id); }}
                >
                  {isExpanded ? '▼' : '▶'}
                </button>
                <div 
                  className={styles.nodeContent}
                  onClick={() => onSelect({ type: 'agent', agentId: agent.id })}
                >
                  <span className={styles.nodeIcon}>🤖</span>
                  <span className={styles.nodeName}>{agent.name || agent.id}</span>
                  {agent.role && (
                    <span className={styles.roleTag}>{agent.role}</span>
                  )}
                </div>
                {agentScore && (
                  <span 
                    className={styles.scoreBadge}
                    style={{ background: getAgentScoreColor(agentScore.overall) }}
                  >
                    {(agentScore.overall * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Turn Nodes (when expanded) */}
              {isExpanded && (
                <div className={styles.turnList}>
                  {turns.map((turnInfo, idx) => (
                    <div
                      key={`${agent.id}-${turnInfo.turnIndex}`}
                      className={`${styles.node} ${styles.turnNode} ${isSelected({ type: 'turn', turnIndex: turnInfo.turnIndex, agentId: agent.id }) ? styles.selected : ''}`}
                      onClick={() => onSelect({ type: 'turn', turnIndex: turnInfo.turnIndex, agentId: agent.id })}
                    >
                      <div className={styles.turnConnector}>
                        <span className={styles.turnLine} />
                      </div>
                      <div className={styles.nodeContent}>
                        <span className={styles.turnIndex}>Turn {turnInfo.turnIndex + 1}</span>
                        {turnInfo.userMessage && (
                          <span className={styles.turnPreview}>
                            {turnInfo.userMessage.slice(0, 30)}{turnInfo.userMessage.length > 30 ? '...' : ''}
                          </span>
                        )}
                      </div>
                      {turnInfo.score !== undefined && (
                        <span 
                          className={styles.turnScoreDot}
                          style={{ background: getScoreColor(turnInfo.score, turnInfo.isBad) }}
                          title={`${(turnInfo.score * 100).toFixed(0)}%`}
                        />
                      )}
                      {turnInfo.isBad && (
                        <span className={styles.badIndicator}>!</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Quick Stats at Bottom */}
      {results && (
        <div className={styles.quickStats}>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>Turns</span>
            <span className={styles.statValue}>{trace.turns?.length || 0}</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>Agents</span>
            <span className={styles.statValue}>{agents.length}</span>
          </div>
          {results.conversation && (
            <div className={styles.statItem}>
              <span className={styles.statLabel}>Issues</span>
              <span 
                className={styles.statValue}
                style={{ color: results.conversation.bad_responses > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}
              >
                {results.conversation.bad_responses}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

