'use client';

import { useState } from 'react';
import styles from './AnalysisSelector.module.css';

export type AnalysisType = 
  | 'conversation'    // Original CCM/RDM/Hallucination
  | 'trajectory'      // Agent Trajectory Analysis
  | 'tools'           // Tool Use Quality
  | 'self_correction' // Self-Correction Detection
  | 'full_agent'      // All agent analyses combined
  | 'full_all';       // Everything (conversation + agent)

interface AnalysisOption {
  id: AnalysisType;
  name: string;
  description: string;
  icon: string;
  badge?: string;
  features: string[];
  category: 'conversation' | 'agent' | 'all';
}

const ANALYSIS_OPTIONS: AnalysisOption[] = [
  {
    id: 'full_all',
    name: 'Full Analysis',
    description: 'Run all evaluations (conversation + agent)',
    icon: '🚀',
    badge: 'RECOMMENDED',
    category: 'all',
    features: ['Conversation Detection', 'Trajectory Analysis', 'Tool Use Quality', 'Self-Correction'],
  },
  {
    id: 'conversation',
    name: 'Conversation Detection',
    description: 'Detect bad responses via re-ask patterns',
    icon: '💬',
    category: 'conversation',
    features: ['CCM - Re-ask Detection', 'RDM - Correction Detection', 'Hallucination Check', 'LLM Judge Fallback'],
  },
  {
    id: 'trajectory',
    name: 'Trajectory Analysis',
    description: 'Analyze agent execution paths',
    icon: '🔄',
    badge: 'NEW',
    category: 'agent',
    features: ['Circular Pattern Detection', 'Regression Detection', 'Progress Tracking', 'Efficiency Score'],
  },
  {
    id: 'tools',
    name: 'Tool Use Quality',
    description: 'Evaluate agent tool usage',
    icon: '🔧',
    badge: 'NEW',
    category: 'agent',
    features: ['Tool Selection Accuracy', 'Parameter Validation', 'Chain Efficiency', 'Hallucination Detection'],
  },
  {
    id: 'self_correction',
    name: 'Self-Correction',
    description: 'Track error awareness & recovery',
    icon: '🔁',
    badge: 'NEW',
    category: 'agent',
    features: ['Error Detection', 'Recovery Success', 'Loop Detection', 'Awareness Score'],
  },
];

interface AnalysisSelectorProps {
  selectedAnalysis: AnalysisType;
  onSelect: (type: AnalysisType) => void;
  disabled?: boolean;
}

export default function AnalysisSelector({
  selectedAnalysis,
  onSelect,
  disabled = false,
}: AnalysisSelectorProps) {
  const [showDetails, setShowDetails] = useState<AnalysisType | null>(null);

  // Group options by category
  const conversationOptions = ANALYSIS_OPTIONS.filter(o => o.category === 'conversation');
  const agentOptions = ANALYSIS_OPTIONS.filter(o => o.category === 'agent');
  const allOptions = ANALYSIS_OPTIONS.filter(o => o.category === 'all');

  const renderOption = (option: AnalysisOption) => (
    <button
      key={option.id}
      className={`${styles.option} ${selectedAnalysis === option.id ? styles.selected : ''}`}
      onClick={() => !disabled && onSelect(option.id)}
      onMouseEnter={() => setShowDetails(option.id)}
      onMouseLeave={() => setShowDetails(null)}
      disabled={disabled}
    >
      <div className={styles.optionHeader}>
        <span className={styles.optionIcon}>{option.icon}</span>
        <div className={styles.optionTitleArea}>
          <span className={styles.optionName}>{option.name}</span>
          {option.badge && (
            <span className={`${styles.badge} ${option.badge === 'RECOMMENDED' ? styles.badgeRecommended : styles.badgeNew}`}>
              {option.badge}
            </span>
          )}
        </div>
      </div>
      <p className={styles.optionDescription}>{option.description}</p>
      
      {showDetails === option.id && (
        <div className={styles.features}>
          {option.features.map((feature, i) => (
            <span key={i} className={styles.feature}>
              <span className={styles.featureCheck}>✓</span>
              {feature}
            </span>
          ))}
        </div>
      )}
    </button>
  );

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3 className={styles.title}>Select Analysis Type</h3>
      </div>

      {/* Full Analysis Option */}
      <div className={styles.section}>
        <div className={styles.grid}>
          {allOptions.map(renderOption)}
        </div>
      </div>

      {/* Conversation Analysis */}
      <div className={styles.section}>
        <h4 className={styles.sectionTitle}>
          <span>💬</span> Conversation Detection
        </h4>
        <div className={styles.grid}>
          {conversationOptions.map(renderOption)}
        </div>
      </div>

      {/* Agent Analysis */}
      <div className={styles.section}>
        <h4 className={styles.sectionTitle}>
          <span>🤖</span> Agent Metrics
        </h4>
        <div className={styles.grid}>
          {agentOptions.map(renderOption)}
        </div>
      </div>
    </div>
  );
}

export { ANALYSIS_OPTIONS };
