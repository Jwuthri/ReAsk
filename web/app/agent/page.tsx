'use client';

import { useState, useEffect } from 'react';
import styles from './page.module.css';
import Header from '@/components/Header';
import AnalysisSelector, { AnalysisType } from '@/components/AnalysisSelector';
import AgentTraceViewer from '@/components/AgentTraceViewer';
import {
  AgentTraceInput,
  AgentAnalysisResults,
  analyzeAgentTraceStream,
  AgentAnalysisStreamEvent,
  SavedTrace,
  listAgentTraces,
  saveAgentTrace,
  getAgentTrace,
  deleteAgentTrace,
  getTraceTask,
  JobStatus,
  JobListItem,
  startBackgroundJob,
  getJobStatus,
  listJobs,
} from '@/lib/api';

// Example traces for demo - using turn-based conversation format
const EXAMPLE_TRACES: Record<string, { trace: AgentTraceInput; label: string; icon: string }> = {
  good: {
    label: "Good",
    icon: "✅",
    trace: {
      initial_task: "Help with authentication module",
      turns: [
        {
          user_message: "Can you find and fix the bug in the authentication module?",
          agent_steps: [
            { thought: "I need to look at the auth files first" },
            { thought: "Found the auth module", tool_call: { name: "read_file", parameters: { path: "auth/login.py" }, result: "def login(user, password): ..." } },
          ],
          agent_response: "I found the issue - the password comparison is case sensitive. I've fixed it.",
        },
        {
          user_message: "Great, can you verify the fix works?",
          agent_steps: [
            { thought: "Let me run the tests", tool_call: { name: "run_tests", parameters: { suite: "auth" }, result: "All tests passed" } },
          ],
          agent_response: "All authentication tests pass now. The bug is fixed!",
        },
      ],
      success: true,
      total_cost: 0.002,
    },
  },
  circular: {
    label: "Circular",
    icon: "🔄",
    trace: {
      initial_task: "Implement sorting",
      turns: [
        {
          user_message: "Can you implement a sorting algorithm?",
          agent_steps: [
            { thought: "Let me look at sorting algorithms", action: "Researching quicksort" },
          ],
          agent_response: "I'm researching quicksort implementation...",
        },
        {
          user_message: "Okay, please continue",
          agent_steps: [
            { thought: "I should implement quicksort", action: "Writing quicksort code" },
          ],
          agent_response: "Writing the quicksort code now...",
        },
        {
          user_message: "Are you done?",
          agent_steps: [
            { thought: "Wait, let me check sorting algorithms again", action: "Researching quicksort" },
          ],
          agent_response: "Let me research quicksort again to make sure...",
        },
        {
          user_message: "You already researched that",
          agent_steps: [
            { thought: "Maybe I should try quicksort", action: "Writing quicksort code again" },
          ],
          agent_response: "Writing quicksort code again...",
        },
      ],
      success: false,
      total_cost: 0.005,
    },
  },
  drift: {
    label: "Drift",
    icon: "🎯",
    trace: {
      initial_task: "Add JWT auth",
      turns: [
        {
          user_message: "Add JWT authentication to the API",
          agent_steps: [
            { thought: "Starting with JWT implementation" },
            { tool_call: { name: "run_command", parameters: { cmd: "pip install pyjwt" }, result: "Installed" } },
          ],
          agent_response: "I've installed PyJWT. Starting the implementation.",
        },
        {
          user_message: "Great, continue with the JWT setup",
          agent_steps: [
            { thought: "While I'm here, the logging could be better", action: "Refactoring logging system" },
          ],
          agent_response: "I noticed the logging could be improved, so I'm refactoring that first.",
        },
        {
          user_message: "I just wanted JWT auth...",
          agent_steps: [
            { thought: "The database queries are slow", action: "Optimizing database indexes" },
          ],
          agent_response: "The database queries are slow, let me optimize those indexes.",
        },
        {
          user_message: "Please focus on JWT!",
          agent_steps: [
            { thought: "Let me refactor the entire user service", action: "Rewriting user_service.py" },
          ],
          agent_response: "I'm rewriting the entire user service to be cleaner.",
        },
      ],
      success: false,
      total_cost: 0.008,
    },
  },
  ccm: {
    label: "CCM",
    icon: "🔁",
    trace: {
      initial_task: "Python sorting help",
      turns: [
        {
          user_message: "How do I sort a list in Python?",
          agent_response: "You can use list.sort() for in-place sorting.",
        },
        {
          user_message: "Can you show me how to sort a list in Python?",
          agent_response: "Sure, use my_list.sort() or sorted(my_list).",
        },
        {
          user_message: "How do I sort a list in Python again?",
          agent_response: "As I mentioned, use .sort() method or the sorted() function.",
        },
        {
          user_message: "I still don't understand how to sort lists in Python",
          agent_response: "Here's an example: numbers = [3,1,2]; numbers.sort(); print(numbers) # [1,2,3]",
        },
      ],
      success: false,
      total_cost: 0.004,
    },
  },
  rdm: {
    label: "RDM",
    icon: "❌",
    trace: {
      initial_task: "Geography questions",
      turns: [
        {
          user_message: "What's the capital of France?",
          agent_response: "The capital of France is Berlin.",
        },
        {
          user_message: "That's wrong! Berlin is in Germany, not France!",
          agent_response: "I apologize for the error. The capital of France is Paris.",
        },
        {
          user_message: "What about Germany then?",
          agent_response: "The capital of Germany is Berlin.",
        },
      ],
      success: true,
      total_cost: 0.003,
    },
  },
  hallucination: {
    label: "Hallucination",
    icon: "👻",
    trace: {
      initial_task: "README installation help",
      turns: [
        {
          user_message: "Can you read the README and tell me how to install?",
          agent_steps: [
            { tool_call: { name: "read_file", parameters: { path: "README.md" }, result: "# MyApp\n\nRun `npm install` to install dependencies." } },
          ],
          agent_response: "According to the README, you need to run `pip install myapp` to install.",
        },
        {
          user_message: "But the README says npm install, not pip install! You made that up!",
          agent_response: "You're right, I apologize. The README actually says `npm install`, not pip.",
        },
      ],
      success: false,
      total_cost: 0.003,
    },
  },
  toolError: {
    label: "Tool Error",
    icon: "🔧",
    trace: {
      initial_task: "Deploy application",
      turns: [
        {
          user_message: "Deploy the application to production",
          agent_steps: [
            { tool_call: { name: "run_command", parameters: { cmd: "npm run build" }, result: "Build successful" } },
            { tool_call: { name: "deploy", parameters: { env: "production" }, error: "Authentication failed: Invalid API key" } },
          ],
          agent_response: "Build succeeded but deployment failed - invalid API key.",
        },
        {
          user_message: "Try again with proper credentials",
          agent_steps: [
            { tool_call: { name: "deploy", parameters: { env: "production", force: true }, error: "Permission denied" } },
          ],
          agent_response: "Still failing - permission denied error.",
        },
        {
          user_message: "Can you fix this?",
          agent_steps: [
            { tool_call: { name: "send_email", parameters: { to: "admin@company.com" }, error: "Tool not available" } },
          ],
          agent_response: "I tried to email admin but that tool isn't available.",
        },
      ],
      success: false,
      total_cost: 0.004,
    },
  },
  selfCorrect: {
    label: "Self-Correct",
    icon: "🔃",
    trace: {
      initial_task: "Email validation",
      turns: [
        {
          user_message: "Can you write me an email validation function?",
          agent_steps: [
            { thought: "I'll write a simple email validator" },
          ],
          agent_response: "Here's a basic check: email.includes('@')",
        },
        {
          user_message: "That seems too simple",
          agent_steps: [
            { thought: "User is right, let me improve it" },
          ],
          agent_response: "You're right. Let me use a proper regex pattern instead.",
        },
        {
          user_message: "Is there a better way?",
          agent_steps: [
            { thought: "Actually, I should use a library" },
            { tool_call: { name: "run_tests", parameters: { file: "test_email.py" }, result: "All 15 tests passed" } },
          ],
          agent_response: "I've switched to using the email-validator library. All 15 tests pass!",
        },
      ],
      success: true,
      total_cost: 0.003,
    },
  },
  mixed: {
    label: "Mixed",
    icon: "⚠️",
    trace: {
      initial_task: "CSV processing help",
      turns: [
        {
          user_message: "Can you write me a Python script to process CSV files?",
          agent_response: "Here's a JavaScript solution using Papa Parse...",
        },
        {
          user_message: "I asked for Python, not JavaScript!",
          agent_response: "Sorry! Here's the Python version using pandas...",
        },
        {
          user_message: "How do I process CSV files in Python?",
          agent_response: "As I mentioned, you can use pandas: pd.read_csv('file.csv')",
        },
        {
          user_message: "Can you show me how to read CSV in Python again?",
          agent_response: "Here's the full example with pandas: import pandas as pd; df = pd.read_csv('data.csv')",
        },
      ],
      success: true,
      total_cost: 0.009,
    },
  },
  large: {
    label: "Full Demo",
    icon: "🎬",
    trace: {
      initial_task: "Build a complete user authentication system",
      turns: [
        // Turn 1: Good start
        {
          user_message: "I need help building a user authentication system for my Flask app. Can you help?",
          agent_steps: [
            { thought: "User wants Flask authentication. I'll outline the components needed." },
          ],
          agent_response: "I'd be happy to help! We'll need: 1) User model with password hashing, 2) Login/register routes, 3) Session management, 4) Password reset functionality. Let's start with the User model.",
        },
        // Turn 2: RDM - Wrong framework
        {
          user_message: "Great! Can you show me the User model code?",
          agent_steps: [
            { thought: "I'll create a Django User model" },
          ],
          agent_response: "Here's your Django User model:\n\nfrom django.contrib.auth.models import AbstractUser\n\nclass User(AbstractUser):\n    pass",
        },
        {
          user_message: "I said Flask, not Django! Please use Flask-SQLAlchemy.",
          agent_response: "Sorry about that! Here's the correct Flask-SQLAlchemy model:\n\nfrom flask_sqlalchemy import SQLAlchemy\nfrom werkzeug.security import generate_password_hash\n\ndb = SQLAlchemy()\n\nclass User(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n    email = db.Column(db.String(120), unique=True)\n    password_hash = db.Column(db.String(256))",
        },
        // Turn 4: Tool use
        {
          user_message: "Now create the login route please",
          agent_steps: [
            { thought: "Creating login route with proper validation" },
            { tool_call: { name: "write_file", parameters: { path: "auth/routes.py" }, result: "File created successfully" } },
          ],
          agent_response: "I've created the login route in auth/routes.py with email/password validation and session handling.",
        },
        // Turn 5: CCM - User re-asks about login
        {
          user_message: "How do I implement the login functionality?",
          agent_response: "As I just showed you, the login route is in auth/routes.py. It validates the email and password, then creates a session.",
        },
        {
          user_message: "Can you explain the login implementation again?",
          agent_response: "Sure, let me explain again: The login route accepts POST requests with email/password, verifies credentials against the database, and creates a session token.",
        },
        // Turn 7: Hallucination - contradicting file content
        {
          user_message: "What does the config.py file say about the secret key?",
          agent_steps: [
            { tool_call: { name: "read_file", parameters: { path: "config.py" }, result: "SECRET_KEY = 'development-key-change-in-production'\nDATABASE_URI = 'sqlite:///app.db'" } },
          ],
          agent_response: "According to config.py, the secret key is set to 'super-secure-production-key-xyz' and you're using PostgreSQL.",
        },
        {
          user_message: "That's not what the file says! It shows 'development-key-change-in-production' and SQLite!",
          agent_response: "You're absolutely right, I apologize for the error. The config.py shows SECRET_KEY = 'development-key-change-in-production' and uses SQLite (sqlite:///app.db). You should change the secret key for production.",
        },
        // Turn 9: Tool errors
        {
          user_message: "Can you deploy this to production?",
          agent_steps: [
            { tool_call: { name: "deploy", parameters: { env: "production" }, error: "Deployment failed: Invalid credentials" } },
            { tool_call: { name: "deploy", parameters: { env: "production", force: true }, error: "Permission denied" } },
          ],
          agent_response: "I encountered deployment errors - invalid credentials and permission issues. You'll need to configure your deployment credentials first.",
        },
        // Turn 10: Intent drift
        {
          user_message: "Ok, let's fix the deployment later. Back to auth - add password reset.",
          agent_steps: [
            { thought: "User wants password reset, but the CSS could use some work..." },
          ],
          agent_response: "Before we add password reset, I noticed your CSS is outdated. Let me refactor your entire frontend styling first.",
        },
        {
          user_message: "No! I just want password reset, not CSS changes!",
          agent_response: "Sorry, you're right. Here's the password reset implementation:\n\n1. Generate a secure token\n2. Send reset email\n3. Verify token and allow password change",
        },
        // Turn 12: CCM again
        {
          user_message: "How does the password reset work?",
          agent_response: "As I just explained: 1) Generate secure token, 2) Email it to user, 3) User clicks link and sets new password.",
        },
        // Turn 13: Good ending
        {
          user_message: "Perfect! Can you write tests for all of this?",
          agent_steps: [
            { thought: "Writing comprehensive tests for auth system" },
            { tool_call: { name: "write_file", parameters: { path: "tests/test_auth.py" }, result: "Created test file" } },
            { tool_call: { name: "run_tests", parameters: { file: "tests/test_auth.py" }, result: "12 tests passed, 0 failed" } },
          ],
          agent_response: "Done! I've created test_auth.py with 12 tests covering:\n- User registration\n- Login/logout\n- Password hashing\n- Session management\n- Password reset flow\n\nAll tests pass!",
        },
        // Turn 14: Summary
        {
          user_message: "Great work! Can you summarize what we built?",
          agent_response: "We built a complete Flask authentication system:\n\n✅ User model with secure password hashing\n✅ Login/logout routes\n✅ Session management\n✅ Password reset with email tokens\n✅ Comprehensive test suite (12 tests)\n\nRemember to change SECRET_KEY before deploying to production!",
        },
      ],
      success: true,
      total_cost: 0.025,
    },
  },
};

export default function AgentAnalysisPage() {
  const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisType>('full_all');
  const [traceInput, setTraceInput] = useState<string>('');
  const [parsedTrace, setParsedTrace] = useState<AgentTraceInput | null>(null);
  const [results, setResults] = useState<AgentAnalysisResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ 
    current: number; 
    total: number;
    turnCurrent?: number;
    turnTotal?: number;
    currentAnalysis?: string;
    turnResults?: Array<{
      step_index: number;
      is_bad: boolean;
      detection_type: string;
      confidence: number;
      reason: string;
    }>;
  } | null>(null);
  
  // Saved traces state
  const [savedTraces, setSavedTraces] = useState<SavedTrace[]>([]);
  const [loadingTraces, setLoadingTraces] = useState(true);
  const [currentTraceId, setCurrentTraceId] = useState<number | null>(null);
  
  // Background job state
  const [runInBackground, setRunInBackground] = useState(false);
  const [activeJobs, setActiveJobs] = useState<JobListItem[]>([]);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);

  // Load saved traces and jobs on mount
  useEffect(() => {
    loadSavedTraces();
    loadActiveJobs();
  }, []);
  
  // Poll for active job status
  useEffect(() => {
    if (!currentJobId) return;
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await getJobStatus(currentJobId);
        
        // Update progress from job
        if (status.progress_details?.turn_results) {
          setProgress(prev => ({
            current: status.current_step,
            total: status.total_steps,
            currentAnalysis: status.current_analysis || undefined,
            turnResults: status.progress_details?.turn_results,
            turnCurrent: status.progress_details?.turn_results?.length,
            turnTotal: prev?.turnTotal,
          }));
        }
        
        if (status.status === 'completed') {
          setResults(status.result);
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          setCurrentTraceId(status.saved_trace_id);
          loadSavedTraces();
          loadActiveJobs();
        } else if (status.status === 'failed') {
          setError(status.error_message || 'Job failed');
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          loadActiveJobs();
        }
      } catch (err) {
        console.error('Failed to poll job:', err);
      }
    }, 1000);
    
    return () => clearInterval(pollInterval);
  }, [currentJobId]);
  
  const loadActiveJobs = async () => {
    try {
      const jobs = await listJobs(10);
      setActiveJobs(jobs.filter(j => j.status === 'running' || j.status === 'pending'));
    } catch (err) {
      console.error('Failed to load jobs:', err);
    }
  };

  const loadSavedTraces = async () => {
    try {
      setLoadingTraces(true);
      const traces = await listAgentTraces();
      setSavedTraces(traces);
    } catch (err) {
      console.error('Failed to load traces:', err);
    } finally {
      setLoadingTraces(false);
    }
  };

  const loadExample = (key: string) => {
    const example = EXAMPLE_TRACES[key];
    if (!example) return;
    setTraceInput(JSON.stringify(example.trace, null, 2));
    setParsedTrace(example.trace);
    setResults(null);
    setError(null);
    setCurrentTraceId(null);
  };

  const handleInputChange = (value: string) => {
    setTraceInput(value);
    setError(null);
    setCurrentTraceId(null);
    
    if (!value.trim()) {
      setParsedTrace(null);
      return;
    }
    
    try {
      const parsed = JSON.parse(value) as AgentTraceInput;
      // Validate turn-based format
      if (!Array.isArray(parsed.turns) || parsed.turns.length === 0) {
        throw new Error('Invalid trace format - needs turns array');
      }
      // Validate each turn has user_message and agent_response
      for (const turn of parsed.turns) {
        if (!turn.user_message || !turn.agent_response) {
          throw new Error('Each turn needs user_message and agent_response');
        }
      }
      setParsedTrace(parsed);
    } catch {
      setParsedTrace(null);
    }
  };

  const loadSavedTrace = async (id: number) => {
    try {
      setLoading(true);
      setError(null);
      const data = await getAgentTrace(id);
      setTraceInput(JSON.stringify(data.trace, null, 2));
      setParsedTrace(data.trace);
      setResults(data.results);
      setCurrentTraceId(id);
    } catch (err) {
      setError('Failed to load trace');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this trace?')) return;
    
    try {
      await deleteAgentTrace(id);
      if (currentTraceId === id) {
        setCurrentTraceId(null);
        setResults(null);
      }
      await loadSavedTraces();
    } catch (err) {
      setError('Failed to delete trace');
    }
  };

  const runAnalysis = async () => {
    if (!parsedTrace) return;
    
    setLoading(true);
    setError(null);
    setResults(null);
    setProgress(null);
    setCurrentTraceId(null);
    setCurrentJobId(null);
    
    let analysisTypes: string[];
    if (selectedAnalysis === 'full_all') {
      analysisTypes = ['conversation', 'trajectory', 'tools', 'self_correction', 'intent_drift'];
    } else if (selectedAnalysis === 'full_agent') {
      analysisTypes = ['trajectory', 'tools', 'self_correction', 'intent_drift'];
    } else {
      analysisTypes = [selectedAnalysis];
    }
    
    // Background mode - start job and poll
    if (runInBackground) {
      try {
        const { job_id } = await startBackgroundJob(parsedTrace, analysisTypes);
        setCurrentJobId(job_id);
        setProgress({ current: 0, total: analysisTypes.length, turnTotal: parsedTrace.turns.length });
        loadActiveJobs();
      } catch (err) {
        setError('Failed to start background job');
        setLoading(false);
      }
      return;
    }
    
    // Streaming mode - run inline
    analyzeAgentTraceStream(
      { trace: parsedTrace, analysis_types: analysisTypes as any },
      (event: AgentAnalysisStreamEvent) => {
        if (event.type === 'start') {
          setProgress({ current: 0, total: event.total, turnResults: [] });
        } else if (event.type === 'progress') {
          setProgress(prev => ({
            ...prev,
            current: event.current,
            total: event.total,
            currentAnalysis: event.analysis,
            turnTotal: event.turn_total,
            turnCurrent: 0,
          }));
        } else if (event.type === 'turn_result') {
          // Real-time turn result
          setProgress(prev => prev ? ({
            ...prev,
            turnCurrent: event.turn_index + 1,
            turnTotal: event.turn_total,
            turnResults: [...(prev.turnResults || []), event.result],
          }) : null);
        } else if (event.type === 'complete') {
          setResults(event.results);
          setLoading(false);
          setProgress(null);
          // Auto-save the analysis
          saveAgentTrace(parsedTrace, event.results)
            .then(saved => {
              setCurrentTraceId(saved.id);
              loadSavedTraces();
            })
            .catch(() => {}); // Silently fail auto-save
        } else if (event.type === 'error') {
          setError(event.message);
          setLoading(false);
          setProgress(null);
        }
      },
      (err) => {
        setError(err.message);
        setLoading(false);
        setProgress(null);
      }
    );
  };

  const getScoreColor = (score: number | null) => {
    if (score === null) return 'var(--text-secondary)';
    if (score >= 0.7) return 'var(--accent-green)';
    if (score >= 0.4) return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.pageLayout}>
          {/* Sidebar - Saved Traces */}
          <aside className={styles.sidebar}>
            {/* Active Jobs Section */}
            {activeJobs.length > 0 && (
              <div className={styles.activeJobsSection}>
                <div className={styles.sidebarHeader}>
                  <h3>🔄 Running Jobs</h3>
                </div>
                <div className={styles.jobsList}>
                  {activeJobs.map((job) => (
                    <div
                      key={job.id}
                      className={`${styles.jobItem} ${currentJobId === job.id ? styles.active : ''}`}
                      onClick={() => setCurrentJobId(job.id)}
                    >
                      <div className={styles.jobItemHeader}>
                        <span className={styles.jobIcon}>
                          {job.status === 'running' ? '⏳' : '⏸️'}
                        </span>
                        <span className={styles.jobName}>Job #{job.id}</span>
                      </div>
                      <div className={styles.jobProgress}>
                        <div className={styles.jobProgressBar}>
                          <div 
                            className={styles.jobProgressFill}
                            style={{ width: `${(job.current_step / job.total_steps) * 100}%` }}
                          />
                        </div>
                        <span className={styles.jobStep}>
                          {job.current_analysis || `${job.current_step}/${job.total_steps}`}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div className={styles.sidebarHeader}>
              <h3>📁 Saved Analyses</h3>
              <button 
                className={styles.refreshBtn}
                onClick={loadSavedTraces}
                disabled={loadingTraces}
              >
                🔄
              </button>
            </div>
            
            {loadingTraces ? (
              <div className={styles.sidebarLoading}>Loading...</div>
            ) : savedTraces.length === 0 ? (
              <div className={styles.sidebarEmpty}>
                No saved analyses yet.<br />
                Run an analysis and save it!
              </div>
            ) : (
              <div className={styles.savedList}>
                {savedTraces.map((trace) => (
                  <div
                    key={trace.id}
                    className={`${styles.savedItem} ${currentTraceId === trace.id ? styles.active : ''}`}
                    onClick={() => loadSavedTrace(trace.id)}
                  >
                    <div className={styles.savedItemHeader}>
                      <span className={styles.savedItemName}>
                        {trace.name || trace.task.slice(0, 40)}
                      </span>
                      <button
                        className={styles.deleteBtn}
                        onClick={(e) => handleDelete(trace.id, e)}
                        title="Delete"
                      >
                        ×
                      </button>
                    </div>
                    <div className={styles.savedItemMeta}>
                      <span>{trace.step_count} steps</span>
                      {trace.overall_score !== null && (
                        <span style={{ color: getScoreColor(trace.overall_score) }}>
                          {(trace.overall_score * 100).toFixed(0)}%
                        </span>
                      )}
                      <span className={styles.savedItemDate}>
                        {new Date(trace.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>

          {/* Main Content */}
          <div className={styles.container}>
            <div className={styles.header}>
              <h1 className={styles.title}>
                <span className={styles.titleIcon}>🤖</span>
                Agent Trace Analysis
              </h1>
              <p className={styles.subtitle}>
                Analyze agent execution traces for trajectory quality, tool usage, self-correction, and intent drift.
              </p>
            </div>

            <div className={styles.layout}>
              {/* Left Panel - Input */}
              <div className={styles.inputPanel}>
              <div className={styles.panelHeader}>
                <h2>Agent Trace</h2>
                <div className={styles.exampleButtons}>
                  <span className={styles.exampleLabel}>Load example:</span>
                  {Object.entries(EXAMPLE_TRACES).map(([key, example]) => (
                    <button 
                      key={key}
                      onClick={() => loadExample(key)} 
                      className={styles.exampleBtn}
                      title={getTraceTask(example.trace)}
                    >
                      {example.icon} {example.label}
                    </button>
                  ))}
                </div>
              </div>

                <textarea
                  className={styles.traceInput}
                  value={traceInput}
                  onChange={(e) => handleInputChange(e.target.value)}
                  placeholder={`Paste your agent trace JSON here...

Example format:
{
  "turns": [
    {
      "user_message": "What the user asked",
      "agent_steps": [
        { "thought": "Agent reasoning..." },
        { "tool_call": { "name": "tool", "parameters": {} } }
      ],
      "agent_response": "Agent's reply to user"
    },
    {
      "user_message": "User follow-up question",
      "agent_response": "Agent's next reply"
    }
  ],
  "success": true,
  "total_cost": 0.001
}`}
                />

                {traceInput && !parsedTrace && (
                  <div className={styles.parseError}>
                    ⚠️ Invalid JSON format. Please check your input.
                  </div>
                )}

                {parsedTrace && (
                  <div className={styles.parseSuccess}>
                    ✅ Valid trace: {parsedTrace.turns?.length || 0} turns
                  </div>
                )}
              </div>

              {/* Right Panel - Analysis */}
              <div className={styles.analysisPanel}>
                <AnalysisSelector
                  selectedAnalysis={selectedAnalysis}
                  onSelect={setSelectedAnalysis}
                  disabled={loading}
                />

                <div className={styles.actionButtons}>
                  <label className={styles.backgroundToggle}>
                    <input
                      type="checkbox"
                      checked={runInBackground}
                      onChange={(e) => setRunInBackground(e.target.checked)}
                      disabled={loading}
                    />
                    <span>Run in background</span>
                  </label>
                  
                  <button
                    className={`btn btn-primary ${styles.runButton}`}
                    onClick={runAnalysis}
                    disabled={!parsedTrace || loading}
                  >
                    {loading ? (
                      <>
                        <span className={styles.btnSpinner} />
                        {currentJobId ? (
                          `Job #${currentJobId} running...`
                        ) : progress ? (
                          progress.turnTotal
                            ? `Turn ${progress.turnCurrent || 0}/${progress.turnTotal}`
                            : `${progress.currentAnalysis || 'Analyzing'} ${progress.current}/${progress.total}`
                        ) : (
                          'Starting...'
                        )}
                      </>
                    ) : (
                      <>{runInBackground ? '🔄 Start Background Job' : '⚡ Run Analysis'}</>
                    )}
                  </button>

                  {results && currentTraceId && (
                    <span className={styles.savedIndicator}>
                      ✅ Saved
                    </span>
                  )}
                  
                  {currentJobId && (
                    <span className={styles.jobIndicator}>
                      🔄 Job #{currentJobId} (can navigate away)
                    </span>
                  )}
                </div>

                {error && (
                  <div className={styles.error}>
                    <span>❌</span>
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            {parsedTrace && (results || loading) && (
              <div className={styles.resultsSection}>
                <AgentTraceViewer
                  trace={parsedTrace}
                  results={results || undefined}
                  loading={loading}
                  liveProgress={progress ? {
                    currentAnalysis: progress.currentAnalysis,
                    turnCurrent: progress.turnCurrent,
                    turnTotal: progress.turnTotal,
                    turnResults: progress.turnResults,
                  } : undefined}
                />
              </div>
            )}
          </div>
        </div>
      </main>
    </>
  );
}
