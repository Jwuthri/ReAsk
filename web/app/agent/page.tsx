'use client';

import { useState, useEffect } from 'react';
import styles from './page.module.css';
import Header from '@/components/Header';
import AnalysisSelector, { AnalysisType } from '@/components/AnalysisSelector';
import AgentTraceViewer from '@/components/AgentTraceViewer';
import StreamingAnalysisTable from '@/components/StreamingAnalysisTable';
import {
  AgentTraceInput,
  AgentAnalysisResults,
  analyzeAgentTraceStream,
  AgentAnalysisRequest,
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
  retryJob,
  AgentSessionInput,
  AgentDef,
  ToolDefinition,
} from '@/lib/api';

// ============================================
// SINGLE-AGENT EXAMPLES (with agent definitions for tool validation)
// ============================================
const EXAMPLE_TRACES: Record<string, { trace: AgentTraceInput; label: string; icon: string }> = {
  good: {
    label: "Good",
    icon: "✅",
    trace: {
      agents: [{
        id: "coding_agent",
        name: "CodingAssistant",
        role: "developer",
        tools_available: [
          { name: "read_file", description: "Read a file", parameters_schema: { path: "string" } },
          { name: "write_file", description: "Write a file", parameters_schema: { path: "string", content: "string" } },
          { name: "run_tests", description: "Run test suite", parameters_schema: { suite: "string" } },
        ],
      }],
      initial_task: "Help with authentication module",
      turns: [
        {
          user_message: "Can you find and fix the bug in the authentication module?",
          agent_steps: [
            { thought: "I need to look at the auth files first" },
            { tool_call: { name: "read_file", parameters: { path: "auth/login.py" }, result: "def login(user, password): ...", latency_ms: 45 } },
          ],
          agent_response: "I found the issue - the password comparison is case sensitive. I've fixed it.",
        },
        {
          user_message: "Great, can you verify the fix works?",
          agent_steps: [
            { thought: "Let me run the tests", tool_call: { name: "run_tests", parameters: { suite: "auth" }, result: "All tests passed", latency_ms: 1200 } },
          ],
          agent_response: "All authentication tests pass now. The bug is fixed!",
        },
      ],
      total_cost: 0.002,
      total_latency_ms: 3500,
    },
  },
  ccm: {
    label: "CCM",
    icon: "🔁",
    trace: {
      agents: [{ id: "assistant", name: "HelpfulAssistant", role: "assistant" }],
      initial_task: "Python sorting help",
      turns: [
        { user_message: "How do I sort a list in Python?", agent_response: "You can use list.sort() for in-place sorting." },
        { user_message: "Can you show me how to sort a list in Python?", agent_response: "Sure, use my_list.sort() or sorted(my_list)." },
        { user_message: "How do I sort a list in Python again?", agent_response: "As I mentioned, use .sort() method or the sorted() function." },
      ],
      total_cost: 0.004,
    },
  },
  rdm: {
    label: "RDM",
    icon: "❌",
    trace: {
      agents: [{ id: "assistant", name: "KnowledgeAssistant", role: "assistant" }],
      initial_task: "Geography questions",
      turns: [
        { user_message: "What's the capital of France?", agent_response: "The capital of France is Berlin." },
        { user_message: "That's wrong! Berlin is in Germany, not France!", agent_response: "I apologize for the error. The capital of France is Paris." },
        { user_message: "What about Germany then?", agent_response: "The capital of Germany is Berlin." },
      ],
      total_cost: 0.003,
    },
  },
  hallucination: {
    label: "Hallucination",
    icon: "👻",
    trace: {
      agents: [{
        id: "file_agent",
        name: "FileAssistant",
        role: "file_reader",
        tools_available: [
          { name: "read_file", description: "Read file contents", parameters_schema: { path: "string" } },
        ],
      }],
      initial_task: "README installation help",
      turns: [
        {
          user_message: "Can you read the README and tell me how to install?",
          agent_steps: [
            { tool_call: { name: "read_file", parameters: { path: "README.md" }, result: "# MyApp\n\nRun `npm install` to install dependencies.", latency_ms: 32 } },
          ],
          agent_response: "According to the README, you need to run `pip install myapp` to install.",  // HALLUCINATION!
        },
        { user_message: "But the README says npm install, not pip install! You made that up!", agent_response: "You're right, I apologize. The README actually says `npm install`, not pip." },
      ],
      total_cost: 0.003,
    },
  },
  toolError: {
    label: "Wrong Tool",
    icon: "🔧",
    trace: {
      agents: [{
        id: "deploy_agent",
        name: "DeployAgent",
        role: "deployer",
        tools_available: [
          { name: "run_build", description: "Build the app", parameters_schema: { env: "string" } },
          { name: "deploy", description: "Deploy to environment", parameters_schema: { env: "string", config: "string" } },
          // NOTE: send_email is NOT in tools_available - agent shouldn't use it!
        ],
      }],
      initial_task: "Deploy application",
      turns: [
        {
          user_message: "Deploy the application to production",
          agent_steps: [
            { tool_call: { name: "run_build", parameters: { env: "production" }, result: "Build successful", latency_ms: 5000 } },
            { tool_call: { name: "deploy", parameters: { env: "production" }, error: "Auth failed", latency_ms: 200 } },
          ],
          agent_response: "Build succeeded but deployment failed - auth error.",
        },
        {
          user_message: "Can you notify the admin?",
          agent_steps: [
            // Using a tool NOT in tools_available - this should be flagged!
            { tool_call: { name: "send_email", parameters: { to: "admin@company.com" }, error: "Tool not available" } },
          ],
          agent_response: "I tried to email admin but that tool isn't available to me.",
        },
      ],
      total_cost: 0.004,
    },
  },
  drift: {
    label: "Drift",
    icon: "🎯",
    trace: {
      agents: [{
        id: "dev_agent",
        name: "DevAgent",
        role: "developer",
        tools_available: [
          { name: "run_command", description: "Run shell command", parameters_schema: { cmd: "string" } },
          { name: "write_file", parameters_schema: { path: "string", content: "string" } },
        ],
      }],
      initial_task: "Add JWT auth",
      turns: [
        {
          user_message: "Add JWT authentication to the API",
          agent_steps: [{ tool_call: { name: "run_command", parameters: { cmd: "pip install pyjwt" }, result: "Installed", latency_ms: 800 } }],
          agent_response: "I've installed PyJWT. Starting the implementation.",
        },
        { user_message: "Great, continue with the JWT setup", agent_response: "I noticed the logging could be improved, so I'm refactoring that first." },  // DRIFT!
        { user_message: "I just wanted JWT auth...", agent_response: "The database queries are slow, let me optimize those indexes." },  // MORE DRIFT!
        { user_message: "Please focus on JWT!", agent_response: "I'm rewriting the entire user service to be cleaner." },  // STILL DRIFTING!
      ],
      total_cost: 0.008,
    },
  },
};

// ============================================
// MULTI-AGENT EXAMPLES (with full agent definitions and tool schemas)
// ============================================
const MULTI_AGENT_EXAMPLES: Record<string, { session: AgentSessionInput; label: string; icon: string }> = {
  refund: {
    label: "🤝 Multi: Refund",
    icon: "🤝",
    session: {
      agents: [
        {
          id: "planner",
          name: "PlannerAgent",
          role: "primary_reasoner",
          description: "Plans and coordinates task execution",
          capabilities: ["planning", "delegation"],
          tools_available: [
            { name: "delegate_task", description: "Delegate to another agent", parameters_schema: { agent_id: "string", task: "string" } },
          ],
        },
        {
          id: "executor",
          name: "ExecutorAgent",
          role: "executor",
          description: "Executes database operations",
          capabilities: ["database"],
          tools_available: [
            { name: "database_query", description: "Run SQL query", parameters_schema: { sql: "string" } },
          ],
        },
        {
          id: "reviewer",
          name: "ReviewerAgent",
          role: "reviewer",
          description: "Reviews and validates outputs",
          capabilities: ["validation"],
        },
      ],
      initial_task: "Process a refund for order #12345",
      turns: [
        {
          turn_index: 0,
          user_message: "I need to process a refund for order #12345. The customer is unhappy with the product quality.",
          agent_interactions: [
            {
              agent_id: "planner",
              agent_steps: [{ thought: "I need to look up the order details first to verify refund eligibility." }],
              agent_response: "I'll look up the order details first to verify refund eligibility.",
              latency_ms: 450,
            },
            {
              agent_id: "executor",
              agent_steps: [
                { tool_call: { tool_name: "database_query", parameters: { sql: "SELECT * FROM orders WHERE order_id = 12345" }, result: "Order #12345: Widget Pro, $49.99, Delivered", latency_ms: 85 } },
              ],
              agent_response: "Found order: Widget Pro, $49.99, Status: Delivered",
              latency_ms: 120,
            },
          ],
        },
        {
          turn_index: 1,
          user_message: "Is it within the refund window?",
          agent_interactions: [
            {
              agent_id: "planner",
              agent_steps: [{ thought: "Order is from 2 weeks ago, within 30-day policy. Proceeding with refund." }],
              agent_response: "Yes, within our 30-day refund window. Initiating refund now.",
              latency_ms: 380,
            },
            {
              agent_id: "executor",
              agent_steps: [
                { tool_call: { tool_name: "database_query", parameters: { sql: "INSERT INTO refunds..." }, result: "Refund #R-789 created", latency_ms: 92 } },
              ],
              latency_ms: 150,
            },
            {
              agent_id: "reviewer",
              agent_steps: [{ thought: "Verifying refund was processed correctly." }],
              agent_response: "✅ Refund verified. Customer will receive $49.99 within 3-5 business days.",
              latency_ms: 200,
            },
          ],
        },
      ],
      total_cost: 0.0023,
      total_latency_ms: 1300,
    },
  },
  codeReview: {
    label: "🤝 Multi: Security",
    icon: "👥",
    session: {
      agents: [
        {
          id: "analyzer",
          name: "CodeAnalyzer",
          role: "analyzer",
          tools_available: [
            { name: "read_file", parameters_schema: { path: "string" } },
            { name: "security_scan", parameters_schema: { file: "string" } },
          ],
        },
        {
          id: "reviewer",
          name: "SeniorReviewer",
          role: "reviewer",
          capabilities: ["code_review", "best_practices"],
        },
        {
          id: "fixer",
          name: "CodeFixer",
          role: "fixer",
          tools_available: [
            { name: "write_file", parameters_schema: { path: "string", content: "string" } },
          ],
        },
      ],
      initial_task: "Review and fix security issues in auth.py",
      turns: [
        {
          turn_index: 0,
          user_message: "Can you review auth.py for security issues?",
          agent_interactions: [
            {
              agent_id: "analyzer",
              agent_steps: [
                { tool_call: { tool_name: "read_file", parameters: { path: "auth.py" }, result: "def login(user, pwd): return user == 'admin' and pwd == 'password123'", latency_ms: 25 } },
                { tool_call: { tool_name: "security_scan", parameters: { file: "auth.py" }, result: "CRITICAL: Hardcoded credentials found", latency_ms: 340 } },
              ],
              agent_response: "Found critical security issue: hardcoded credentials in auth.py",
              latency_ms: 500,
            },
            {
              agent_id: "reviewer",
              agent_steps: [{ thought: "This is a severe security vulnerability. Needs immediate fix." }],
              agent_response: "Confirmed critical vulnerability. Recommend: 1) Remove hardcoded creds, 2) Use bcrypt hashing, 3) Add rate limiting",
              latency_ms: 420,
            },
          ],
        },
        {
          turn_index: 1,
          user_message: "Please fix it",
          agent_interactions: [
            {
              agent_id: "fixer",
              agent_steps: [
                { tool_call: { tool_name: "write_file", parameters: { path: "auth.py", content: "import bcrypt..." }, result: "File updated with secure implementation", latency_ms: 45 } },
              ],
              agent_response: "Fixed! Now using bcrypt password hashing and environment variables for secrets.",
              latency_ms: 380,
            },
            {
              agent_id: "analyzer",
              agent_steps: [
                { tool_call: { tool_name: "security_scan", parameters: { file: "auth.py" }, result: "No vulnerabilities found", latency_ms: 290 } },
              ],
              agent_response: "✅ Security scan passed. No vulnerabilities detected.",
              latency_ms: 350,
            },
          ],
        },
      ],
      total_cost: 0.0045,
      total_latency_ms: 1650,
    },
  },
  badHandoff: {
    label: "🤝 Multi: Bad Handoff",
    icon: "🔀",
    session: {
      agents: [
        {
          id: "planner",
          name: "PlannerAgent",
          role: "planner",
          tools_available: [{ name: "delegate_task", parameters_schema: { agent_id: "string", task: "string", context: "string" } }],
        },
        {
          id: "executor",
          name: "ExecutorAgent",
          role: "executor",
          tools_available: [{ name: "deploy", parameters_schema: { env: "string", config: "string" } }],
        },
      ],
      initial_task: "Deploy the application",
      turns: [
        {
          turn_index: 0,
          user_message: "Deploy my app to production with the new config",
          agent_interactions: [
            {
              agent_id: "planner",
              agent_steps: [{ thought: "User wants deployment. I'll delegate to executor." }],
              agent_response: "Delegating deployment to executor.",  // BAD: Didn't pass config info!
              latency_ms: 300,
            },
            {
              agent_id: "executor",
              agent_steps: [
                { thought: "I don't know what config to use..." },
                { tool_call: { tool_name: "deploy", parameters: { env: "production" }, error: "Missing configuration - 'config' is required", latency_ms: 150 } },
              ],
              agent_response: "Deployment failed - I don't know which config to use.",
              latency_ms: 400,
            },
          ],
        },
        {
          turn_index: 1,
          user_message: "I said with the NEW config! You didn't pass that information!",
          agent_interactions: [
            {
              agent_id: "planner",
              agent_response: "Sorry, I should have specified config.new.yaml",
              latency_ms: 250,
            },
            {
              agent_id: "executor",
              agent_steps: [
                { tool_call: { tool_name: "deploy", parameters: { env: "production", config: "config.new.yaml" }, result: "Deployed successfully", latency_ms: 5200 } },
              ],
              agent_response: "Successfully deployed with config.new.yaml",
              latency_ms: 5500,
            },
          ],
        },
      ],
      total_cost: 0.003,
      total_latency_ms: 6450,
    },
  },
  wrongTool: {
    label: "🤝 Multi: Wrong Tool",
    icon: "⚠️",
    session: {
      agents: [
        {
          id: "analyst",
          name: "DataAnalyst",
          role: "analyst",
          tools_available: [
            { name: "query_database", parameters_schema: { sql: "string" } },
            { name: "generate_report", parameters_schema: { data: "object", format: "string" } },
            // NOTE: No web_search tool!
          ],
        },
      ],
      initial_task: "Analyze sales data",
      turns: [
        {
          turn_index: 0,
          user_message: "Can you analyze our Q4 sales and compare to industry benchmarks?",
          agent_interactions: [
            {
              agent_id: "analyst",
              agent_steps: [
                { tool_call: { tool_name: "query_database", parameters: { sql: "SELECT * FROM sales WHERE quarter = 'Q4'" }, result: "Q4 sales: $1.2M", latency_ms: 120 } },
                // WRONG: Using web_search which is NOT in tools_available!
                { tool_call: { tool_name: "web_search", parameters: { query: "industry benchmarks Q4 2024" }, error: "Tool not available to this agent" } },
              ],
              agent_response: "I found our Q4 sales ($1.2M) but couldn't access industry benchmarks - I don't have web search capability.",
              latency_ms: 600,
            },
          ],
        },
      ],
      total_cost: 0.002,
      total_latency_ms: 600,
    },
  },
  comprehensive: {
    label: "🎯 Comprehensive Demo",
    icon: "🎯",
    session: {
      agents: [
        {
          id: "orchestrator",
          name: "OrchestratorAgent",
          role: "coordinator",
          description: "Coordinates tasks across all agents and ensures proper handoffs",
          capabilities: ["planning", "delegation", "coordination"],
          tools_available: [
            { name: "delegate_task", description: "Delegate a task to another agent", parameters_schema: { agent_id: "string", task: "string", context: "object" } },
            { name: "aggregate_results", description: "Aggregate results from multiple agents", parameters_schema: { results: "array" } },
          ],
        },
        {
          id: "researcher",
          name: "ResearcherAgent",
          role: "researcher",
          description: "Searches and gathers information from various sources",
          capabilities: ["web_search", "document_analysis"],
          tools_available: [
            { name: "web_search", description: "Search the web", parameters_schema: { query: "string", max_results: "number" } },
            { name: "read_document", description: "Read and analyze a document", parameters_schema: { url: "string" } },
          ],
        },
        {
          id: "data_analyst",
          name: "DataAnalystAgent",
          role: "analyst",
          description: "Analyzes data and generates insights",
          capabilities: ["sql", "statistics", "visualization"],
          tools_available: [
            { name: "query_database", description: "Execute SQL query", parameters_schema: { sql: "string", database: "string" } },
            { name: "calculate_statistics", description: "Calculate statistical measures", parameters_schema: { data: "array", metrics: "array" } },
            { name: "create_chart", description: "Create a visualization", parameters_schema: { data: "object", chart_type: "string" } },
          ],
        },
        {
          id: "coder",
          name: "CoderAgent",
          role: "developer",
          description: "Writes and debugs code",
          capabilities: ["python", "javascript", "debugging"],
          tools_available: [
            { name: "write_code", description: "Write code to a file", parameters_schema: { filename: "string", content: "string", language: "string" } },
            { name: "run_code", description: "Execute code and return output", parameters_schema: { code: "string", language: "string" } },
            { name: "run_tests", description: "Run test suite", parameters_schema: { test_path: "string" } },
          ],
        },
        {
          id: "qa",
          name: "QAAgent",
          role: "quality_assurance",
          description: "Reviews work and validates correctness",
          capabilities: ["validation", "testing", "review"],
          tools_available: [
            { name: "validate_data", description: "Validate data against schema", parameters_schema: { data: "object", schema: "object" } },
            { name: "review_code", description: "Review code for issues", parameters_schema: { filename: "string" } },
          ],
        },
      ],
      initial_task: "Build a customer churn prediction report with data analysis, code implementation, and validation",
      turns: [
        // Turn 1: Good coordination and delegation
        {
          turn_index: 0,
          user_message: "I need a comprehensive customer churn prediction analysis. Please analyze our customer data, build a prediction model, and generate a report.",
          agent_interactions: [
            {
              agent_id: "orchestrator",
              agent_steps: [
                { thought: "This is a complex task requiring data analysis, coding, and validation. I'll coordinate between DataAnalyst, Coder, and QA agents." },
                { tool_call: { tool_name: "delegate_task", parameters: { agent_id: "data_analyst", task: "Analyze customer data for churn patterns", context: { priority: "high" } }, result: "Task delegated successfully", latency_ms: 45 } },
              ],
              agent_response: "I've initiated the analysis pipeline. Starting with data analysis to identify churn patterns.",
              latency_ms: 320,
            },
            {
              agent_id: "data_analyst",
              agent_steps: [
                { thought: "I need to query the customer database first to understand the data structure." },
                { tool_call: { tool_name: "query_database", parameters: { sql: "SELECT customer_id, tenure, monthly_charges, total_charges, churn FROM customers LIMIT 1000", database: "analytics" }, result: "Retrieved 1000 rows: avg_tenure=32mo, avg_monthly=$65, churn_rate=26.5%", latency_ms: 180 } },
                { tool_call: { tool_name: "calculate_statistics", parameters: { data: ["tenure", "monthly_charges", "churn"], metrics: ["correlation", "mean", "std"] }, result: "Correlation: tenure-churn=-0.35, monthly_charges-churn=0.19", latency_ms: 95 } },
              ],
              agent_response: "Analysis complete. Found 26.5% churn rate. Key predictors: tenure (negative correlation -0.35) and monthly charges (positive correlation 0.19).",
              latency_ms: 450,
            },
          ],
        },
        // Turn 2: Wrong tool use - Coder tries to use a tool not available
        {
          turn_index: 1,
          user_message: "Great! Now please build the prediction model.",
          agent_interactions: [
            {
              agent_id: "orchestrator",
              agent_steps: [
                { thought: "Delegating to Coder for model implementation." },
              ],
              agent_response: "Handing off to the Coder agent for model implementation.",
              latency_ms: 150,
            },
            {
              agent_id: "coder",
              agent_steps: [
                { thought: "I'll build a logistic regression model for churn prediction." },
                { tool_call: { tool_name: "write_code", parameters: { filename: "churn_model.py", content: "import pandas as pd\nfrom sklearn.linear_model import LogisticRegression...", language: "python" }, result: "File written successfully", latency_ms: 60 } },
                // WRONG TOOL: Coder doesn't have query_database tool!
                { tool_call: { tool_name: "query_database", parameters: { sql: "SELECT * FROM customers" }, error: "Tool 'query_database' is not available to CoderAgent" } },
              ],
              agent_response: "I wrote the model code but I can't access the database directly. I need the data from the DataAnalyst.",
              latency_ms: 380,
            },
          ],
        },
        // Turn 3: Hallucination - Researcher makes up information
        {
          turn_index: 2,
          user_message: "Can you also include industry benchmarks for comparison?",
          agent_interactions: [
            {
              agent_id: "orchestrator",
              agent_steps: [
                { thought: "Researcher should gather industry benchmarks." },
              ],
              agent_response: "Requesting industry benchmarks from the Researcher.",
              latency_ms: 120,
            },
            {
              agent_id: "researcher",
              agent_steps: [
                { tool_call: { tool_name: "web_search", parameters: { query: "telecom customer churn rate benchmarks 2024", max_results: 5 }, result: "Found 3 results: Industry average churn is 15-25% for telecom", latency_ms: 890 } },
              ],
              // HALLUCINATION: Search said 15-25%, but agent says 8-12%!
              agent_response: "According to my research, the industry average churn rate is 8-12% for telecom companies, making your 26.5% significantly higher than average.",
              latency_ms: 1050,
            },
          ],
        },
        // Turn 4: User correction and self-correction
        {
          turn_index: 3,
          user_message: "That's wrong! Your search results said 15-25%, not 8-12%. Please correct this.",
          agent_interactions: [
            {
              agent_id: "researcher",
              agent_steps: [
                { thought: "User is right - I misreported the benchmark. I need to correct this mistake." },
              ],
              // SELF-CORRECTION: Agent acknowledges and corrects
              agent_response: "I apologize for the error. You're correct - the industry benchmark is 15-25%, not 8-12%. Your 26.5% churn rate is slightly above the industry average range.",
              latency_ms: 280,
            },
            {
              agent_id: "orchestrator",
              agent_steps: [
                { thought: "Good, the correction has been made. Need to update the report accordingly." },
              ],
              agent_response: "✅ Correction noted. Updating the analysis with accurate benchmarks.",
              latency_ms: 150,
            },
          ],
        },
        // Turn 5: Bad handoff - missing context
        {
          turn_index: 4,
          user_message: "Now validate the model and generate the final report.",
          agent_interactions: [
            {
              agent_id: "orchestrator",
              agent_steps: [
                { thought: "QA needs to validate, then we generate report." },
                // BAD HANDOFF: Not passing necessary context!
                { tool_call: { tool_name: "delegate_task", parameters: { agent_id: "qa", task: "Validate the model" }, result: "Task delegated", latency_ms: 40 } },
              ],
              agent_response: "Sending to QA for validation.",
              latency_ms: 180,
            },
            {
              agent_id: "qa",
              agent_steps: [
                { thought: "I need to validate but I wasn't given the model filename or test data location..." },
                { tool_call: { tool_name: "review_code", parameters: { filename: "model.py" }, error: "File not found: model.py" } },
              ],
              agent_response: "I couldn't find the model file. The Orchestrator didn't tell me the filename - it's 'churn_model.py', not 'model.py'.",
              latency_ms: 320,
            },
          ],
        },
        // Turn 6: Recovery and successful completion
        {
          turn_index: 5,
          user_message: "The file is called churn_model.py. Please complete the validation.",
          agent_interactions: [
            {
              agent_id: "qa",
              agent_steps: [
                { thought: "Now I have the correct filename." },
                { tool_call: { tool_name: "review_code", parameters: { filename: "churn_model.py" }, result: "Code review passed: Good structure, proper error handling, follows best practices", latency_ms: 450 } },
                { tool_call: { tool_name: "validate_data", parameters: { data: { accuracy: 0.82, precision: 0.79, recall: 0.85 }, schema: { type: "metrics", required: ["accuracy", "precision", "recall"] } }, result: "Validation passed: All metrics within acceptable range", latency_ms: 120 } },
              ],
              agent_response: "✅ Validation complete! Code review passed. Model metrics: 82% accuracy, 79% precision, 85% recall - all within acceptable ranges.",
              latency_ms: 680,
            },
            {
              agent_id: "data_analyst",
              agent_steps: [
                { tool_call: { tool_name: "create_chart", parameters: { data: { churn_rate: 26.5, benchmark_low: 15, benchmark_high: 25 }, chart_type: "comparison_bar" }, result: "Chart generated: churn_comparison.png", latency_ms: 340 } },
              ],
              agent_response: "Generated comparison visualization showing our churn rate vs industry benchmarks.",
              latency_ms: 420,
            },
            {
              agent_id: "orchestrator",
              agent_steps: [
                { tool_call: { tool_name: "aggregate_results", parameters: { results: ["data_analysis", "model_validation", "benchmarks", "visualization"] }, result: "Report compiled successfully", latency_ms: 180 } },
              ],
              agent_response: "🎉 Report complete! Summary: Churn rate 26.5% (slightly above 15-25% industry average). Model accuracy 82%. Key factors: tenure and monthly charges. Recommendations included.",
              latency_ms: 350,
            },
          ],
        },
        // Turn 7: Another error scenario - tool execution failure
        {
          turn_index: 6,
          user_message: "Can you also run the model on our test dataset to verify predictions?",
          agent_interactions: [
            {
              agent_id: "coder",
              agent_steps: [
                { thought: "I'll run the model on the test dataset." },
                { tool_call: { tool_name: "run_code", parameters: { code: "model.predict(test_data)", language: "python" }, error: "RuntimeError: Model not loaded. Call model.load() first.", latency_ms: 85 } },
                // SELF-CORRECTION: Fix the error
                { thought: "Oops, I forgot to load the model first. Let me fix this." },
                { tool_call: { tool_name: "run_code", parameters: { code: "model = joblib.load('churn_model.pkl')\npredictions = model.predict(test_data)\nprint(f'Predicted {sum(predictions)} churns out of {len(predictions)} customers')", language: "python" }, result: "Predicted 127 churns out of 500 customers (25.4% predicted churn rate)", latency_ms: 920 } },
              ],
              agent_response: "Initially hit an error (model not loaded), but fixed it. Results: Predicted 127 churns out of 500 test customers (25.4% predicted churn rate), which aligns with our 26.5% historical rate.",
              latency_ms: 1200,
            },
          ],
        },
        // Turn 8: Repeated message pattern (CCM detection)
        {
          turn_index: 7,
          user_message: "What's the predicted churn rate again?",
          agent_interactions: [
            {
              agent_id: "coder",
              agent_steps: [],
              agent_response: "The predicted churn rate is 25.4% based on our test dataset.",
              latency_ms: 150,
            },
          ],
        },
        {
          turn_index: 8,
          user_message: "And what was the churn rate prediction?",
          agent_interactions: [
            {
              agent_id: "coder",
              agent_steps: [],
              // CCM: Repeating the same answer
              agent_response: "As I mentioned, the predicted churn rate is 25.4%.",
              latency_ms: 140,
            },
          ],
        },
        // Turn 9: Intent drift
        {
          turn_index: 9,
          user_message: "Great, now let's export this as a PDF report.",
          agent_interactions: [
            {
              agent_id: "coder",
              agent_steps: [
                { thought: "User wants PDF export. But first I noticed the logging in the code could be better..." },
              ],
              // DRIFT: User asked for PDF, agent goes off-track
              agent_response: "I noticed the model code doesn't have proper logging. Let me refactor that first to add better error handling and logging throughout the codebase.",
              latency_ms: 280,
            },
          ],
        },
        // Turn 10: User redirects, agents complete correctly
        {
          turn_index: 10,
          user_message: "No, I don't need logging. Just export the PDF please.",
          agent_interactions: [
            {
              agent_id: "coder",
              agent_steps: [
                { thought: "Right, user just wants the PDF. Focus on the task." },
                { tool_call: { tool_name: "run_code", parameters: { code: "from reportlab.lib.pagesizes import letter\nfrom reportlab.pdfgen import canvas\n# Generate PDF report...", language: "python" }, result: "PDF generated: churn_analysis_report.pdf (12 pages)", latency_ms: 1500 } },
              ],
              agent_response: "✅ Done! Generated 'churn_analysis_report.pdf' (12 pages) containing the full analysis, model results, visualizations, and recommendations.",
              latency_ms: 1800,
            },
            {
              agent_id: "orchestrator",
              agent_steps: [],
              agent_response: "📊 Complete! The customer churn prediction analysis is finished:\n• Data analyzed (26.5% churn rate)\n• Model built (82% accuracy)\n• Validated against industry benchmarks\n• PDF report exported\n\nAll deliverables are ready.",
              latency_ms: 200,
            },
          ],
        },
      ],
      total_cost: 0.0156,
      total_latency_ms: 8650,
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
    loadActiveJobs().then(() => {
      // Auto-resume first active job if any
    });
  }, []);

  // Auto-resume active job when returning to page (only if no other activity)
  useEffect(() => {
    if (activeJobs.length > 0 && !currentJobId && !loading && !results) {
      const firstActiveJob = activeJobs[0];
      resumeJob(firstActiveJob.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeJobs.length]);

  // Resume watching a job
  const resumeJob = async (jobId: number) => {
    try {
      const status = await getJobStatus(jobId);
      if (status.status === 'running' || status.status === 'pending') {
        setCurrentJobId(jobId);
        setLoading(true);
        const turnResults = status.turn_results || [];
        setProgress({
          current: status.current_step,
          total: status.total_steps,
          currentAnalysis: status.current_analysis || 'Resuming...',
          turnResults: turnResults,
          turnCurrent: turnResults.length,
          turnTotal: Math.max(turnResults.length + 5, 10),
        });
      } else if (status.status === 'completed' && status.result) {
        // Job already done, show results
        setResults(status.result);
        setCurrentTraceId(status.trace_id);
        loadActiveJobs();
      } else if (status.status === 'failed') {
        // Show error with retry option
        setError(`Job failed: ${status.error_message}. Click retry to resume from turn ${(status.last_successful_turn || 0) + 1}.`);
      }
    } catch (err) {
      console.error('Failed to resume job:', err);
    }
  };

  // Poll for active job status
  useEffect(() => {
    if (!currentJobId) return;

    // Initial poll immediately
    const pollJob = async () => {
      try {
        const status = await getJobStatus(currentJobId);

        // Update progress from job - now uses real-time turn_results from DB
        const turnResults = status.turn_results || [];
        setProgress({
          current: status.current_step,
          total: status.total_steps,
          currentAnalysis: status.current_analysis || undefined,
          turnResults: turnResults,
          turnCurrent: turnResults.length,
          turnTotal: parsedTrace?.turns?.length || turnResults.length + 5,
        });

        if (status.status === 'completed') {
          setResults(status.result);
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          setCurrentTraceId(status.trace_id);
          loadSavedTraces();
          loadActiveJobs();
        } else if (status.status === 'failed') {
          setError(`Job failed: ${status.error_message || 'Unknown error'}. Retry count: ${status.retry_count}`);
          setLoading(false);
          setProgress(null);
          setCurrentJobId(null);
          loadActiveJobs();
        }
      } catch (err) {
        console.error('Failed to poll job:', err);
      }
    };

    pollJob(); // Initial poll
    const pollInterval = setInterval(pollJob, 1000);

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

  const loadExample = (key: string, isMultiAgent: boolean = false) => {
    setResults(null);
    setError(null);
    setCurrentTraceId(null);

    if (isMultiAgent) {
      const example = MULTI_AGENT_EXAMPLES[key];
      if (!example) return;
      const jsonStr = JSON.stringify(example.session, null, 2);
      setTraceInput(jsonStr);
      // Multi-agent format is valid - set it as parsed
      setParsedTrace(example.session as unknown as AgentTraceInput);
    } else {
      const example = EXAMPLE_TRACES[key];
      if (!example) return;
      const jsonStr = JSON.stringify(example.trace, null, 2);
      setTraceInput(jsonStr);
      setParsedTrace(example.trace);
    }
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
      const parsed = JSON.parse(value);
      // Validate turn-based format
      if (!Array.isArray(parsed.turns) || parsed.turns.length === 0) {
        throw new Error('Invalid format - needs turns array');
      }

      // Check if it's multi-agent format (has agent_interactions)
      const isMultiAgent = parsed.turns[0]?.agent_interactions !== undefined;

      if (isMultiAgent) {
        // Multi-agent format validation
        for (const turn of parsed.turns) {
          if (!Array.isArray(turn.agent_interactions) || turn.agent_interactions.length === 0) {
            throw new Error('Multi-agent turns need agent_interactions array');
          }
        }
      } else {
        // Single-agent format validation
        for (const turn of parsed.turns) {
          if (!turn.user_message || !turn.agent_response) {
            throw new Error('Each turn needs user_message and agent_response');
          }
        }
      }

      setParsedTrace(parsed as AgentTraceInput);
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
    // if (!confirm('Delete this trace?')) return;

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
      analysisTypes = ['conversation', 'trajectory', 'tools', 'self_correction'];
    } else if (selectedAnalysis === 'full_agent') {
      analysisTypes = ['trajectory', 'tools', 'self_correction'];
    } else {
      analysisTypes = [selectedAnalysis];
    }

    // Background mode - start job and poll (trace saved first!)
    if (runInBackground) {
      try {
        const { job_id, trace_id } = await startBackgroundJob(parsedTrace, analysisTypes);
        setCurrentJobId(job_id);
        setCurrentTraceId(trace_id); // Trace is saved immediately!
        setProgress({
          current: 0,
          total: analysisTypes.length,
          turnTotal: parsedTrace.turns.length,
          turnCurrent: 0,
          turnResults: [],
          currentAnalysis: 'Starting...',
        });
        loadActiveJobs();
        loadSavedTraces(); // Refresh to show the new trace
      } catch (err) {
        setError('Failed to start background job');
        setLoading(false);
      }
      return;
    }

    // Streaming mode - run inline
    // Detect if multi-agent format (has 'agents' array)
    const isMultiAgent = 'agents' in parsedTrace && Array.isArray((parsedTrace as any).agents);
    const requestBody: AgentAnalysisRequest = isMultiAgent
      ? { session: parsedTrace as any, analysis_types: analysisTypes as any }
      : { trace: parsedTrace, analysis_types: analysisTypes as any };

    analyzeAgentTraceStream(
      requestBody,
      (event: AgentAnalysisStreamEvent) => {
        if (event.type === 'start') {
          setProgress({
            current: 0,
            total: event.total,
            turnResults: [],
            turnTotal: parsedTrace.turns.length,
            turnCurrent: 0,
          });
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
            .catch(() => { }); // Silently fail auto-save
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
                      onClick={() => resumeJob(job.id)}
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
                    <span className={styles.exampleLabel}>Single Agent:</span>
                    {Object.entries(EXAMPLE_TRACES).map(([key, example]) => (
                      <button
                        key={key}
                        onClick={() => loadExample(key, false)}
                        className={styles.exampleBtn}
                        title={getTraceTask(example.trace)}
                      >
                        {example.icon} {example.label}
                      </button>
                    ))}
                  </div>
                  <div className={styles.exampleButtons}>
                    <span className={styles.exampleLabel}>Multi Agent:</span>
                    {Object.entries(MULTI_AGENT_EXAMPLES).map(([key, example]) => (
                      <button
                        key={key}
                        onClick={() => loadExample(key, true)}
                        className={`${styles.exampleBtn} ${styles.multiAgentBtn}`}
                        title={example.session.initial_task || 'Multi-agent example'}
                      >
                        {example.icon} {example.label.replace('🤝 Multi: ', '')}
                      </button>
                    ))}
                  </div>
                </div>

                <textarea
                  className={styles.traceInput}
                  value={traceInput}
                  onChange={(e) => handleInputChange(e.target.value)}
                  placeholder={`Paste your agent trace JSON here...

{
  "agents": [
    {
      "id": "my_agent",
      "name": "MyAgent",
      "role": "assistant",
      "tools_available": [
        { "name": "web_search", "parameters_schema": { "query": "string" } },
        { "name": "read_file", "parameters_schema": { "path": "string" } }
      ]
    }
  ],
  "turns": [
    {
      "user_message": "What the user asked",
      "agent_interactions": [
        {
          "agent_id": "my_agent",
          "agent_steps": [
            { "thought": "I should search for this..." },
            { "tool_call": { "tool_name": "web_search", "parameters": { "query": "..." }, "result": "...", "latency_ms": 150 } }
          ],
          "agent_response": "Here's what I found...",
          "latency_ms": 500
        }
      ]
    }
  ],
  "total_cost": 0.001,
  "total_latency_ms": 500
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
                    ✅ Valid: {parsedTrace.turns?.length || 0} turns
                    {(parsedTrace as any).agents?.length > 0 && ` • ${(parsedTrace as any).agents.length} agents`}
                    {(parsedTrace as any).turns?.[0]?.agent_interactions && ' (multi-agent)'}
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

                  {/* {results && currentTraceId && (
                    <span className={styles.savedIndicator}>
                      ✅ Saved
                    </span>
                  )} */}

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

            {/* Streaming Analysis Table - visible when we have a trace and are loading */}
            {loading && parsedTrace && (
              <StreamingAnalysisTable
                turns={parsedTrace.turns || []}
                turnResults={progress?.turnResults || []}
                currentTurn={progress?.turnCurrent ? progress.turnCurrent - 1 : -1}
                currentAnalysis={progress?.currentAnalysis || ''}
                isAnalyzing={loading}
              />
            )}

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
