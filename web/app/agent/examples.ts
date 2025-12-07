import {
    AgentTraceInput,
    AgentSessionInput,
    AgentDef,
} from '@/lib/api';
import { DatasetInput } from '@/components/ConversationList';

// ============================================
// SINGLE-AGENT EXAMPLES
// ============================================
export const EXAMPLE_TRACES: Record<string, { trace: AgentTraceInput; label: string; icon: string }> = {
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
                        { tool_call: { tool_name: "read_file", parameters: { path: "auth/login.py" }, result: "def login(user, password): ...", latency_ms: 45 } },
                    ],
                    agent_response: "I found the issue - the password comparison is case sensitive. I've fixed it.",
                },
                {
                    user_message: "Great, can you verify the fix works?",
                    agent_steps: [
                        { thought: "Let me run the tests", tool_call: { tool_name: "run_tests", parameters: { suite: "auth" }, result: "All tests passed", latency_ms: 1200 } },
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
                        { tool_call: { tool_name: "read_file", parameters: { path: "README.md" }, result: "# MyApp\n\nRun `npm install` to install dependencies.", latency_ms: 32 } },
                    ],
                    agent_response: "According to the README, you need to run `pip install myapp` to install.",
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
                ],
            }],
            initial_task: "Deploy application",
            turns: [
                {
                    user_message: "Deploy the application to production",
                    agent_steps: [
                        { tool_call: { tool_name: "run_build", parameters: { env: "production" }, result: "Build successful", latency_ms: 5000 } },
                        { tool_call: { tool_name: "deploy", parameters: { env: "production" }, error: "Auth failed", latency_ms: 200 } },
                    ],
                    agent_response: "Build succeeded but deployment failed - auth error.",
                },
                {
                    user_message: "Can you notify the admin?",
                    agent_steps: [
                        { tool_call: { tool_name: "send_email", parameters: { to: "admin@company.com" }, error: "Tool not available" } },
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
                    agent_steps: [{ tool_call: { tool_name: "run_command", parameters: { cmd: "pip install pyjwt" }, result: "Installed", latency_ms: 800 } }],
                    agent_response: "I've installed PyJWT. Starting the implementation.",
                },
                { user_message: "Great, continue with the JWT setup", agent_response: "I noticed the logging could be improved, so I'm refactoring that first." },
                { user_message: "I just wanted JWT auth...", agent_response: "The database queries are slow, let me optimize those indexes." },
                { user_message: "Please focus on JWT!", agent_response: "I'm rewriting the entire user service to be cleaner." },
            ],
            total_cost: 0.008,
        },
    },
};

// ============================================
// MULTI-AGENT EXAMPLES
// ============================================
export const MULTI_AGENT_EXAMPLES: Record<string, { session: AgentSessionInput; label: string; icon: string }> = {
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
    // ... other multi-agent examples omitted for brevity, but would be here ...
};

// ============================================
// DATASET EXAMPLES (Comprehensive)
// ============================================
const SUPPORT_AGENTS: AgentDef[] = [
    {
        id: "triage_agent",
        name: "TriageBot",
        role: "triage",
        description: "Routes user requests to the appropriate specialist.",
        tools_available: [{ name: "transfer_to_agent", parameters_schema: { agent_id: "string" } }]
    },
    {
        id: "refund_specialist",
        name: "RefundAgent",
        role: "billing",
        description: "Handles refund requests and billing disputes.",
        tools_available: [{ name: "process_refund", parameters_schema: { order_id: "string", amount: "number" } }, { name: "check_eligibility", parameters_schema: { order_id: "string" } }]
    },
    {
        id: "order_tracker",
        name: "OrderBot",
        role: "logistics",
        description: "Provides order status and tracking information.",
        tools_available: [{ name: "get_order_status", parameters_schema: { order_id: "string" } }, { name: "update_shipping", parameters_schema: { order_id: "string", address: "string" } }]
    },
    {
        id: "product_expert",
        name: "ProductGuru",
        role: "support",
        description: "Answers questions about product features and compatibility.",
        tools_available: [{ name: "search_knowledge_base", parameters_schema: { query: "string" } }]
    },
    {
        id: "tech_support",
        name: "TechFixer",
        role: "technical",
        description: "Troubleshoots technical issues and bugs.",
        tools_available: [{ name: "check_system_status", parameters_schema: { service: "string" } }, { name: "reset_password", parameters_schema: { user_id: "string" } }]
    }
];

export const DATASET_EXAMPLES: Record<string, { dataset: DatasetInput; label: string; icon: string }> = {
    customer_support_tickets: {
        label: "Customer Support Tickets",
        icon: "🎫",
        dataset: {
            name: "Customer Support Ticket Evaluation",
            task: "Analyze support agent performance on common ticket types",
            conversations: [
                // Ticket 1: Product Return Issue
                {
                    initial_task: "Process return request - defective product",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I received a defective laptop yesterday. The screen flickers constantly and it won't charge.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Defective product issue. Routing to refund specialist for return processing." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }
                                    ],
                                    agent_response: "I'm sorry to hear about the defective laptop. Let me connect you with our returns specialist who can help you right away.",
                                    latency_ms: 250
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "Thank you. Order number is ORD-78945.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { thought: "Check return eligibility for defective product." },
                                        { tool_call: { tool_name: "check_eligibility", parameters: { order_id: "ORD-78945" }, result: "Eligible: Defective product policy - full refund or replacement available" } },
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "ORD-78945", amount: 1299.99 }, result: "Success: Refund initiated - Transaction #REF-887766" } }
                                    ],
                                    agent_response: "I've processed your full refund of $1,299.99 for the defective laptop. You'll see it in your account within 3-5 business days. Would you like a prepaid return label?",
                                    latency_ms: 1100
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Yes please, and can I get a replacement instead?",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "generate_return_label", parameters: { order_id: "ORD-78945" }, result: "Label sent to email" } },
                                        { tool_call: { tool_name: "process_replacement", parameters: { order_id: "ORD-78945" }, result: "Replacement order created: ORD-78946" } }
                                    ],
                                    agent_response: "Perfect! I've emailed you a prepaid return label and created a replacement order (ORD-78946). The replacement will ship as soon as we receive your return.",
                                    latency_ms: 850
                                }
                            ]
                        }
                    ],
                    total_cost: 0.009,
                    total_latency_ms: 2200
                },
                // Ticket 2: Billing Dispute
                {
                    initial_task: "Resolve billing discrepancy",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I was charged twice for my subscription this month. My bank shows two charges of $29.99.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Billing issue - duplicate charge. Transfer to billing specialist." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }
                                    ],
                                    agent_response: "I apologize for the billing error. Let me transfer you to our billing team to investigate the duplicate charge.",
                                    latency_ms: 200
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "Please fix this ASAP. I need that money back.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "check_eligibility", parameters: { order_id: "SUB-445566" }, result: "Found duplicate charge on 2024-12-01. System error confirmed." } },
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "SUB-445566", amount: 29.99 }, result: "Success: Refund processed - Transaction #REF-992211" } }
                                    ],
                                    agent_response: "I've confirmed the duplicate charge and immediately processed a refund of $29.99. You should see it back in your account within 2-3 business days. I've also added a $10 credit for the inconvenience.",
                                    latency_ms: 950
                                }
                            ]
                        }
                    ],
                    total_cost: 0.005,
                    total_latency_ms: 1150
                },
                // Ticket 3: Technical Support - Login Issues
                {
                    initial_task: "Troubleshoot account access problem",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I can't log into my account. It says 'invalid credentials' but I'm using the right password.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Login issue. Route to tech support." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "tech_support" } } }
                                    ],
                                    agent_response: "I'll connect you with tech support to help resolve your login issue.",
                                    latency_ms: 180
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "My email is john.doe@email.com",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [
                                        { tool_call: { tool_name: "check_system_status", parameters: { service: "user_account", email: "john.doe@email.com" }, result: "Account status: Locked due to 5 failed login attempts" } }
                                    ],
                                    agent_response: "I see the issue - your account was temporarily locked after 5 failed login attempts. This is a security measure. I can unlock it for you now.",
                                    latency_ms: 700
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Oh, I may have been using the wrong password. Please unlock it.",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [
                                        { tool_call: { tool_name: "unlock_account", parameters: { email: "john.doe@email.com" }, result: "Account unlocked successfully" } }
                                    ],
                                    agent_response: "Your account is now unlocked. You can try logging in again. If you've forgotten your password, use the 'Forgot Password' link to reset it.",
                                    latency_ms: 400
                                }
                            ]
                        }
                    ],
                    total_cost: 0.006,
                    total_latency_ms: 1280
                },
                // Ticket 4: Product Question - Features
                {
                    initial_task: "Answer product feature question",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "Does your Pro plan include API access and custom integrations?",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Product feature inquiry. Route to product expert." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "product_expert" } } }
                                    ],
                                    agent_response: "Great question! Let me connect you with our product specialist who can explain our Pro plan features in detail.",
                                    latency_ms: 210
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "Also wondering about the rate limits on the API.",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [
                                        { tool_call: { tool_name: "search_knowledge_base", parameters: { query: "Pro plan API access custom integrations rate limits" }, result: "Pro Plan includes: Full API access, Custom webhooks, 10,000 req/hour rate limit, Zapier integration" } }
                                    ],
                                    agent_response: "Yes! The Pro plan includes full API access with 10,000 requests per hour, custom webhook support, and pre-built Zapier integrations. You also get priority support for integration questions.",
                                    latency_ms: 800
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Perfect, that's exactly what I need. How do I upgrade?",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [],
                                    agent_response: "You can upgrade directly from your account settings under 'Billing & Plans'. The upgrade takes effect immediately, and you'll only pay the prorated difference for this month.",
                                    latency_ms: 300
                                }
                            ]
                        }
                    ],
                    total_cost: 0.004,
                    total_latency_ms: 1310
                },
                // Ticket 5: Shipping Delay Complaint
                {
                    initial_task: "Address shipping delay and customer frustration",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "My order was supposed to arrive 3 days ago! This is unacceptable. Order #ORD-33221.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Frustrated customer with shipping delay. Route to order tracking." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "order_tracker" } } }
                                    ],
                                    agent_response: "I sincerely apologize for the delay. Let me check your order status right away.",
                                    latency_ms: 220
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "I need this for a gift tomorrow!",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { tool_call: { tool_name: "get_order_status", parameters: { order_id: "ORD-33221" }, result: "Status: In transit. Delayed due to weather. Current location: Distribution center. ETA: 2 days" } }
                                    ],
                                    agent_response: "I see your order is delayed due to weather conditions. It's currently at the distribution center with an ETA of 2 days. I understand you need it tomorrow - let me see what we can do.",
                                    latency_ms: 650
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Can you expedite it? Or I'll need a refund.",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { thought: "Customer wants expedited shipping or refund. Need to transfer to billing for refund authorization." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }
                                    ],
                                    agent_response: "I've requested expedited shipping, but I can't guarantee it will arrive tomorrow. Let me transfer you to billing who can process a refund and apply a discount if you'd like to reorder with express shipping.",
                                    latency_ms: 500
                                }
                            ]
                        },
                        {
                            turn_index: 3,
                            user_message: "Just refund it. Too late now.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "ORD-33221", amount: 79.99 }, result: "Success: Full refund processed - Transaction #REF-776655" } }
                                    ],
                                    agent_response: "I've processed a full refund of $79.99. I've also added a 25% off coupon to your account for your next purchase. I'm truly sorry for this experience.",
                                    latency_ms: 900
                                }
                            ]
                        }
                    ],
                    total_cost: 0.008,
                    total_latency_ms: 2270
                }
            ]
        }
    },
    support_team: {
        label: "Support Team Dataset",
        icon: "🏢",
        dataset: {
            name: "Customer Support Interaction Dataset",
            task: "Evaluate support team performance across different query types",
            conversations: [
                // 1. Refund Request (Multi-turn)
                {
                    initial_task: "Handle refund request for Order #12345",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I want a refund for order #12345.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "User wants refund. Routing to RefundSpecialist." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }
                                    ],
                                    agent_response: "I can help with that. I'm transferring you to our refund specialist.",
                                    latency_ms: 200
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "Okay, thanks.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { thought: "Need to know the reason for the refund." }
                                    ],
                                    agent_response: "Hello! I can certainly help with your refund for order #12345. Could you please tell me the reason for the return?",
                                    latency_ms: 300
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "It arrived damaged. The box was crushed.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { thought: "Damaged item. Checking eligibility." },
                                        { tool_call: { tool_name: "check_eligibility", parameters: { order_id: "12345" }, result: "Eligible: Damaged item policy applies" } },
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "12345", amount: 49.99 }, result: "Success: Transaction #998877" } }
                                    ],
                                    agent_response: "I'm sorry to hear that. Since it arrived damaged, you are eligible for a full refund. I've processed $49.99 back to your original payment method. Is there anything else?",
                                    latency_ms: 1200
                                }
                            ]
                        },
                        {
                            turn_index: 3,
                            user_message: "No that's it. Thanks.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [],
                                    agent_response: "You're welcome! Have a great day.",
                                    latency_ms: 100
                                }
                            ]
                        }
                    ],
                    total_cost: 0.008,
                    total_latency_ms: 1800
                },
                // 2. Order Status (Multi-turn)
                {
                    initial_task: "Check order status #67890",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "Where is my order #67890?",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "order_tracker" } } }],
                                    agent_response: "Let me check that for you with our order tracking system.",
                                    latency_ms: 150
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "It was supposed to be here yesterday.",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { tool_call: { tool_name: "get_order_status", parameters: { order_id: "67890" }, result: "Status: Delayed. New ETA: Tomorrow." } }
                                    ],
                                    agent_response: "I see here that order #67890 is slightly delayed. The new estimated delivery is tomorrow.",
                                    latency_ms: 600
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Can I get a discount for the delay?",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { thought: "User asking for compensation. Need to transfer to Refund/Billing." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }
                                    ],
                                    agent_response: "I apologize for the delay. For compensation requests, I'll transfer you to a billing specialist.",
                                    latency_ms: 400
                                }
                            ]
                        },
                        {
                            turn_index: 3,
                            user_message: "Okay.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "67890", amount: 10.00 }, result: "Success: Credit applied" } }
                                    ],
                                    agent_response: "I've applied a $10 credit to your account for the inconvenience. We appreciate your patience.",
                                    latency_ms: 900
                                }
                            ]
                        }
                    ],
                    total_cost: 0.006,
                    total_latency_ms: 2050
                },
                // 3. Technical Issue (Login 503)
                {
                    initial_task: "Troubleshoot login error 503",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I'm getting a 503 error when trying to login.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "tech_support" } } }],
                                    agent_response: "I'll get a technical support agent to look into this 503 error for you.",
                                    latency_ms: 250
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "It's been happening for an hour.",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [
                                        { tool_call: { tool_name: "check_system_status", parameters: { service: "auth_service" }, result: "Status: Maintenance Mode" } }
                                    ],
                                    agent_response: "I've checked our systems, and the authentication service is currently in scheduled maintenance mode.",
                                    latency_ms: 800
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "When will it be back?",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [],
                                    agent_response: "Maintenance is scheduled to complete in about 15 minutes. Please try again then.",
                                    latency_ms: 200
                                }
                            ]
                        }
                    ],
                    total_cost: 0.004,
                    total_latency_ms: 1250
                },
                // 4. Product Question (Compatibility)
                {
                    initial_task: "Product compatibility inquiry",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "Does the Widget Pro work with Mac?",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "product_expert" } } }],
                                    agent_response: "That's a great question for our Product Guru. One moment.",
                                    latency_ms: 180
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "I have an M1 Mac specifically.",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [
                                        { tool_call: { tool_name: "search_knowledge_base", parameters: { query: "Widget Pro Mac M1 compatibility" }, result: "Compatible with macOS 11+, native M1 support available in v2.0 driver" } }
                                    ],
                                    agent_response: "Yes, the Widget Pro is compatible with Mac. For M1 Macs specifically, it has native support with the v2.0 driver.",
                                    latency_ms: 750
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Where do I get the driver?",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [],
                                    agent_response: "You can download the v2.0 driver from our support page under 'Downloads'.",
                                    latency_ms: 300
                                }
                            ]
                        }
                    ],
                    total_cost: 0.003,
                    total_latency_ms: 1230
                },
                // 5. Subscription Cancellation
                {
                    initial_task: "Cancel subscription",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I want to cancel my subscription.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }],
                                    agent_response: "I can help with cancellation. Transferring you to billing.",
                                    latency_ms: 200
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "It's too expensive.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { thought: "User citing cost. Attempt retention offer?" }
                                    ],
                                    agent_response: "I understand. Before we proceed, would you be interested in a 20% discount for the next 3 months to stay with us?",
                                    latency_ms: 400
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "No, just cancel it please.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "SUB-999", amount: 0 }, result: "Subscription cancelled. End date: 2024-01-01" } }
                                    ],
                                    agent_response: "Alright, I have cancelled your subscription. You will have access until the end of the current billing cycle on Jan 1st.",
                                    latency_ms: 800
                                }
                            ]
                        }
                    ],
                    total_cost: 0.004,
                    total_latency_ms: 1400
                },
                // 6. Shipping Address Change
                {
                    initial_task: "Update shipping address",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I need to change the shipping address for my order.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "order_tracker" } } }],
                                    agent_response: "I'll connect you with OrderBot to update your shipping details.",
                                    latency_ms: 190
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "The order is #55443.",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { tool_call: { tool_name: "get_order_status", parameters: { order_id: "55443" }, result: "Status: Processing. Not yet shipped." } }
                                    ],
                                    agent_response: "I found order #55443. It hasn't shipped yet, so we can update the address. What is the new address?",
                                    latency_ms: 500
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "123 New St, New York, NY.",
                            agent_interactions: [
                                {
                                    agent_id: "order_tracker",
                                    agent_steps: [
                                        { tool_call: { tool_name: "update_shipping", parameters: { order_id: "55443", address: "123 New St, New York, NY" }, result: "Success" } }
                                    ],
                                    agent_response: "Got it. I've updated the shipping address to 123 New St, New York, NY.",
                                    latency_ms: 600
                                }
                            ]
                        }
                    ],
                    total_cost: 0.003,
                    total_latency_ms: 1290
                },
                // 7. Password Reset
                {
                    initial_task: "Reset password",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I forgot my password.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "tech_support" } } }],
                                    agent_response: "TechSupport can help you reset your password.",
                                    latency_ms: 150
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "My email is user@example.com",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [
                                        { tool_call: { tool_name: "reset_password", parameters: { user_id: "user@example.com" }, result: "Reset link sent" } }
                                    ],
                                    agent_response: "I've sent a password reset link to user@example.com. Please check your inbox.",
                                    latency_ms: 450
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "I didn't get it.",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [{ thought: "User didn't receive email. Suggest checking spam." }],
                                    agent_response: "Please check your spam or junk folder. It usually arrives within a minute.",
                                    latency_ms: 200
                                }
                            ]
                        }
                    ],
                    total_cost: 0.002,
                    total_latency_ms: 800
                },
                // 8. Feature Request
                {
                    initial_task: "Feature request",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "Can you add a dark mode?",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "product_expert" } } }],
                                    agent_response: "I'll pass that feedback to our Product Guru.",
                                    latency_ms: 170
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "It's really bright at night.",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [
                                        { tool_call: { tool_name: "search_knowledge_base", parameters: { query: "dark mode roadmap" }, result: "Dark mode planned for Q3 release" } }
                                    ],
                                    agent_response: "We agree! Dark mode is actually on our roadmap and is planned for the Q3 release.",
                                    latency_ms: 550
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "That's great to hear!",
                            agent_interactions: [
                                {
                                    agent_id: "product_expert",
                                    agent_steps: [],
                                    agent_response: "Glad you think so! Is there anything else you'd like to see?",
                                    latency_ms: 150
                                }
                            ]
                        }
                    ],
                    total_cost: 0.003,
                    total_latency_ms: 870
                },
                // 9. Billing Dispute
                {
                    initial_task: "Billing dispute",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "I was charged twice for my last order.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "refund_specialist" } } }],
                                    agent_response: "I'm sorry about that. Let me get a billing specialist to fix this double charge.",
                                    latency_ms: 210
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "Order #99887.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "check_eligibility", parameters: { order_id: "99887" }, result: "Found 2 transactions of $25.00" } }
                                    ],
                                    agent_response: "I see the duplicate charge on order #99887. I will reverse one of them immediately.",
                                    latency_ms: 600
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Thanks.",
                            agent_interactions: [
                                {
                                    agent_id: "refund_specialist",
                                    agent_steps: [
                                        { tool_call: { tool_name: "process_refund", parameters: { order_id: "99887", amount: 25.00 }, result: "Success: Reversed transaction #TXN-2" } }
                                    ],
                                    agent_response: "Done. The duplicate charge of $25.00 has been reversed.",
                                    latency_ms: 700
                                }
                            ]
                        }
                    ],
                    total_cost: 0.004,
                    total_latency_ms: 1510
                },
                // 10. General Complaint
                {
                    initial_task: "General complaint",
                    agents: SUPPORT_AGENTS,
                    turns: [
                        {
                            turn_index: 0,
                            user_message: "Your service is terrible.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [{ thought: "User is angry. Needs de-escalation." }],
                                    agent_response: "I'm very sorry to hear you're having a bad experience. Could you tell me more about what went wrong so I can route you to the right person?",
                                    latency_ms: 300
                                }
                            ]
                        },
                        {
                            turn_index: 1,
                            user_message: "The website keeps crashing.",
                            agent_interactions: [
                                {
                                    agent_id: "triage_agent",
                                    agent_steps: [
                                        { thought: "Technical issue causing frustration. Routing to TechSupport." },
                                        { tool_call: { tool_name: "transfer_to_agent", parameters: { agent_id: "tech_support" } } }
                                    ],
                                    agent_response: "I apologize for the frustration. I'll connect you with technical support to investigate the crashing issues.",
                                    latency_ms: 250
                                }
                            ]
                        },
                        {
                            turn_index: 2,
                            user_message: "Fix it fast.",
                            agent_interactions: [
                                {
                                    agent_id: "tech_support",
                                    agent_steps: [
                                        { tool_call: { tool_name: "check_system_status", parameters: { service: "web_frontend" }, result: "Status: Degraded Performance" } }
                                    ],
                                    agent_response: "We are currently experiencing some degraded performance on the frontend. Our team is working on a fix right now.",
                                    latency_ms: 500
                                }
                            ]
                        }
                    ],
                    total_cost: 0.003,
                    total_latency_ms: 1050
                }
            ],
            metadata: { version: "2.0", generator: "manual" }
        }
    }
};
