

const agents = [
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

const scenarios = [
    {
        task: "Handle refund request for Order #12345",
        user_msg: "I want a refund for order #12345, it arrived damaged.",
        flow: [
            ["triage_agent", "I see you have a refund request. I'll transfer you to our billing specialist.", "transfer_to_agent", { agent_id: "refund_specialist" }],
            ["refund_specialist", "Hello, I can help with your refund. Let me check the order details.", "check_eligibility", { order_id: "12345" }],
            ["refund_specialist", "The order is eligible for a refund due to damage. I'm processing it now.", "process_refund", { order_id: "12345", amount: 49.99 }],
            ["refund_specialist", "Your refund of $49.99 has been processed. Is there anything else?", null, null]
        ]
    },
    {
        task: "Check status of Order #67890",
        user_msg: "Where is my order #67890? It's late.",
        flow: [
            ["triage_agent", "I can help you check your order status. Transferring to OrderBot.", "transfer_to_agent", { agent_id: "order_tracker" }],
            ["order_tracker", "Checking status for #67890...", "get_order_status", { order_id: "67890" }],
            ["order_tracker", "It looks like your order was delayed due to weather. It is expected to arrive tomorrow.", null, null]
        ]
    },
    {
        task: "Product Compatibility Question",
        user_msg: "Does the X-2000 work with Mac OS?",
        flow: [
            ["triage_agent", "That's a product question. Let me get an expert.", "transfer_to_agent", { agent_id: "product_expert" }],
            ["product_expert", "Let me check the compatibility list for X-2000.", "search_knowledge_base", { query: "X-2000 mac os compatibility" }],
            ["product_expert", "Yes, the X-2000 is fully compatible with Mac OS 12 and later.", null, null]
        ]
    },
    {
        task: "Login Error 503",
        user_msg: "I keep getting error 503 when I try to login.",
        flow: [
            ["triage_agent", "I'm sorry you're having trouble logging in. Connecting you to tech support.", "transfer_to_agent", { agent_id: "tech_support" }],
            ["tech_support", "Error 503 usually indicates a server issue. Let me check our systems.", "check_system_status", { service: "auth_service" }],
            ["tech_support", "It looks like our authentication service is experiencing high load. Please try again in 10 minutes.", null, null]
        ]
    },
    {
        task: "Cancel Subscription",
        user_msg: "I want to cancel my subscription immediately.",
        flow: [
            ["triage_agent", "I can help with cancellation. Transferring to billing.", "transfer_to_agent", { agent_id: "refund_specialist" }],
            ["refund_specialist", "I can help you cancel. May I ask why you are leaving?", null, null],
            ["refund_specialist", "I understand. I have processed the cancellation effective today.", "process_refund", { order_id: "sub_999", amount: 0 }]
        ]
    },
    {
        task: "Billing Dispute",
        user_msg: "I was charged twice for my last month.",
        flow: [
            ["triage_agent", "I'll connect you with a billing specialist to resolve this double charge.", "transfer_to_agent", { agent_id: "refund_specialist" }],
            ["refund_specialist", "Let me investigate your transaction history.", "check_eligibility", { order_id: "last_month" }],
            ["refund_specialist", "I see the duplicate charge. I'm reversing one of them now.", "process_refund", { order_id: "tx_777", amount: 19.99 }]
        ]
    },
    {
        task: "Feature Request",
        user_msg: "You should add a dark mode to the app.",
        flow: [
            ["triage_agent", "That's a great suggestion! I'll pass it to our product team.", "transfer_to_agent", { agent_id: "product_expert" }],
            ["product_expert", "Thanks for the feedback! I've logged 'Dark Mode' as a feature request.", "search_knowledge_base", { query: "feature request dark mode" }],
            ["product_expert", "We actually have this on our roadmap for Q3!", null, null]
        ]
    },
    {
        task: "Password Reset",
        user_msg: "I forgot my password and the email link isn't working.",
        flow: [
            ["triage_agent", "I'll get tech support to help you reset your password.", "transfer_to_agent", { agent_id: "tech_support" }],
            ["tech_support", "I can manually trigger a reset for you. What is your username?", null, null],
            ["tech_support", "Sending a temporary password to your registered phone number.", "reset_password", { user_id: "user_123" }]
        ]
    },
    {
        task: "Change Shipping Address",
        user_msg: "I need to change the address for my order #55555.",
        flow: [
            ["triage_agent", "OrderBot can help you update shipping details.", "transfer_to_agent", { agent_id: "order_tracker" }],
            ["order_tracker", "Let me check if order #55555 has shipped yet.", "get_order_status", { order_id: "55555" }],
            ["order_tracker", "It hasn't shipped yet. What is the new address?", null, null],
            ["order_tracker", "Updating address to 123 New St...", "update_shipping", { order_id: "55555", address: "123 New St" }]
        ]
    },
    {
        task: "General Complaint",
        user_msg: "Your service is terrible and slow.",
        flow: [
            ["triage_agent", "I'm very sorry to hear that. I'll connect you with a supervisor (Product Expert) to discuss your concerns.", "transfer_to_agent", { agent_id: "product_expert" }],
            ["product_expert", "I apologize for your bad experience. Can you tell me more about what happened?", null, null],
            ["product_expert", "Thank you for your feedback. We will strive to do better.", null, null]
        ]
    }
];

const conversations = scenarios.map((scenario, i) => {
    const turns: any[] = [];
    // Initial user message
    turns.push({
        role: "user",
        content: scenario.user_msg,
        step: 0,
        timestamp: new Date().toISOString()
    });

    let step_count = 1;
    scenario.flow.forEach((step: any) => {
        const [agent_id, content, tool_name, tool_args] = step;
        const agent_def = agents.find(a => a.id === agent_id);

        const interaction = {
            agent_id: agent_id,
            agent_name: agent_def?.name,
            step_description: `${agent_def?.name} thinking...`,
            thought: tool_name ? `I should ${tool_name.replace('_', ' ')}` : "I should respond to the user.",
            tool_calls: tool_name ? [{ name: tool_name, arguments: tool_args }] : [],
            output: content
        };

        turns.push({
            role: "assistant",
            content: content,
            step: step_count,
            timestamp: new Date().toISOString(),
            agent_interactions: [interaction]
        });
        step_count++;
    });

    return {
        session_id: `conv_${i + 1}`,
        initial_task: scenario.task,
        agents: agents,
        turns: turns,
        total_cost: Number((Math.random() * 0.05).toFixed(4)),
        total_latency_ms: Math.floor(Math.random() * 1500) + 500
    };
});

const dataset = {
    name: "Comprehensive Support Team Dataset",
    task: "Customer Support Automation Evaluation",
    conversations: conversations,
    metadata: { version: "2.0", generator: "script" }
};

console.log(JSON.stringify(dataset, null, 2));
