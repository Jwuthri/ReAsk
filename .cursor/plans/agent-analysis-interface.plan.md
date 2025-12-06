<!-- 31fa4e20-eb87-4f80-897a-9edfb8999d94 9c67f92c-5556-4b7a-8dcb-571392f0cf9d -->
# Agent Analysis Interface Redesign

## New Layout Structure

**Split-View Architecture:**

```
+--------------------------------------------------+
|   Global Score Hero (Overall + Quick Stats)      |
+--------------------------------------------------+
|  Agent Tree      |   Detail Panel                |
|  +-----------+   |   +----------------------+    |
|  | Global    |   |   | Selected View        |    |
|  | └ Agent1  |   |   | - Metrics Grid       |    |
|  |   └ Turn1 |   |   | - Score Breakdown    |    |
|  |   └ Turn2 |   |   | - Issues/Recs        |    |
|  | └ Agent2  |   |   +----------------------+    |
|  |   └ Turn1 |   |                              |
|  +-----------+   |   Timeline/Conversation      |
+--------------------------------------------------+
```

## Key Changes

### 1. Global Score Hero (Top Section)

- Large circular score gauge (keep current style)
- Quick metric pills: Good/Bad responses, Efficiency, Drift, Coordination
- Cleaner horizontal layout with better spacing

### 2. Agent Hierarchy Tree (Left Panel)

- Collapsible tree structure:
  - **Global** (root) - shows overall metrics
  - **Agent nodes** - each agent with score badge
  - **Turn nodes** - individual turns per agent with mini score
- Click to select and show details in right panel
- Color-coded by score (green/orange/red indicators)

### 3. Detail Panel (Right Panel)

Context-aware based on tree selection:

- **Global selected**: All-agent summary, coordination score, detection method breakdown
- **Agent selected**: Agent-specific metrics (overall, tool use, reasoning, handoff), issues, recommendations
- **Turn selected**: Full step breakdown with all scores, reasoning chain, tool calls, detection details

### 4. Step-Level Score Breakdown (Expandable)

When viewing a turn/step:

- Detection type badge (CCM/RDM/LLM_JUDGE/HALLUCINATION)
- Confidence score with visual gauge
- Full reasoning explanation
- If multi-agent: show which agent responded and their individual contribution score

### 5. Conversation Timeline (Bottom)

- Simplified timeline showing conversation flow
- Each turn shows: user message preview + score badge + agent(s) involved
- Click expands to show full content in detail panel

## Files to Modify

1. [`web/components/AgentTraceViewer.tsx`](web/components/AgentTraceViewer.tsx) - Complete restructure to split-view
2. [`web/components/AgentTraceViewer.module.css`](web/components/AgentTraceViewer.module.css) - New styles for split layout
3. May extract new components:

   - `AgentTree.tsx` - Left panel tree component
   - `DetailPanel.tsx` - Right panel detail view

## Visual Style

- Keep current dark theme and accent colors
- Cleaner card borders with subtle depth
- Better hierarchy through font sizes and spacing
- Interactive tree with smooth expand/collapse animations

### To-dos

- [ ] Create AgentTree component with collapsible hierarchy (Global > Agents > Turns)
- [ ] Create DetailPanel component with context-aware views (global/agent/turn)
- [ ] Refactor AgentTraceViewer to use split-view layout with new components
- [ ] Implement expandable step-level score breakdown with full metric details
- [ ] Update CSS module with split-view layout, tree styles, and detail panel styles