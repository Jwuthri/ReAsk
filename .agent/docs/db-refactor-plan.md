# Database Schema Refactor - From Chaos to Clean

## Problem Statement

The current database schema has major issues:

1. **Confusing naming**: `Message`, `MessageAnalysisResult`, `InteractionAnalysisResult` - what does any of this mean?
2. **Empty/unused tables**: `datasets`, `dataset_analyses`, `conversations` are empty because the FE sends data directly to the agent evaluation endpoints
3. **Two disconnected flows**:
   - **Dataset flow** (file upload): `Dataset` → `Conversation` → `Message` → `MessageAnalysisResult`
   - **Agent flow** (API payload): `AgentSession` → `SessionTurn` → `AgentInteraction` → `AgentStep` + various analysis results
4. **No unified concept**: No clear "Dataset → Analysis" relationship

## User Review Required

> [!IMPORTANT]
> Before I implement, please confirm:
> 1. Do you want to **merge** the two flows into one? (FE sends data → stored as Dataset → analyze)
> 2. Or do you want to **keep them separate** but clean up naming?
> 3. What's the primary use case - file uploads (CSV/JSON) or direct API payloads?

---

## Current Schema (The Mess)

```mermaid
erDiagram
    %% File Upload Flow (UNUSED)
    Dataset ||--o{ Conversation : contains
    Dataset ||--o{ DatasetAnalysis : has
    Conversation ||--o{ Message : contains
    Message ||--o| MessageAnalysisResult : has
    DatasetAnalysis ||--o{ MessageAnalysisResult : produces
    
    %% Agent API Flow (ACTUALLY USED)
    AgentSession ||--o{ AgentDefinition : defines
    AgentSession ||--o{ SessionTurn : contains
    AgentSession ||--o{ AgentAnalysis : has
    SessionTurn ||--o{ AgentInteraction : contains
    SessionTurn ||--o{ TurnAnalysisResult : has
    AgentInteraction ||--o{ AgentStep : contains
    AgentInteraction ||--o{ InteractionAnalysisResult : has
```

---

## Proposed New Schema (Clean & Simple)

The key insight: **Everything is a Dataset. Analysis runs on Datasets.**

```mermaid
erDiagram
    Dataset ||--o{ Turn : contains
    Dataset ||--o{ Analysis : has
    Dataset ||--o{ Agent : defines
    
    Turn ||--o{ Interaction : contains
    Turn ||--o{ TurnResult : has
    
    Interaction ||--o{ Step : contains
    Interaction ||--o{ InteractionResult : has
    
    Analysis ||--o{ TurnResult : produces
    Analysis ||--o{ InteractionResult : produces
    
    Dataset {
        int id PK
        string name
        string source "file_upload or api"
        string task "initial task/goal"
        datetime created_at
        json metadata
    }
    
    Agent {
        int id PK
        int dataset_id FK
        string agent_id "e.g. planner"
        string name
        string role
        json tools_available
    }
    
    Turn {
        int id PK
        int dataset_id FK
        int turn_index
        string user_message
        string final_response
    }
    
    Interaction {
        int id PK
        int turn_id FK
        int agent_id FK "which agent"
        int sequence
        string agent_response
        json tool_execution
    }
    
    Step {
        int id PK
        int interaction_id FK
        int step_index
        string step_type "thought, tool_call, observation"
        string content
        string tool_name
        json tool_params
        string tool_result
    }
    
    Analysis {
        int id PK
        int dataset_id FK
        string status "pending, running, completed, failed"
        json analysis_types
        datetime started_at
        datetime completed_at
        float overall_score
        json results_summary
    }
    
    TurnResult {
        int id PK
        int analysis_id FK
        int turn_id FK
        boolean is_bad
        string detection_type
        float confidence
        string reason
    }
    
    InteractionResult {
        int id PK
        int analysis_id FK
        int interaction_id FK
        json tool_use_score
        json reasoning_score
        json handoff_score
    }
```

---

## Proposed Changes

### Backend (`api/database.py`)

#### [DELETE] Old tables to remove:
- `Conversation` - merged into Turn concept
- `Message` - too specific to file upload, replaced by Turn/Interaction
- `MessageAnalysisResult` - replaced by unified TurnResult
- `DatasetAnalysis` - merged into Analysis

#### [MODIFY] Tables to rename/simplify:
| Old Name | New Name | Notes |
|----------|----------|-------|
| `AgentSession` | `Dataset` | The core data container |
| `AgentDefinition` | `Agent` | Simpler name |
| `SessionTurn` | `Turn` | Simpler name |
| `AgentInteraction` | `Interaction` | Simpler name |
| `AgentStep` | `Step` | Simpler name |
| `AgentAnalysis` | `Analysis` | Simpler name |
| `TurnAnalysisResult` | `TurnResult` | Simpler name |
| `InteractionAnalysisResult` | `InteractionResult` | Simpler name |

#### [MODIFY] [database.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/database.py)
- Rename all models as shown above
- Add `source` field to Dataset ("file_upload" | "api_payload")
- Consolidate the two flows into one

---

### Backend Routes

#### [DELETE] [datasets.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/routes/datasets.py)
- Remove the separate file upload flow
- Or: Keep but make it create a `Dataset` in the new unified schema

#### [MODIFY] [agent_eval.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/routes/agent_eval.py)
- Update all references to use new model names
- Simplify the API surface

---

### Frontend

#### [MODIFY] `web/app/agent/page.tsx`
- Update API types to match new schema
- Update any references to old field names

#### [MODIFY] `web/lib/api.ts` (if accessible)
- Update type definitions
- Update API endpoints

---

## Verification Plan

### Manual Testing
1. Start the backend: `cd api && uvicorn main:app --reload`
2. Start the frontend: `cd web && npm run dev`
3. Test the following flows:
   - Load an example trace in the Agent page
   - Run analysis on it
   - Verify the data is stored correctly in SQLite
   - Check that results display properly

### Database Verification
```bash
sqlite3 reask_data.db "SELECT name FROM sqlite_master WHERE type='table';"
```
Should show the new simplified table names.

---

## Questions for You

1. **Do you want me to keep the file upload feature?** (CSV/JSON → Dataset)
2. **Should I create a migration script** for any existing data, or start fresh?
3. **Any specific naming preferences** for the new tables?
