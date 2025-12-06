# Database Schema Refactor - Complete

## What Changed

### Before (14 confusing tables)
- `AgentSession`, `AgentDefinition`, `SessionTurn`, `AgentInteraction`, `AgentStep`
- `AgentAnalysis`, `TurnAnalysisResult`, `InteractionAnalysisResult`
- Empty unused: `Dataset`, `Conversation`, `Message`, `DatasetAnalysis`, `MessageAnalysisResult`

### After (9 clean tables)

```
Dataset
├── agents (shared across dataset)
├── conversations (was turns)
│   └── messages (was interactions)
│       └── steps
└── analyses
    ├── conversation_results
    ├── message_results
    └── step_results
```

## Key Column Renames

| Old Name | New Name |
|----------|----------|
| `initial_task` | `task` |
| `session_id` | `dataset_id` |
| `turn_index` | `conversation_index` |
| `user_message` | `user_input` |
| `turn_id` | `conversation_id` |
| `interaction_id` | `message_id` |
| `agent_response` | `content` |
| `turns` (relationship) | `conversations` |
| `interactions` (relationship) | `messages` |

## Files Changed

| File | Action |
|------|--------|
| [database.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/database.py) | New clean schema with JSON columns |
| [agent_eval.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/routes/agent_eval.py) | Updated all save/query functions |
| [main.py](file:///Users/julienwuthrich/GitHub/ReAsk/api/main.py) | Removed unused route imports |
| datasets.py | **Deleted** |
| evaluate.py | **Deleted** |

## Verification

```bash
# API starts successfully
python -c "from api.main import app; print('OK')"
# Output: OK

# No session_id references remain
grep -n "session_id" api/routes/agent_eval.py
# Output: (empty)

# Tables created
sqlite3 reask_data.db ".tables"
# agents, analyses, conversation_results, conversations, datasets,
# message_results, messages, step_results, steps
```

## Notes
- JSON columns use SQLAlchemy `JSON` type (stored as TEXT in SQLite)
- Backwards-compatible aliases maintained for smooth transition
- Frontend unchanged (API response format preserved)
