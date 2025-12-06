# Database Schema Refactor

## Objective
Refactor the database to have a clean, simple schema with proper separation between:
- **Dataset**: The data uploaded/received from frontend
- **Analysis**: The evaluation/analysis run on top of a dataset

## Tasks

- [x] Understand current database schema and issues
- [x] Design new simplified schema  
- [x] Create implementation plan for refactor
- [x] Get user approval on the plan
- [x] Refactor database models
  - [x] Create new `database.py` with clean schema
  - [x] Dataset → Conversation → Message → Step hierarchy
  - [x] Agent table (shared across dataset)
  - [x] Analysis → ConversationResult, MessageResult, StepResult
- [x] Update backend API routes
  - [x] Remove unused `datasets.py` and `evaluate.py` routes
  - [x] Update `main.py` 
- [x] Update frontend to match new API (backwards-compatible aliases used)
- [x] Test the complete flow

