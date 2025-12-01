# ReAsk Web Application

A modern web interface for evaluating LLM conversation datasets using the ReAsk detection methods.

## Quick Start

### 1. Install Dependencies

```bash
# Backend (from project root)
pip install -r requirements.txt

# Frontend
cd web && npm install
```

### 2. Start the API Server

```bash
# From project root
uvicorn api.main:app --reload --port 8000
```

API will be available at `http://localhost:8000`

### 3. Start the Web App

```bash
# From web directory
cd web && npm run dev
```

Web app will be available at `http://localhost:3000`

## Dataset Format

### CSV Format

Required columns:
- `conversation_id` - Groups messages into conversations
- `message_index` - Order within conversation (0-based)
- `role` - Either `user` or `assistant`
- `content` - The message text

Example:
```csv
conversation_id,message_index,role,content
conv_1,0,user,Hello how are you?
conv_1,1,assistant,I'm doing great!
conv_1,2,user,What's the weather like?
```

### JSON Format

```json
{
  "conversations": [
    {
      "id": "unique_id",
      "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
      ]
    }
  ]
}
```

## Sample Data

Sample datasets are provided in the `samples/` directory:
- `sample.csv` - CSV format example
- `sample.json` - JSON format example

## Features

- **Drag & Drop Upload** - Upload CSV or JSON conversation datasets
- **ReAsk Evaluation** - Run CCM, RDM, and LLM Judge detection
- **Results Visualization** - See good/bad responses with confidence scores
- **Detection Breakdown** - View which detection methods found issues
- **Conversation Browser** - Expand and explore individual conversations

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/datasets/upload` | Upload a dataset file |
| GET | `/api/datasets` | List all datasets |
| GET | `/api/datasets/{id}` | Get dataset details |
| DELETE | `/api/datasets/{id}` | Delete a dataset |
| POST | `/api/datasets/{id}/evaluate` | Run ReAsk evaluation |
| GET | `/api/datasets/{id}/results` | Get evaluation results |

## Environment Variables

Make sure you have an OpenAI API key set:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

