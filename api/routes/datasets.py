"""Dataset routes for ReAsk API"""

import csv
import json
import io
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database import get_db, Dataset, Conversation, Message, EvalResult
from ..schemas import (
    DatasetResponse, DatasetDetail, DatasetWithStats, 
    UploadResponse, EvalStats, ConversationWithEvals, MessageWithEval
)

router = APIRouter()


def parse_csv(content: str) -> dict:
    """Parse CSV content into conversations dict"""
    reader = csv.DictReader(io.StringIO(content))
    
    # Validate required columns
    required_cols = {'conversation_id', 'message_index', 'role', 'content'}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        missing = required_cols - set(reader.fieldnames or [])
        raise ValueError(f"Missing required columns: {missing}")
    
    conversations = {}
    for row in reader:
        conv_id = row['conversation_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append({
            'index': int(row['message_index']),
            'role': row['role'].lower(),
            'content': row['content']
        })
    
    # Sort messages by index
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda m: m['index'])
    
    return conversations


def parse_json(content: str) -> dict:
    """Parse JSON content into conversations dict"""
    data = json.loads(content)
    
    if 'conversations' not in data:
        raise ValueError("JSON must have 'conversations' array")
    
    conversations = {}
    for conv in data['conversations']:
        conv_id = conv.get('id', str(len(conversations)))
        messages = []
        for i, msg in enumerate(conv.get('messages', [])):
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Message missing 'role' or 'content' in conversation {conv_id}")
            messages.append({
                'index': i,
                'role': msg['role'].lower(),
                'content': msg['content']
            })
        conversations[conv_id] = messages
    
    return conversations


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a CSV or JSON dataset"""
    # Determine file type
    filename = file.filename or "unknown"
    if filename.endswith('.csv'):
        file_type = 'csv'
    elif filename.endswith('.json'):
        file_type = 'json'
    else:
        raise HTTPException(400, "File must be .csv or .json")
    
    # Read and parse content
    content = (await file.read()).decode('utf-8')
    
    try:
        if file_type == 'csv':
            conversations = parse_csv(content)
        else:
            conversations = parse_json(content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    if not conversations:
        raise HTTPException(400, "No conversations found in file")
    
    # Create dataset
    dataset = Dataset(
        name=filename.rsplit('.', 1)[0],
        file_type=file_type
    )
    db.add(dataset)
    db.flush()
    
    # Create conversations and messages
    total_messages = 0
    for conv_id, messages in conversations.items():
        conv = Conversation(
            dataset_id=dataset.id,
            conversation_id=conv_id
        )
        db.add(conv)
        db.flush()
        
        for msg in messages:
            db_msg = Message(
                conversation_id=conv.id,
                index=msg['index'],
                role=msg['role'],
                content=msg['content']
            )
            db.add(db_msg)
            total_messages += 1
    
    db.commit()
    db.refresh(dataset)
    
    return UploadResponse(
        id=dataset.id,
        name=dataset.name,
        file_type=dataset.file_type,
        conversations_imported=len(conversations),
        messages_imported=total_messages
    )


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(db: Session = Depends(get_db)):
    """List all datasets"""
    datasets = db.query(Dataset).order_by(Dataset.uploaded_at.desc()).all()
    
    result = []
    for ds in datasets:
        conv_count = db.query(func.count(Conversation.id)).filter(
            Conversation.dataset_id == ds.id
        ).scalar()
        
        msg_count = db.query(func.count(Message.id)).join(Conversation).filter(
            Conversation.dataset_id == ds.id
        ).scalar()
        
        result.append(DatasetResponse(
            id=ds.id,
            name=ds.name,
            file_type=ds.file_type,
            uploaded_at=ds.uploaded_at,
            evaluated=ds.evaluated,
            conversation_count=conv_count,
            message_count=msg_count
        ))
    
    return result


@router.get("/{dataset_id}", response_model=DatasetWithStats)
async def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get dataset with conversations and evaluation results"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    # Get conversations with messages and eval results
    conversations = db.query(Conversation).filter(
        Conversation.dataset_id == dataset_id
    ).all()
    
    conv_count = len(conversations)
    msg_count = 0
    
    conv_responses = []
    for conv in conversations:
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.index).all()
        
        msg_responses = []
        for msg in messages:
            msg_count += 1
            eval_result = db.query(EvalResult).filter(
                EvalResult.message_id == msg.id
            ).first()
            
            msg_responses.append(MessageWithEval(
                id=msg.id,
                index=msg.index,
                role=msg.role,
                content=msg.content,
                eval_result=eval_result
            ))
        
        conv_responses.append(ConversationWithEvals(
            id=conv.id,
            conversation_id=conv.conversation_id,
            messages=msg_responses
        ))
    
    # Calculate stats if evaluated
    stats = None
    if dataset.evaluated:
        eval_results = db.query(EvalResult).join(Message).join(Conversation).filter(
            Conversation.dataset_id == dataset_id
        ).all()
        
        if eval_results:
            total = len(eval_results)
            bad = sum(1 for r in eval_results if r.is_bad)
            good = total - bad
            ccm = sum(1 for r in eval_results if r.detection_type == 'ccm')
            rdm = sum(1 for r in eval_results if r.detection_type == 'rdm')
            llm = sum(1 for r in eval_results if r.detection_type == 'llm_judge')
            avg_conf = sum(r.confidence for r in eval_results) / total if total > 0 else 0
            
            stats = EvalStats(
                total_responses=total,
                good_responses=good,
                bad_responses=bad,
                ccm_detections=ccm,
                rdm_detections=rdm,
                llm_judge_detections=llm,
                avg_confidence=avg_conf
            )
    
    return DatasetWithStats(
        id=dataset.id,
        name=dataset.name,
        file_type=dataset.file_type,
        uploaded_at=dataset.uploaded_at,
        evaluated=dataset.evaluated,
        conversation_count=conv_count,
        message_count=msg_count,
        conversations=conv_responses,
        stats=stats
    )


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Delete a dataset and all its data"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    db.delete(dataset)
    db.commit()
    
    return {"message": "Dataset deleted", "id": dataset_id}

