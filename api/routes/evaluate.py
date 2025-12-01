"""Evaluation routes for ReAsk API"""

import sys
import os
import json
import asyncio

# Add parent directory to path for reask import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Generator, AsyncGenerator

from ..database import get_db, Dataset, Conversation, Message, EvalResult, SessionLocal
from ..schemas import EvalStats

from reask import ReAskDetector, Message as ReAskMessage

router = APIRouter()


def run_evaluation(dataset_id: int, db: Session):
    """Run ReAsk evaluation on a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        return
    
    # Initialize detector
    detector = ReAskDetector(
        ccm_model="gpt-5-nano",
        rdm_model="gpt-5-nano",
        judge_model="gpt-5-mini",
        similarity_threshold=0.66,
        use_llm_confirmation=True,
        use_llm_judge_fallback=True
    )
    
    # Get all conversations
    conversations = db.query(Conversation).filter(
        Conversation.dataset_id == dataset_id
    ).all()
    
    for conv in conversations:
        # Get messages in order
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.index).all()
        
        # Convert to ReAsk messages
        reask_messages = []
        for msg in messages:
            if msg.role == 'user':
                reask_messages.append(ReAskMessage.user(msg.content, knowledge=msg.knowledge))
            else:
                reask_messages.append(ReAskMessage.assistant(msg.content))
        
        # Evaluate conversation
        if len(reask_messages) >= 2:
            results = detector.evaluate_conversation(reask_messages)
            
            # Map results back to database messages
            for idx, result in results:
                # Find the corresponding message (idx is the position in reask_messages)
                if idx < len(messages):
                    db_msg = messages[idx]
                    
                    # Check if eval result already exists
                    existing = db.query(EvalResult).filter(
                        EvalResult.message_id == db_msg.id
                    ).first()
                    
                    if existing:
                        existing.is_bad = result.is_bad
                        existing.detection_type = result.detection_type.value
                        existing.confidence = result.confidence
                        existing.reason = result.reason
                    else:
                        eval_result = EvalResult(
                            message_id=db_msg.id,
                            is_bad=result.is_bad,
                            detection_type=result.detection_type.value,
                            confidence=result.confidence,
                            reason=result.reason
                        )
                        db.add(eval_result)
    
    # Mark dataset as evaluated
    dataset.evaluated = True
    db.commit()


@router.post("/datasets/{dataset_id}/evaluate")
async def evaluate_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Run ReAsk evaluation on a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    # Run evaluation synchronously (no celery as per requirements)
    run_evaluation(dataset_id, db)
    
    # Get stats
    eval_results = db.query(EvalResult).join(Message).join(Conversation).filter(
        Conversation.dataset_id == dataset_id
    ).all()
    
    if not eval_results:
        return {"message": "No responses to evaluate", "stats": None}
    
    total = len(eval_results)
    bad = sum(1 for r in eval_results if r.is_bad)
    good = total - bad
    ccm = sum(1 for r in eval_results if r.detection_type == 'ccm')
    rdm = sum(1 for r in eval_results if r.detection_type == 'rdm')
    llm = sum(1 for r in eval_results if r.detection_type == 'llm_judge')
    hallucination = sum(1 for r in eval_results if r.detection_type == 'hallucination')
    avg_conf = sum(r.confidence for r in eval_results) / total if total > 0 else 0
    
    stats = EvalStats(
        total_responses=total,
        good_responses=good,
        bad_responses=bad,
        ccm_detections=ccm,
        rdm_detections=rdm,
        llm_judge_detections=llm,
        hallucination_detections=hallucination,
        avg_confidence=avg_conf
    )
    
    return {"message": "Evaluation complete", "stats": stats}


@router.get("/datasets/{dataset_id}/evaluate/stream")
async def evaluate_dataset_stream(dataset_id: int):
    """Run ReAsk evaluation on a dataset with SSE streaming progress"""
    
    async def event_generator() -> AsyncGenerator[str, None]:
        # Create a new DB session for this generator
        db = SessionLocal()
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Dataset not found'})}\n\n"
                return
            
            # Initialize detector
            detector = ReAskDetector(
                ccm_model="gpt-5-nano",
                rdm_model="gpt-5-nano",
                judge_model="gpt-5-mini",
                similarity_threshold=0.5,
                use_llm_confirmation=True,
                use_llm_judge_fallback=True
            )
            
            # Get all conversations
            conversations = db.query(Conversation).filter(
                Conversation.dataset_id == dataset_id
            ).all()
            
            total_convs = len(conversations)
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'start', 'total': total_convs})}\n\n"
            
            for conv_index, conv in enumerate(conversations):
                # Get messages in order
                messages = db.query(Message).filter(
                    Message.conversation_id == conv.id
                ).order_by(Message.index).all()
                
                # Convert to ReAsk messages
                reask_messages = []
                for msg in messages:
                    if msg.role == 'user':
                        reask_messages.append(ReAskMessage.user(msg.content, knowledge=msg.knowledge))
                    else:
                        reask_messages.append(ReAskMessage.assistant(msg.content))
                
                # Evaluate conversation
                conv_results = []
                if len(reask_messages) >= 2:
                    results = detector.evaluate_conversation(reask_messages)
                    
                    # Map results back to database messages
                    for idx, result in results:
                        if idx < len(messages):
                            db_msg = messages[idx]
                            
                            # Check if eval result already exists
                            existing = db.query(EvalResult).filter(
                                EvalResult.message_id == db_msg.id
                            ).first()
                            
                            if existing:
                                existing.is_bad = result.is_bad
                                existing.detection_type = result.detection_type.value
                                existing.confidence = result.confidence
                                existing.reason = result.reason
                            else:
                                eval_result = EvalResult(
                                    message_id=db_msg.id,
                                    is_bad=result.is_bad,
                                    detection_type=result.detection_type.value,
                                    confidence=result.confidence,
                                    reason=result.reason
                                )
                                db.add(eval_result)
                            
                            conv_results.append({
                                'message_id': db_msg.id,
                                'is_bad': result.is_bad,
                                'detection_type': result.detection_type.value,
                                'confidence': result.confidence,
                                'reason': result.reason
                            })
                
                db.commit()
                
                # Yield progress update
                progress_data = {
                    'type': 'progress',
                    'conversation_id': conv.conversation_id,
                    'conversation_db_id': conv.id,
                    'current': conv_index + 1,
                    'total': total_convs,
                    'results': conv_results
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                # Small delay to allow browser to process
                await asyncio.sleep(0.01)
            
            # Mark dataset as evaluated
            dataset.evaluated = True
            db.commit()
            
            # Calculate final stats
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
                hallucination = sum(1 for r in eval_results if r.detection_type == 'hallucination')
                avg_conf = sum(r.confidence for r in eval_results) / total if total > 0 else 0
                
                complete_data = {
                    'type': 'complete',
                    'stats': {
                        'total_responses': total,
                        'good_responses': good,
                        'bad_responses': bad,
                        'ccm_detections': ccm,
                        'rdm_detections': rdm,
                        'llm_judge_detections': llm,
                        'hallucination_detections': hallucination,
                        'avg_confidence': avg_conf
                    }
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'complete', 'stats': None})}\n\n"
        finally:
            db.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/datasets/{dataset_id}/results")
async def get_results(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Get evaluation results for a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    
    if not dataset.evaluated:
        return {"evaluated": False, "results": [], "stats": None}
    
    # Get all eval results with message info
    results = db.query(
        EvalResult, Message, Conversation
    ).join(Message).join(Conversation).filter(
        Conversation.dataset_id == dataset_id
    ).all()
    
    result_list = []
    for eval_res, msg, conv in results:
        result_list.append({
            "conversation_id": conv.conversation_id,
            "message_index": msg.index,
            "role": msg.role,
            "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
            "is_bad": eval_res.is_bad,
            "detection_type": eval_res.detection_type,
            "confidence": eval_res.confidence,
            "reason": eval_res.reason
        })
    
    # Calculate stats
    total = len(results)
    bad = sum(1 for r in results if r[0].is_bad)
    good = total - bad
    
    return {
        "evaluated": True,
        "results": result_list,
        "stats": {
            "total": total,
            "good": good,
            "bad": bad
        }
    }

