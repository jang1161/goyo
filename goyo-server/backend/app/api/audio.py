"""
Audio Control API
ANC 제어 및 모니터링 (오디오 스트리밍은 AI Server에서 직접 처리)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import asyncio
import logging
import json

from app.database import get_db
from app.utils.dependencies import get_current_user
from app.utils.redis_client import get_redis_client
from app.models.user import User
from app.models.device import Device

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/start")
async def start_audio_stream(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    ANC 시작 명령
    - 클라이언트는 이 API 호출 후 AI Server에 직접 WebSocket 연결
    """
    # 디바이스 구성 확인
    devices = db.query(Device).filter(Device.user_id == current_user.id).all()
    
    source_device = next((d for d in devices if d.device_type == "microphone_source"), None)
    reference_device = next((d for d in devices if d.device_type == "microphone_reference"), None)
    speaker = next((d for d in devices if d.device_type == "speaker"), None)
    
    if not all([source_device, reference_device, speaker]):
        return {
            "success": False,
            "error": "Device setup incomplete"
        }
    
    # Redis에 ANC 시작 명령 전송 (AI Server가 수신)
    redis_client = await get_redis_client()
    await redis_client.publish(
        "anc:control",
        json.dumps({
            "user_id": str(current_user.id),
            "command": "start",
            "params": {
                "suppression_level": current_user.anc_suppression_level
            }
        })
    )
    
    return {
        "success": True,
        "message": "ANC started. Connect to AI Server WebSocket.",
        "ai_server_url": f"ws://localhost:8001/ws/audio/{current_user.id}",
        "source_device": source_device.device_name,
        "reference_device": reference_device.device_name,
        "speaker": speaker.device_name
    }


@router.post("/stop")
async def stop_audio_stream(
    current_user: User = Depends(get_current_user)
):
    """ANC 중지 명령"""
    
    # Redis에 ANC 중지 명령 전송
    redis_client = await get_redis_client()
    await redis_client.publish(
        "anc:control",
        json.dumps({
            "user_id": str(current_user.id),
            "command": "stop"
        })
    )
    
    return {
        "success": True,
        "message": "ANC stopped"
    }


@router.websocket("/ws/monitor")
async def monitor_websocket(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    """
    모니터링용 WebSocket
    AI Server로부터 받은 ANC 결과를 클라이언트에 전송
    """
    await websocket.accept()
    
    try:
        # 첫 메시지로 user_id 받기
        auth_message = await websocket.receive_json()
        user_id = auth_message.get("user_id")  # TODO: JWT 검증
        
        logger.info(f"📊 Monitor WebSocket connected: user {user_id}")
        
        # Redis Pub/Sub으로 ANC 결과 수신
        redis_client = await get_redis_client()
        pubsub = redis_client.client.pubsub()
        await pubsub.subscribe("anc:result")
        
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            
            data = json.loads(message["data"])
            
            # 해당 사용자 데이터만 전송
            if data.get("user_id") == str(user_id):
                await websocket.send_json({
                    "timestamp": data.get("timestamp"),
                    "noise_level_db": data.get("noise_level_db"),
                    "reduction_db": data.get("reduction_db"),
                    "status": data.get("status")
                })
    
    except WebSocketDisconnect:
        logger.info(f"📊 Monitor WebSocket disconnected: user {user_id}")
    
    except Exception as e:
        logger.error(f"❌ Monitor WebSocket error: {e}")
        await websocket.close()