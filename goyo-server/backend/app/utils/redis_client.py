import redis
from app.config import settings
from typing import Optional
import json

class RedisClient:
    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    def set_device_status(self, device_id: str, status: dict, expire: int = 300):
        '''디바이스 상태를 Redis에 저장 (5분 TTL)'''
        key = f"device:status:{device_id}"
        self.client.setex(key, expire, json.dumps(status))
    
    def get_device_status(self, device_id: str) -> Optional[dict]:
        '''디바이스 상태 조회'''
        key = f"device:status:{device_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None
    
    def delete_device_status(self, device_id: str):
        '''디바이스 상태 삭제'''
        key = f"device:status:{device_id}"
        self.client.delete(key)
    
    def set_audio_buffer(self, mic_type: str, audio_data: bytes, expire: int = 1):
        '''실시간 오디오 버퍼 저장 (1초 TTL)'''
        key = f"audio:buffer:{mic_type}"  # source or reference
        self.client.setex(key, expire, audio_data)
    
    def get_audio_buffer(self, mic_type: str) -> Optional[bytes]:
        '''오디오 버퍼 조회'''
        key = f"audio:buffer:{mic_type}"
        return self.client.get(key)
    
    def set_user_session(self, user_id: int, session_data: dict, expire: int = 3600):
        '''사용자 세션 저장 (1시간 TTL)'''
        key = f"user:session:{user_id}"
        self.client.setex(key, expire, json.dumps(session_data))
    
    def get_user_session(self, user_id: int) -> Optional[dict]:
        '''사용자 세션 조회'''
        key = f"user:session:{user_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

redis_client = RedisClient()