## How to run backend & AI server
```
docker-compose up
```

## Swagger UI
backend - http://localhost:8000/docs#/
AI - http://localhost:8001/docs#/

## Server Architecture
```
┌─────────────────┐
│   Client App    │  (Captures device mic audio)
│  (Flutter/Web)  │
└────┬────────┬───┘
     │        │
     │        └──────────────────────────────┐
     │ HTTP (Auth, Device, Profile)          │ WebSocket (Audio Stream)
     ↓                                       ↓
┌─────────────────────┐     Redis Pub/Sub     ┌──────────────────────┐
│  FastAPI Backend    │◄───────────────────-─►│   AI Server          │
│  (Port 8000)        │  anc:control          │   (Port 8001)        │
│                     │  anc:result           │                      │
│  - Auth (JWT)       │  device:status        │  - Audio Processing  │
│  - Device CRUD      │                       │  - ANC Algorithm     │
│  - Profile API      │                       │  - Noise Detection   │
│  - Metadata Only    │                       │  - Signal Generation │
└─────────┬───────────┘                       └────────┬─────────────┘
          │                                            │
          ↓                                            ↓
    PostgreSQL                                   MQTT Broker
    (User/Device Data)                                ↓
                                                [Wi-Fi Speaker]
```
