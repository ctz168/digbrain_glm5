"""
WebSocket模块
实现实时双向通信
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    host: str = "0.0.0.0"
    port: int = 8001
    max_connections: int = 100
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_message_size: int = 1024 * 1024  # 1MB


class WebSocketHandler:
    """
    WebSocket处理器
    
    处理WebSocket连接和消息
    """
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or WebSocketConfig()
        self._brain = None
        self._connections: Set[Any] = set()
        self._sessions: Dict[str, Dict] = {}
        
        # 统计
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_received": 0,
            "messages_sent": 0
        }
    
    async def initialize(self, brain: Any) -> None:
        """初始化"""
        self._brain = brain
        logger.info("WebSocket handler initialized")
    
    async def on_connect(self, websocket: Any) -> None:
        """连接事件"""
        if len(self._connections) >= self.config.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            return
        
        self._connections.add(websocket)
        self._stats["total_connections"] += 1
        self._stats["active_connections"] = len(self._connections)
        
        # 创建会话
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "websocket": websocket,
            "created_at": time.time()
        }
        
        # 发送会话ID
        await self._send_message(websocket, {
            "type": "connected",
            "session_id": session_id
        })
        
        logger.info(f"WebSocket connected: {session_id}")
    
    async def on_disconnect(self, websocket: Any) -> None:
        """断开事件"""
        self._connections.discard(websocket)
        self._stats["active_connections"] = len(self._connections)
        
        # 清理会话
        session_to_remove = None
        for session_id, session in self._sessions.items():
            if session["websocket"] == websocket:
                session_to_remove = session_id
                break
        
        if session_to_remove:
            del self._sessions[session_to_remove]
            logger.info(f"WebSocket disconnected: {session_to_remove}")
    
    async def on_message(self, websocket: Any, message: str) -> None:
        """消息事件"""
        self._stats["messages_received"] += 1
        
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            if message_type == "process":
                await self._handle_process(websocket, data)
            elif message_type == "stream":
                await self._handle_stream(websocket, data)
            elif message_type == "memory_search":
                await self._handle_memory_search(websocket, data)
            elif message_type == "ping":
                await self._send_message(websocket, {"type": "pong"})
            else:
                await self._send_message(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
        except json.JSONDecodeError:
            await self._send_message(websocket, {
                "type": "error",
                "message": "Invalid JSON"
            })
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await self._send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
    
    async def _handle_process(self, websocket: Any, data: Dict) -> None:
        """处理普通请求"""
        input_text = data.get("input", "")
        session_id = data.get("session_id")
        
        result = []
        async for chunk in self._brain.process(
            input_text,
            session_id=session_id,
            stream=False
        ):
            result.append(chunk)
        
        await self._send_message(websocket, {
            "type": "result",
            "output": "".join(result)
        })
    
    async def _handle_stream(self, websocket: Any, data: Dict) -> None:
        """处理流式请求"""
        input_text = data.get("input", "")
        session_id = data.get("session_id")
        
        async for chunk in self._brain.process(
            input_text,
            session_id=session_id,
            stream=True
        ):
            await self._send_message(websocket, {
                "type": "chunk",
                "content": chunk
            })
        
        await self._send_message(websocket, {
            "type": "done"
        })
    
    async def _handle_memory_search(self, websocket: Any, data: Dict) -> None:
        """处理记忆搜索"""
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not self._brain or not self._brain._memory_system:
            await self._send_message(websocket, {
                "type": "error",
                "message": "Memory system not available"
            })
            return
        
        results = await self._brain._memory_system.retrieve(query, top_k=top_k)
        
        await self._send_message(websocket, {
            "type": "memory_results",
            "results": results
        })
    
    async def _send_message(self, websocket: Any, data: Dict) -> None:
        """发送消息"""
        try:
            await websocket.send(json.dumps(data))
            self._stats["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Send message error: {e}")
    
    async def broadcast(self, data: Dict) -> None:
        """广播消息"""
        for websocket in self._connections:
            await self._send_message(websocket, data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "sessions": len(self._sessions)
        }


class WebSocketServer:
    """
    WebSocket服务器
    """
    
    def __init__(
        self,
        handler: Optional[WebSocketHandler] = None,
        config: Optional[WebSocketConfig] = None
    ):
        self.config = config or WebSocketConfig()
        self.handler = handler or WebSocketHandler(self.config)
        self._server = None
    
    async def start(self) -> None:
        """启动服务器"""
        try:
            import websockets
            
            async def connection_handler(websocket, path):
                await self.handler.on_connect(websocket)
                try:
                    async for message in websocket:
                        await self.handler.on_message(websocket, message)
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    await self.handler.on_disconnect(websocket)
            
            self._server = await websockets.serve(
                connection_handler,
                self.config.host,
                self.config.port,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                max_size=self.config.max_message_size
            )
            
            logger.info(f"WebSocket server started on ws://{self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("websockets not installed, WebSocket server not started")
    
    async def stop(self) -> None:
        """停止服务器"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("WebSocket server stopped")
