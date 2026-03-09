"""
REST API模块
实现HTTP API接口
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS
    cors_origins: List[str] = None
    
    # 限流
    rate_limit: int = 100
    rate_window: int = 60
    
    # 超时
    request_timeout: float = 60.0
    
    # 认证
    api_key: Optional[str] = None


class APIServer:
    """
    API服务器
    
    提供REST API接口
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._brain = None
        self._routes: Dict[str, Any] = {}
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
    
    async def initialize(self, brain: Any) -> None:
        """初始化"""
        self._brain = brain
        self._setup_routes()
        logger.info(f"API Server initialized on {self.config.host}:{self.config.port}")
    
    def _setup_routes(self) -> None:
        """设置路由"""
        self._routes = {
            "/api/process": self._handle_process,
            "/api/stream": self._handle_stream,
            "/api/memory": self._handle_memory,
            "/api/memory/search": self._handle_memory_search,
            "/api/tools": self._handle_tools,
            "/api/wiki": self._handle_wiki,
            "/api/status": self._handle_status,
            "/api/stats": self._handle_stats
        }
    
    async def start(self) -> None:
        """启动服务器"""
        # 使用aiohttp或fastapi
        try:
            from aiohttp import web
            app = await self._create_aiohttp_app()
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner,
                self.config.host,
                self.config.port
            )
            await self._site.start()
            logger.info(f"API Server started on http://{self.config.host}:{self.config.port}")
        except ImportError:
            logger.warning("aiohttp not installed, using simple server")
    
    async def _create_aiohttp_app(self):
        """创建aiohttp应用"""
        from aiohttp import web
        
        app = web.Application()
        
        # CORS中间件
        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == "OPTIONS":
                response = web.Response()
            else:
                response = await handler(request)
            
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response
        
        app.middlewares.append(cors_middleware)
        
        # 注册路由
        app.router.add_post("/api/process", self._handle_process)
        app.router.add_post("/api/stream", self._handle_stream)
        app.router.add_get("/api/memory", self._handle_memory)
        app.router.add_post("/api/memory/search", self._handle_memory_search)
        app.router.add_post("/api/tools", self._handle_tools)
        app.router.add_post("/api/wiki", self._handle_wiki)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/stats", self._handle_stats)
        
        return app
    
    async def _handle_process(self, request) -> Any:
        """处理普通请求"""
        self._stats["total_requests"] += 1
        
        try:
            data = await request.json()
            
            input_text = data.get("input", "")
            session_id = data.get("session_id")
            search_memory = data.get("search_memory", True)
            search_wiki = data.get("search_wiki", False)
            
            # 处理
            result = []
            async for chunk in self._brain.process(
                input_text,
                session_id=session_id,
                stream=False,
                search_memory=search_memory,
                search_wiki=search_wiki
            ):
                result.append(chunk)
            
            self._stats["successful_requests"] += 1
            
            from aiohttp import web
            return web.json_response({
                "success": True,
                "result": "".join(result),
                "session_id": session_id
            })
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Process error: {e}")
            
            from aiohttp import web
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_stream(self, request) -> Any:
        """处理流式请求"""
        self._stats["total_requests"] += 1
        
        try:
            data = await request.json()
            
            input_text = data.get("input", "")
            session_id = data.get("session_id")
            search_memory = data.get("search_memory", True)
            search_wiki = data.get("search_wiki", False)
            
            # 创建流式响应
            from aiohttp import web
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            response.headers["Cache-Control"] = "no-cache"
            
            await response.prepare(request)
            
            # 流式输出
            async for chunk in self._brain.process(
                input_text,
                session_id=session_id,
                stream=True,
                search_memory=search_memory,
                search_wiki=search_wiki
            ):
                await response.write(f"data: {json.dumps({'chunk': chunk})}\n\n".encode())
            
            await response.write(b"data: [DONE]\n\n")
            
            self._stats["successful_requests"] += 1
            
            return response
            
        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Stream error: {e}")
            
            from aiohttp import web
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_memory(self, request) -> Any:
        """处理记忆请求"""
        from aiohttp import web
        
        if not self._brain or not self._brain._memory_system:
            return web.json_response({"error": "Memory system not available"}, status=503)
        
        stats = self._brain._memory_system.get_stats()
        
        return web.json_response({
            "success": True,
            "stats": stats
        })
    
    async def _handle_memory_search(self, request) -> Any:
        """处理记忆搜索请求"""
        from aiohttp import web
        
        try:
            data = await request.json()
            query = data.get("query", "")
            top_k = data.get("top_k", 5)
            
            if not self._brain or not self._brain._memory_system:
                return web.json_response({"error": "Memory system not available"}, status=503)
            
            results = await self._brain._memory_system.retrieve(query, top_k=top_k)
            
            return web.json_response({
                "success": True,
                "results": results
            })
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_tools(self, request) -> Any:
        """处理工具请求"""
        from aiohttp import web
        
        try:
            data = await request.json()
            tool_name = data.get("tool")
            args = data.get("args", {})
            
            if not self._brain or not self._brain._tool_manager:
                return web.json_response({"error": "Tool manager not available"}, status=503)
            
            result = await self._brain._tool_manager.call_tool(tool_name, **args)
            
            return web.json_response({
                "success": result.success,
                "result": result.result,
                "error": result.error
            })
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_wiki(self, request) -> Any:
        """处理维基百科请求"""
        from aiohttp import web
        
        try:
            data = await request.json()
            query = data.get("query", "")
            language = data.get("language", "zh")
            
            if not self._brain or not self._brain._tool_manager:
                return web.json_response({"error": "Wiki search not available"}, status=503)
            
            result = await self._brain._tool_manager.search_wikipedia(query, language)
            
            return web.json_response({
                "success": result is not None,
                "result": result
            })
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_status(self, request) -> Any:
        """处理状态请求"""
        from aiohttp import web
        
        status = {
            "initialized": self._brain._initialized if self._brain else False,
            "model_loaded": self._brain._model is not None if self._brain else False,
            "memory_active": self._brain._memory_system is not None if self._brain else False,
            "stdp_active": self._brain._stdp_engine is not None if self._brain else False,
            "timestamp": time.time()
        }
        
        return web.json_response({
            "success": True,
            "status": status
        })
    
    async def _handle_stats(self, request) -> Any:
        """处理统计请求"""
        from aiohttp import web
        
        stats = {
            "api": self._stats,
            "brain": self._brain.get_stats() if self._brain else {}
        }
        
        return web.json_response({
            "success": True,
            "stats": stats
        })
    
    async def stop(self) -> None:
        """停止服务器"""
        if hasattr(self, '_runner'):
            await self._runner.cleanup()
        logger.info("API Server stopped")


def create_app(config: Optional[APIConfig] = None) -> APIServer:
    """创建API服务器实例"""
    return APIServer(config)
