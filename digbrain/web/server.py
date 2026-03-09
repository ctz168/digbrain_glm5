"""
Web服务器模块
提供前端界面
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    """Web配置"""
    host: str = "0.0.0.0"
    port: int = 3000
    static_dir: str = "./static"
    template_dir: str = "./templates"
    debug: bool = False


class WebServer:
    """
    Web服务器
    
    提供前端界面
    """
    
    def __init__(self, config: Optional[WebConfig] = None):
        self.config = config or WebConfig()
        self._brain = None
        self._app = None
    
    async def initialize(self, brain: Any) -> None:
        """初始化"""
        self._brain = brain
        logger.info(f"Web server initialized on {self.config.host}:{self.config.port}")
    
    async def start(self) -> None:
        """启动服务器"""
        try:
            from aiohttp import web
            
            app = web.Application()
            
            # 静态文件
            static_path = Path(self.config.static_dir)
            static_path.mkdir(parents=True, exist_ok=True)
            
            app.router.add_static("/static", str(static_path))
            
            # 主页
            app.router.add_get("/", self._handle_index)
            app.router.add_get("/chat", self._handle_chat)
            app.router.add_get("/memory", self._handle_memory)
            app.router.add_get("/settings", self._handle_settings)
            
            # API代理
            app.router.add_post("/api/chat", self._handle_chat_api)
            app.router.add_post("/api/stream", self._handle_stream_api)
            
            self._app = app
            
            from aiohttp import web
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(
                runner,
                self.config.host,
                self.config.port
            )
            await site.start()
            
            logger.info(f"Web server started on http://{self.config.host}:{self.config.port}")
            
        except ImportError:
            logger.warning("aiohttp not installed, web server not started")
    
    async def _handle_index(self, request) -> Any:
        """处理主页"""
        from aiohttp import web
        
        html = self._get_index_html()
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_chat(self, request) -> Any:
        """处理聊天页面"""
        from aiohttp import web
        
        html = self._get_chat_html()
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_memory(self, request) -> Any:
        """处理记忆页面"""
        from aiohttp import web
        
        html = self._get_memory_html()
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_settings(self, request) -> Any:
        """处理设置页面"""
        from aiohttp import web
        
        html = self._get_settings_html()
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_chat_api(self, request) -> Any:
        """处理聊天API"""
        from aiohttp import web
        
        try:
            data = await request.json()
            input_text = data.get("input", "")
            session_id = data.get("session_id")
            
            result = []
            async for chunk in self._brain.process(
                input_text,
                session_id=session_id,
                stream=False
            ):
                result.append(chunk)
            
            return web.json_response({
                "success": True,
                "output": "".join(result)
            })
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_stream_api(self, request) -> Any:
        """处理流式API"""
        from aiohttp import web
        
        try:
            data = await request.json()
            input_text = data.get("input", "")
            session_id = data.get("session_id")
            
            response = web.StreamResponse()
            response.headers["Content-Type"] = "text/event-stream"
            response.headers["Cache-Control"] = "no-cache"
            
            await response.prepare(request)
            
            async for chunk in self._brain.process(
                input_text,
                session_id=session_id,
                stream=True
            ):
                await response.write(f"data: {json.dumps({'chunk': chunk})}\n\n".encode())
            
            await response.write(b"data: [DONE]\n\n")
            
            return response
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    def _get_index_html(self) -> str:
        """获取主页HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DigBrain - 类脑智能系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 40px 0;
        }
        h1 {
            font-size: 3em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 1.2em;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .feature-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0,212,255,0.2);
        }
        .feature-card h3 {
            color: #00d4ff;
            margin-bottom: 15px;
        }
        .feature-card p {
            color: #aaa;
            line-height: 1.6;
        }
        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 40px 0;
        }
        .btn {
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }
        .btn-primary {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: #fff;
            border: none;
        }
        .btn-primary:hover {
            box-shadow: 0 5px 30px rgba(0,212,255,0.4);
        }
        .btn-secondary {
            background: transparent;
            color: #00d4ff;
            border: 2px solid #00d4ff;
        }
        .btn-secondary:hover {
            background: rgba(0,212,255,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 DigBrain</h1>
            <p class="subtitle">类脑智能系统 - 模拟人脑的信息处理机制</p>
        </header>
        
        <div class="nav-buttons">
            <a href="/chat" class="btn btn-primary">开始对话</a>
            <a href="/memory" class="btn btn-secondary">记忆管理</a>
            <a href="/settings" class="btn btn-secondary">系统设置</a>
        </div>
        
        <div class="features">
            <div class="feature-card">
                <h3>⚡ 高刷新流式处理</h3>
                <p>模拟人脑毫秒级处理速度，实现实时流式输入输出，每次处理小批量数据，提供即时响应。</p>
            </div>
            <div class="feature-card">
                <h3>💾 存算分离架构</h3>
                <p>借鉴DeepSeek最新论文框架，实现存储与计算分离，突破传统Transformer内存瓶颈。</p>
            </div>
            <div class="feature-card">
                <h3>🔄 在线STDP学习</h3>
                <p>脉冲时序依赖可塑性实现在线学习，无需离线重训练，实时适应新知识。</p>
            </div>
            <div class="feature-card">
                <h3>🧠 类人脑记忆</h3>
                <p>模拟海马体记忆管理，支持神经累积增长、联想检索和遗忘机制。</p>
            </div>
            <div class="feature-card">
                <h3>🌐 无限知识扩展</h3>
                <p>整合维基百科API，实现知识库无限扩展，支持中英文实时知识检索。</p>
            </div>
            <div class="feature-card">
                <h3>🖼️ 多模态支持</h3>
                <p>整合Qwen3.5-0.8B模型，支持文本、图像、视频的统一处理。</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    def _get_chat_html(self) -> str:
        """获取聊天页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DigBrain - 对话</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            min-height: 100vh;
            color: #fff;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            flex: 1;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .chat-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .chat-header h1 {
            font-size: 1.5em;
            color: #00d4ff;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 0;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
        }
        .message.user {
            justify-content: flex-end;
        }
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 20px;
            line-height: 1.5;
        }
        .message.user .message-content {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        }
        .message.assistant .message-content {
            background: rgba(255,255,255,0.1);
        }
        .input-area {
            display: flex;
            gap: 10px;
            padding: 20px 0;
        }
        .input-area textarea {
            flex: 1;
            padding: 15px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 1em;
            resize: none;
        }
        .input-area button {
            padding: 15px 30px;
            border-radius: 15px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: #fff;
            border: none;
            cursor: pointer;
            font-size: 1em;
        }
        .options {
            display: flex;
            gap: 20px;
            padding: 10px 0;
        }
        .option {
            display: flex;
            align-items: center;
            gap: 5px;
            color: #888;
            font-size: 0.9em;
        }
        .option input {
            accent-color: #00d4ff;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧠 DigBrain 对话</h1>
        </div>
        
        <div class="messages" id="messages">
            <div class="message assistant">
                <div class="message-content">
                    你好！我是DigBrain，一个类脑智能系统。我可以进行对话、搜索记忆、查询维基百科。有什么我可以帮助你的吗？
                </div>
            </div>
        </div>
        
        <div class="options">
            <label class="option">
                <input type="checkbox" id="searchMemory" checked>
                搜索记忆
            </label>
            <label class="option">
                <input type="checkbox" id="searchWiki">
                搜索维基百科
            </label>
            <label class="option">
                <input type="checkbox" id="streamMode" checked>
                流式输出
            </label>
        </div>
        
        <div class="input-area">
            <textarea id="userInput" rows="2" placeholder="输入你的问题..."></textarea>
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>
    
    <script>
        let sessionId = 'session_' + Date.now();
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            messageDiv.innerHTML = '<div class="message-content">' + content + '</div>';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            const searchMemory = document.getElementById('searchMemory').checked;
            const searchWiki = document.getElementById('searchWiki').checked;
            const streamMode = document.getElementById('streamMode').checked;
            
            if (streamMode) {
                // 流式输出
                const response = await fetch('/api/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        input: message,
                        session_id: sessionId,
                        search_memory: searchMemory,
                        search_wiki: searchWiki
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let messageDiv = null;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') break;
                            
                            try {
                                const json = JSON.parse(data);
                                if (json.chunk) {
                                    assistantMessage += json.chunk;
                                    if (!messageDiv) {
                                        addMessage('assistant', assistantMessage);
                                        messageDiv = document.getElementById('messages').lastChild;
                                    } else {
                                        messageDiv.querySelector('.message-content').textContent = assistantMessage;
                                    }
                                }
                            } catch (e) {}
                        }
                    }
                }
            } else {
                // 普通输出
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        input: message,
                        session_id: sessionId,
                        search_memory: searchMemory,
                        search_wiki: searchWiki
                    })
                });
                
                const data = await response.json();
                addMessage('assistant', data.output || data.error);
            }
        }
        
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""
    
    def _get_memory_html(self) -> str:
        """获取记忆页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DigBrain - 记忆管理</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .search-box input {
            flex: 1;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            color: #fff;
            font-size: 1em;
        }
        .search-box button {
            padding: 15px 30px;
            border-radius: 10px;
            background: #00d4ff;
            color: #000;
            border: none;
            cursor: pointer;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-card .value {
            font-size: 2em;
            color: #00d4ff;
        }
        .stat-card .label {
            color: #888;
            margin-top: 5px;
        }
        .results {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
        }
        .result-item {
            padding: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .result-item:last-child { border-bottom: none; }
        .result-item .content { margin-bottom: 10px; }
        .result-item .meta {
            font-size: 0.8em;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 记忆管理</h1>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="value" id="totalMemories">-</div>
                <div class="label">总记忆数</div>
            </div>
            <div class="stat-card">
                <div class="value" id="consolidated">-</div>
                <div class="label">已巩固</div>
            </div>
            <div class="stat-card">
                <div class="value" id="queries">-</div>
                <div class="label">查询次数</div>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchQuery" placeholder="搜索记忆...">
            <button onclick="searchMemory()">搜索</button>
        </div>
        
        <div class="results" id="results">
            <p style="color: #888; text-align: center;">输入关键词搜索记忆</p>
        </div>
    </div>
    
    <script>
        async function loadStats() {
            const response = await fetch('/api/memory');
            const data = await response.json();
            if (data.success) {
                document.getElementById('totalMemories').textContent = data.stats.total_memories || 0;
                document.getElementById('consolidated').textContent = data.stats.consolidated || 0;
                document.getElementById('queries').textContent = data.stats.memory_queries || 0;
            }
        }
        
        async function searchMemory() {
            const query = document.getElementById('searchQuery').value;
            if (!query) return;
            
            const response = await fetch('/api/memory/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: 10 })
            });
            
            const data = await response.json();
            const resultsDiv = document.getElementById('results');
            
            if (data.success && data.results.length > 0) {
                resultsDiv.innerHTML = data.results.map(r => `
                    <div class="result-item">
                        <div class="content">${r.content}</div>
                        <div class="meta">
                            相似度: ${(r.similarity * 100).toFixed(1)}% | 
                            访问: ${r.access_count}次 |
                            类型: ${r.memory_type}
                        </div>
                    </div>
                `).join('');
            } else {
                resultsDiv.innerHTML = '<p style="color: #888; text-align: center;">未找到相关记忆</p>';
            }
        }
        
        loadStats();
    </script>
</body>
</html>
"""
    
    def _get_settings_html(self) -> str:
        """获取设置页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DigBrain - 设置</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 30px; }
        .setting-group {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .setting-group h3 {
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .setting-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
        }
        .setting-item label { color: #aaa; }
        .setting-item input[type="range"] {
            width: 200px;
            accent-color: #00d4ff;
        }
        .setting-item input[type="checkbox"] {
            width: 20px;
            height: 20px;
            accent-color: #00d4ff;
        }
        .setting-item select {
            padding: 8px;
            border-radius: 5px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn-save {
            padding: 15px 40px;
            border-radius: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: #fff;
            border: none;
            cursor: pointer;
            font-size: 1em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ 系统设置</h1>
        
        <div class="setting-group">
            <h3>处理设置</h3>
            <div class="setting-item">
                <label>刷新率 (Hz)</label>
                <input type="range" min="10" max="100" value="30" id="refreshRate">
            </div>
            <div class="setting-item">
                <label>最大上下文长度</label>
                <input type="range" min="512" max="8192" value="4096" id="maxContext">
            </div>
        </div>
        
        <div class="setting-group">
            <h3>记忆设置</h3>
            <div class="setting-item">
                <label>启用记忆系统</label>
                <input type="checkbox" checked id="enableMemory">
            </div>
            <div class="setting-item">
                <label>短期记忆时长 (秒)</label>
                <input type="range" min="10" max="300" value="30" id="shortTermDuration">
            </div>
        </div>
        
        <div class="setting-group">
            <h3>学习设置</h3>
            <div class="setting-item">
                <label>启用在线学习 (STDP)</label>
                <input type="checkbox" checked id="enableSTDP">
            </div>
            <div class="setting-item">
                <label>学习率</label>
                <input type="range" min="0.001" max="0.1" step="0.001" value="0.01" id="learningRate">
            </div>
        </div>
        
        <div class="setting-group">
            <h3>工具设置</h3>
            <div class="setting-item">
                <label>启用维基百科搜索</label>
                <input type="checkbox" checked id="enableWiki">
            </div>
            <div class="setting-item">
                <label>默认语言</label>
                <select id="defaultLang">
                    <option value="zh">中文</option>
                    <option value="en">English</option>
                </select>
            </div>
        </div>
        
        <button class="btn-save" onclick="saveSettings()">保存设置</button>
    </div>
    
    <script>
        function saveSettings() {
            const settings = {
                refreshRate: document.getElementById('refreshRate').value,
                maxContext: document.getElementById('maxContext').value,
                enableMemory: document.getElementById('enableMemory').checked,
                shortTermDuration: document.getElementById('shortTermDuration').value,
                enableSTDP: document.getElementById('enableSTDP').checked,
                learningRate: document.getElementById('learningRate').value,
                enableWiki: document.getElementById('enableWiki').checked,
                defaultLang: document.getElementById('defaultLang').value
            };
            
            localStorage.setItem('digbrain_settings', JSON.stringify(settings));
            alert('设置已保存！');
        }
        
        // 加载保存的设置
        const saved = localStorage.getItem('digbrain_settings');
        if (saved) {
            const settings = JSON.parse(saved);
            for (const [key, value] of Object.entries(settings)) {
                const el = document.getElementById(key);
                if (el) {
                    if (el.type === 'checkbox') {
                        el.checked = value;
                    } else {
                        el.value = value;
                    }
                }
            }
        }
    </script>
</body>
</html>
"""
