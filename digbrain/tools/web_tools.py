"""
网页工具模块
实现网页搜索和工具调用
"""

import asyncio
import json
import re
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """工具类型"""
    SEARCH = "search"
    CALCULATOR = "calculator"
    TRANSLATOR = "translator"
    WEATHER = "weather"
    CUSTOM = "custom"


@dataclass
class ToolConfig:
    """工具配置"""
    enable_wiki: bool = True
    enable_web: bool = True
    enable_calculator: bool = True
    enable_translator: bool = False
    
    # 超时配置
    default_timeout: float = 30.0
    
    # 重试配置
    max_retries: int = 3


@dataclass
class ToolResult:
    """工具结果"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class WebTools:
    """
    网页工具集
    
    提供各种工具调用能力
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """注册默认工具"""
        self._tools["calculator"] = self._calculator
        self._tools["text_processor"] = self._text_processor
        self._tools["json_parser"] = self._json_parser
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = ""
    ) -> None:
        """注册自定义工具"""
        self._tools[name] = func
        logger.info(f"Registered tool: {name}")
    
    async def call(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具结果
        """
        start_time = time.time()
        
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}"
            )
        
        try:
            result = await self._tools[tool_name](**kwargs)
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _calculator(self, expression: str) -> float:
        """计算器工具"""
        # 安全计算
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        try:
            result = eval(expression)
            return float(result)
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")
    
    async def _text_processor(
        self,
        text: str,
        operation: str = "summarize"
    ) -> str:
        """文本处理工具"""
        if operation == "summarize":
            # 简单摘要：取前200字符
            return text[:200] + "..." if len(text) > 200 else text
        elif operation == "word_count":
            return str(len(text.split()))
        elif operation == "char_count":
            return str(len(text))
        else:
            return text
    
    async def _json_parser(
        self,
        text: str,
        key: Optional[str] = None
    ) -> Any:
        """JSON解析工具"""
        try:
            data = json.loads(text)
            if key:
                return data.get(key)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}")
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """获取工具信息"""
        if tool_name in self._tools:
            return {
                "name": tool_name,
                "available": True
            }
        return {
            "name": tool_name,
            "available": False
        }


class ToolManager:
    """
    工具管理器
    
    管理所有工具的调用和协调
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or ToolConfig()
        
        # 工具实例
        self.web_tools = WebTools(self.config)
        self._wiki_search = None
        
        # 调用历史
        self._call_history: List[Dict] = []
        
        # 统计
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0
        }
    
    async def initialize(self) -> None:
        """初始化"""
        if self.config.enable_wiki:
            from .wiki_search import WikiSearch
            self._wiki_search = WikiSearch()
            await self._wiki_search.initialize()
        
        logger.info("ToolManager initialized")
    
    async def search_wikipedia(
        self,
        query: str,
        language: str = "zh"
    ) -> Optional[Dict]:
        """搜索维基百科"""
        if not self._wiki_search:
            return None
        
        return await self._wiki_search.get_summary(query, language)
    
    async def call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """调用工具"""
        self._stats["total_calls"] += 1
        
        result = await self.web_tools.call(tool_name, **kwargs)
        
        if result.success:
            self._stats["successful_calls"] += 1
        else:
            self._stats["failed_calls"] += 1
        
        # 记录历史
        self._call_history.append({
            "tool": tool_name,
            "args": kwargs,
            "success": result.success,
            "timestamp": time.time()
        })
        
        return result
    
    async def batch_call(
        self,
        calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        """批量调用工具"""
        tasks = [
            self.call_tool(call["tool"], **call.get("args", {}))
            for call in calls
        ]
        return await asyncio.gather(*tasks)
    
    def parse_tool_request(
        self,
        text: str
    ) -> Optional[Dict[str, Any]]:
        """
        解析工具请求
        
        从文本中提取工具调用意图
        """
        # 简单的模式匹配
        patterns = {
            "calculator": r"计算[：:]\s*(.+)",
            "search": r"搜索[：:]\s*(.+)",
            "wiki": r"百科[：:]\s*(.+)"
        }
        
        for tool, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                return {
                    "tool": tool,
                    "args": {"query": match.group(1).strip()}
                }
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "available_tools": self.web_tools.list_tools()
        }
    
    def get_call_history(
        self,
        limit: int = 100
    ) -> List[Dict]:
        """获取调用历史"""
        return self._call_history[-limit:]


class ToolChain:
    """
    工具链
    
    支持多工具组合调用
    """
    
    def __init__(self, tool_manager: ToolManager):
        self.manager = tool_manager
        self._chain: List[Dict] = []
    
    def add_step(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output_key: Optional[str] = None
    ) -> 'ToolChain':
        """添加步骤"""
        self._chain.append({
            "tool": tool_name,
            "args": args,
            "output_key": output_key
        })
        return self
    
    async def execute(self) -> Dict[str, Any]:
        """执行工具链"""
        results = {}
        context = {}
        
        for i, step in enumerate(self._chain):
            # 替换参数中的引用
            args = self._resolve_args(step["args"], context)
            
            # 执行工具
            result = await self.manager.call_tool(step["tool"], **args)
            
            # 保存结果
            step_key = step.get("output_key", f"step_{i}")
            results[step_key] = result
            context[step_key] = result.result if result.success else None
        
        return results
    
    def _resolve_args(
        self,
        args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析参数引用"""
        resolved = {}
        
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # 引用上下文
                ref_key = value[1:]
                resolved[key] = context.get(ref_key)
            else:
                resolved[key] = value
        
        return resolved
    
    def clear(self) -> None:
        """清空工具链"""
        self._chain.clear()
