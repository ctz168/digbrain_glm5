"""
维基百科搜索模块
实现无限知识库扩展
"""

import asyncio
import aiohttp
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from urllib.parse import quote, urlencode
import time

logger = logging.getLogger(__name__)


@dataclass
class WikiConfig:
    """维基百科配置"""
    # API端点
    api_url: str = "https://zh.wikipedia.org/w/api.php"
    en_api_url: str = "https://en.wikipedia.org/w/api.php"
    
    # 请求配置
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 结果配置
    max_results: int = 5
    max_extract_length: int = 500
    
    # 缓存
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1小时


class WikiSearch:
    """
    维基百科搜索
    
    实现中英文维基百科搜索，扩展知识库
    """
    
    def __init__(self, config: Optional[WikiConfig] = None):
        self.config = config or WikiConfig()
        
        # 缓存
        self._cache: Dict[str, Dict] = {}
        self._cache_time: Dict[str, float] = {}
        
        # 会话
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 统计
        self._stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0
        }
    
    async def initialize(self) -> None:
        """初始化"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        logger.info("WikiSearch initialized")
    
    async def close(self) -> None:
        """关闭"""
        if self._session:
            await self._session.close()
    
    async def search(
        self,
        query: str,
        language: str = "zh",
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索维基百科
        
        Args:
            query: 搜索查询
            language: 语言 (zh, en)
            max_results: 最大结果数
            
        Returns:
            搜索结果列表
        """
        self._stats["total_searches"] += 1
        max_results = max_results or self.config.max_results
        
        # 检查缓存
        cache_key = f"{language}:{query}"
        if self.config.cache_enabled and cache_key in self._cache:
            if time.time() - self._cache_time[cache_key] < self.config.cache_ttl:
                self._stats["cache_hits"] += 1
                return self._cache[cache_key][:max_results]
        
        # 执行搜索
        results = await self._search_api(query, language, max_results)
        
        # 缓存结果
        if self.config.cache_enabled:
            self._cache[cache_key] = results
            self._cache_time[cache_key] = time.time()
        
        return results
    
    async def _search_api(
        self,
        query: str,
        language: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """调用维基百科API"""
        api_url = self.config.api_url if language == "zh" else self.config.en_api_url
        
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "utf8": 1
        }
        
        self._stats["api_calls"] += 1
        
        try:
            results = await self._api_request(api_url, params)
            
            search_results = []
            for item in results.get("query", {}).get("search", []):
                # 获取摘要
                extract = await self._get_extract(
                    item["title"],
                    language
                )
                
                search_results.append({
                    "title": item["title"],
                    "snippet": self._clean_html(item.get("snippet", "")),
                    "extract": extract,
                    "pageid": item.get("pageid"),
                    "url": self._get_page_url(item["title"], language)
                })
            
            return search_results
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Wiki search error: {e}")
            return []
    
    async def _get_extract(
        self,
        title: str,
        language: str
    ) -> str:
        """获取页面摘要"""
        api_url = self.config.api_url if language == "zh" else self.config.en_api_url
        
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exchars": self.config.max_extract_length,
            "format": "json"
        }
        
        try:
            results = await self._api_request(api_url, params)
            
            pages = results.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "extract" in page:
                    return page["extract"]
            
            return ""
            
        except Exception as e:
            logger.error(f"Get extract error: {e}")
            return ""
    
    async def _api_request(
        self,
        url: str,
        params: Dict
    ) -> Dict:
        """发送API请求"""
        if not self._session:
            await self.initialize()
        
        for attempt in range(self.config.max_retries):
            try:
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
    
    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 解码HTML实体
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        return text
    
    def _get_page_url(self, title: str, language: str) -> str:
        """获取页面URL"""
        base_url = "https://zh.wikipedia.org/wiki/" if language == "zh" else "https://en.wikipedia.org/wiki/"
        return base_url + quote(title.replace(" ", "_"))
    
    async def get_summary(
        self,
        query: str,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        获取搜索摘要
        
        返回最相关结果的摘要
        """
        results = await self.search(query, language, max_results=1)
        
        if results:
            return {
                "query": query,
                "title": results[0]["title"],
                "summary": results[0]["extract"],
                "url": results[0]["url"],
                "found": True
            }
        
        return {
            "query": query,
            "summary": "",
            "found": False
        }
    
    async def batch_search(
        self,
        queries: List[str],
        language: str = "zh"
    ) -> Dict[str, List[Dict]]:
        """批量搜索"""
        tasks = [self.search(q, language) for q in queries]
        results = await asyncio.gather(*tasks)
        
        return {q: r for q, r in zip(queries, results)}
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_time.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "cache_size": len(self._cache)
        }


class WikiKnowledgeBase:
    """
    维基百科知识库
    
    整合维基百科搜索到知识库系统
    """
    
    def __init__(self, wiki_search: Optional[WikiSearch] = None):
        self.wiki = wiki_search or WikiSearch()
        self._knowledge_cache: Dict[str, Dict] = {}
    
    async def query(
        self,
        topic: str,
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        查询知识
        
        Args:
            topic: 查询主题
            depth: 查询深度（是否查询相关主题）
            
        Returns:
            知识结果
        """
        # 主查询
        main_result = await self.wiki.get_summary(topic)
        
        result = {
            "topic": topic,
            "main": main_result,
            "related": []
        }
        
        # 相关主题查询
        if depth > 1 and main_result.get("found"):
            # 提取关键词
            keywords = self._extract_keywords(main_result.get("summary", ""))
            
            for keyword in keywords[:3]:  # 最多查询3个相关主题
                related = await self.wiki.get_summary(keyword)
                if related.get("found"):
                    result["related"].append(related)
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化实现：提取名词短语
        words = text.split()
        keywords = []
        
        for word in words:
            # 过滤短词和常见词
            if len(word) > 2 and word not in ["的", "是", "在", "和", "了"]:
                keywords.append(word)
        
        return keywords[:5]
    
    async def enrich_context(
        self,
        context: str,
        max_topics: int = 3
    ) -> str:
        """
        丰富上下文
        
        从维基百科获取相关知识丰富上下文
        """
        # 提取主题
        topics = self._extract_keywords(context)[:max_topics]
        
        enrichments = []
        for topic in topics:
            result = await self.wiki.get_summary(topic)
            if result.get("found"):
                enrichments.append(f"[{result['title']}] {result['summary'][:200]}")
        
        if enrichments:
            return context + "\n\n相关知识:\n" + "\n".join(enrichments)
        
        return context
