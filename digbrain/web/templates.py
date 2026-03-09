"""
模板渲染器
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TemplateRenderer:
    """模板渲染器"""
    
    def __init__(self, template_dir: str = "./templates"):
        self.template_dir = template_dir
        self._cache: Dict[str, str] = {}
    
    def render(self, template_name: str, context: Dict[str, Any] = None) -> str:
        """渲染模板"""
        context = context or {}
        
        # 检查缓存
        if template_name in self._cache:
            template = self._cache[template_name]
        else:
            # 从文件加载
            try:
                with open(f"{self.template_dir}/{template_name}", 'r', encoding='utf-8') as f:
                    template = f.read()
                self._cache[template_name] = template
            except FileNotFoundError:
                logger.error(f"Template not found: {template_name}")
                return ""
        
        # 简单变量替换
        for key, value in context.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        
        return template
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
