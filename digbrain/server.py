#!/usr/bin/env python3
"""
DigBrain Server
启动服务器
"""

import asyncio
import argparse
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DigBrain Server')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器地址')
    parser.add_argument('--port', type=int, default=8000,
                        help='API端口')
    parser.add_argument('--web-port', type=int, default=3000,
                        help='Web界面端口')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    
    args = parser.parse_args()
    
    # 导入模块
    from digbrain import DigBrain, BrainConfig
    from digbrain.api import create_app
    from digbrain.web import WebServer, WebConfig
    
    # 创建配置
    config = BrainConfig(
        model_path=args.model,
        api_host=args.host,
        api_port=args.port
    )
    
    # 初始化DigBrain
    logger.info("Initializing DigBrain...")
    brain = DigBrain(config)
    await brain.initialize()
    
    # 启动API服务器
    logger.info("Starting API server...")
    from digbrain.api.rest import APIServer, APIConfig
    api_server = APIServer(APIConfig(host=args.host, port=args.port))
    await api_server.initialize(brain)
    await api_server.start()
    
    # 启动Web服务器
    logger.info("Starting Web server...")
    web_server = WebServer(WebConfig(host=args.host, port=args.web_port))
    await web_server.initialize(brain)
    await web_server.start()
    
    logger.info(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    DigBrain Started                       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  API:  http://{args.host}:{args.port}                           ║
    ║  Web:  http://{args.host}:{args.web_port}                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 保持运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await brain.shutdown()
        await api_server.stop()


if __name__ == '__main__':
    asyncio.run(main())
