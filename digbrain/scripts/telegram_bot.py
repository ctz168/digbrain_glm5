#!/usr/bin/env python3
"""
DigBrain Telegram Bot - 流式输出版
展示逐步生成的过程
"""

import sys
import os
import asyncio
import logging
import json
import urllib.parse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = "8627479342:AAGb1YlpbEY3utp1aSA4pKs9ppg1t8PVDDY"


class WikipediaSearch:
    """维基百科搜索"""
    
    async def search(self, query: str) -> dict:
        try:
            import urllib.request
            
            url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'DigBrainBot/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            if data.get('query', {}).get('search'):
                result = data['query']['search'][0]
                title = result['title']
                
                url2 = f"https://zh.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
                req2 = urllib.request.Request(url2, headers={'User-Agent': 'DigBrainBot/1.0'})
                with urllib.request.urlopen(req2, timeout=10) as response2:
                    data2 = json.loads(response2.read().decode())
                
                pages = data2.get('query', {}).get('pages', {})
                for page_id, page_data in pages.items():
                    extract = page_data.get('extract', '')
                    return {
                        "success": True,
                        "title": title,
                        "extract": extract[:500] if extract else "",
                        "url": f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    }
            
            return {"success": False, "error": f"未找到 '{query}'"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class WebSearch:
    """网页搜索"""
    
    KNOWLEDGE_BASE = {
        "python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。简洁易读，广泛用于Web开发、数据科学、AI等领域。",
        "人工智能": "人工智能(AI)是计算机科学分支，创建能执行人类智能任务的系统。包括机器学习、深度学习、NLP等。",
        "机器学习": "机器学习是AI核心技术，让计算机从数据中学习模式。包括监督学习、无监督学习、强化学习。",
        "深度学习": "深度学习使用多层神经网络处理复杂模式，在图像识别、语音处理、NLP领域突破显著。",
        "神经网络": "神经网络是受生物神经系统启发的计算模型，是现代AI的核心技术。",
        "transformer": "Transformer是2017年Google提出的神经网络架构，使用自注意力机制，是GPT、BERT的基础。",
        "gpt": "GPT是OpenAI的大型语言模型系列，GPT-4是最新版本，具有强大的语言理解和生成能力。",
        "chatgpt": "ChatGPT是OpenAI的对话AI系统，可进行自然对话、回答问题、协助写作。",
    }
    
    async def search(self, query: str):
        results = []
        query_lower = query.lower()
        
        for keyword, content in self.KNOWLEDGE_BASE.items():
            if keyword in query_lower:
                results.append({"title": f"📚 {keyword}", "snippet": content, "source": "知识库"})
        
        try:
            import urllib.request
            url = f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=2"
            req = urllib.request.Request(url, headers={'User-Agent': 'DigBrainBot/1.0'})
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode())
            
            for item in data.get('query', {}).get('search', []):
                results.append({
                    "title": f"📖 {item['title']}",
                    "snippet": item.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', ''),
                    "source": "维基百科"
                })
        except:
            pass
        
        return results if results else [{"title": "💡 提示", "snippet": f"未找到 '{query}' 相关信息", "source": "系统"}]


class DigBrainBot:
    """DigBrain Bot - 流式输出版"""
    
    def __init__(self, model_path: str = "./models/qwen"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.ready = False
        self.wiki = WikipediaSearch()
        self.web_search = WebSearch()
        self.conversation_history = {}
        
    async def initialize(self):
        logger.info("正在加载模型...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float32)
        self.model.eval()
        self.ready = True
        logger.info("模型加载完成！")
    
    def _detect_intent(self, message: str) -> dict:
        message_lower = message.lower()
        
        wiki_keywords = ["什么是", "是谁", "介绍", "解释"]
        search_keywords = ["搜索", "查找", "查询", "帮我找"]
        
        for kw in wiki_keywords:
            if kw in message:
                return {"type": "wiki", "query": message.replace(kw, "").strip()}
        
        for kw in search_keywords:
            if kw in message:
                return {"type": "search", "query": message.replace(kw, "").strip()}
        
        return {"type": "chat", "query": message}
    
    async def generate_streaming(self, prompt: str, user_id: int, chat_id: int, bot):
        """流式生成回复 - 实时更新消息"""
        if not self.ready:
            return "⏳ 模型加载中..."
        
        try:
            intent = self._detect_intent(prompt)
            
            # 维基搜索
            if intent["type"] == "wiki":
                status_msg = await bot.send_message(chat_id, "🔍 正在搜索维基百科...")
                result = await self.wiki.search(intent["query"])
                
                if result["success"]:
                    response = f"📚 **{result['title']}**\n\n{result['extract']}"
                    if result["url"]:
                        response += f"\n\n🔗 [查看详情]({result['url']})"
                else:
                    response = f"❌ {result['error']}"
                
                await status_msg.edit_text(response, parse_mode='Markdown')
                return response
            
            # 网页搜索
            elif intent["type"] == "search":
                status_msg = await bot.send_message(chat_id, "🔍 正在搜索...")
                results = await self.web_search.search(intent["query"])
                
                response = f"🔍 **搜索: {intent['query']}**\n\n"
                for i, r in enumerate(results[:3], 1):
                    response += f"{i}. {r['title']}\n   {r['snippet'][:150]}\n\n"
                
                await status_msg.edit_text(response, parse_mode='Markdown')
                return response
            
            # 流式对话
            history = self.conversation_history.get(user_id, [])
            context = ""
            if history:
                context = "\n".join([f"用户: {h['user']}\n助手: {h['bot'][:100]}" for h in history[-2:]])
                context += "\n\n"
            
            full_prompt = f"{context}用户: {prompt}\n\n助手: "
            
            # 发送初始消息
            status_msg = await bot.send_message(chat_id, "🧠 正在思考...")
            
            inputs = self.tokenizer(full_prompt, return_tensors='pt')
            
            # 流式生成
            generated_text = ""
            chunk_size = 15  # 每次生成的token数
            max_tokens = 250
            total_generated = 0
            last_update_time = 0
            
            with self.torch.no_grad():
                current_input = inputs
                
                while total_generated < max_tokens:
                    outputs = self.model.generate(
                        **current_input,
                        max_new_tokens=chunk_size,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True
                    )
                    
                    new_tokens = outputs.sequences[0][current_input['input_ids'].shape[1]:]
                    if len(new_tokens) == 0:
                        break
                    
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    generated_text += new_text
                    total_generated += len(new_tokens)
                    
                    # 更新消息（限制频率）
                    current_time = datetime.now().timestamp()
                    if current_time - last_update_time > 0.5:  # 每0.5秒更新一次
                        try:
                            display_text = generated_text[:4000]
                            if len(display_text) > 50:
                                await status_msg.edit_text(display_text + "▌")
                        except:
                            pass
                        last_update_time = current_time
                    
                    # 检查是否结束
                    if outputs.sequences[0][-1].item() == self.tokenizer.eos_token_id:
                        break
                    
                    # 更新输入
                    current_input = {'input_ids': outputs.sequences}
                    
                    # 小延迟模拟思考过程
                    await asyncio.sleep(0.1)
            
            # 最终消息
            final_text = generated_text.strip()
            if len(final_text) > 50:
                await status_msg.edit_text(final_text[:4000])
            else:
                await status_msg.delete()
                await bot.send_message(chat_id, final_text[:4000])
            
            # 保存历史
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []
            self.conversation_history[user_id].append({"user": prompt, "bot": final_text})
            if len(self.conversation_history[user_id]) > 10:
                self.conversation_history[user_id] = self.conversation_history[user_id][-10:]
            
            return final_text
            
        except Exception as e:
            logger.error(f"生成错误: {e}")
            return f"❌ 错误: {str(e)}"
    
    def clear_history(self, user_id: int):
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]


brain_bot = DigBrainBot()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
🧠 *DigBrain Bot - 流式输出版*

✨ *特色功能*：
• 🌊 实时流式输出
• 📚 维基百科搜索
• 🔍 网页搜索
• 💬 智能对话

📝 *命令*：
/help - 帮助
/status - 状态
/clear - 清除历史
/wiki [词] - 维基搜索
/search [词] - 网页搜索
/glm [内容] - 联系开发者

现在发送消息，看看流式输出效果！
"""
    await update.message.reply_text(msg, parse_mode='Markdown')


async def glm_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法: /glm [您想说的话]")
        return
    
    message = " ".join(context.args)
    user_name = update.effective_user.first_name or "用户"
    logger.info(f"[GLM消息] {user_name}: {message}")
    
    await update.message.reply_text(
        f"✅ 已收到您的消息！\n\n"
        f"📝 内容: {message}\n\n"
        f"开发者正在监控日志，会尽快回复。"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
📚 *帮助*

*命令*：
/start - 开始
/help - 帮助
/status - 状态
/clear - 清除历史
/wiki [词] - 维基搜索
/search [词] - 网页搜索
/glm [内容] - 联系开发者

*智能识别*：
• "什么是XXX" → 维基
• "搜索XXX" → 网页
• 直接发消息 → 流式对话
""", parse_mode='Markdown')


async def wiki_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法: /wiki [关键词]")
        return
    
    query = " ".join(context.args)
    result = await brain_bot.wiki.search(query)
    
    if result["success"]:
        response = f"📚 **{result['title']}**\n\n{result['extract']}"
        if result["url"]:
            response += f"\n\n🔗 [详情]({result['url']})"
    else:
        response = f"❌ {result['error']}"
    
    await update.message.reply_text(response, parse_mode='Markdown')


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法: /search [关键词]")
        return
    
    query = " ".join(context.args)
    results = await brain_bot.web_search.search(query)
    
    response = f"🔍 **{query}**\n\n"
    for i, r in enumerate(results[:3], 1):
        response += f"{i}. {r['title']}\n   {r['snippet'][:150]}\n\n"
    
    await update.message.reply_text(response, parse_mode='Markdown')


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"""
📊 *状态*

• 模型: Qwen2.5-0.5B
• 状态: {'✅ 就绪' if brain_bot.ready else '⏳ 加载中'}
• 流式输出: ✅ 启用
• 维基搜索: ✅ 启用
• 网页搜索: ✅ 启用
""", parse_mode='Markdown')


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    brain_bot.clear_history(update.effective_user.id)
    await update.message.reply_text("🗑️ 已清除")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    user_message = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "用户"
    chat_id = update.effective_chat.id
    
    logger.info(f"消息 [{user_name}]: {user_message[:60]}...")
    
    # 流式生成
    await brain_bot.generate_streaming(
        user_message, user_id, chat_id, context.bot
    )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")


async def post_init(application: Application):
    commands = [
        BotCommand("start", "开始"),
        BotCommand("help", "帮助"),
        BotCommand("status", "状态"),
        BotCommand("clear", "清除"),
        BotCommand("wiki", "维基搜索"),
        BotCommand("search", "网页搜索"),
        BotCommand("glm", "联系开发者"),
    ]
    await application.bot.set_my_commands(commands)
    await brain_bot.initialize()
    logger.info("Bot初始化完成！")


def main():
    logger.info("="*50)
    logger.info("  DigBrain Bot - 流式输出版")
    logger.info("="*50)
    
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("wiki", wiki_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("glm", glm_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    logger.info("Bot启动，等待消息...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
