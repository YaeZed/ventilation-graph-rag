"""
📚 Python logging 模块完整教程
详细讲解如何使用logging记录日志
"""

import logging
from docx import Document  # 正确的导入


# ============================================================
# 第一部分：最基础的 logging 使用
# ============================================================

def basic_logging_example():
    """最基础的logging使用示例"""
    print("\n" + "="*60)
    print("1️⃣ 基础logging示例")
    print("="*60)
    
    # 直接使用logging模块的函数（适合简单脚本）
    logging.debug("这是DEBUG级别的日志（默认不显示）")
    logging.info("这是INFO级别的日志（默认不显示）")
    logging.warning("这是WARNING级别的日志 ⚠️")
    logging.error("这是ERROR级别的日志 ❌")
    logging.critical("这是CRITICAL级别的日志 🔥")
    
    print("\n💡 注意：默认只显示WARNING及以上级别的日志\n")


# ============================================================
# 第二部分：配置 logging（basicConfig）
# ============================================================

def configured_logging_example():
    """配置logging的基本设置"""
    print("\n" + "="*60)
    print("2️⃣ 配置logging示例")
    print("="*60)
    
    # 基本配置（只能调用一次，通常在程序入口调用）
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
    )
    
    # 现在所有级别的日志都会显示
    logging.debug("现在DEBUG日志也显示了 🐛")
    logging.info("INFO日志也显示了 ℹ️")
    logging.warning("WARNING日志 ⚠️")
    logging.error("ERROR日志 ❌")


# ============================================================
# 第三部分：使用 Logger 对象（推荐方式）
# ============================================================

def logger_object_example():
    """使用Logger对象记录日志（推荐的专业方式）"""
    print("\n" + "="*60)
    print("3️⃣ Logger对象示例（推荐）")
    print("="*60)
    
    # API: logging.getLogger(name)
    # 参数: name - logger的名称（通常使用 __name__）
    # 返回: Logger对象
    # 作用: 获取指定名称的logger，如果不存在则创建
    logger = logging.getLogger(__name__)
    
    # __name__ 是什么？
    # 如果这个文件是主程序，__name__ = '__main__'
    # 如果这个文件被导入，__name__ = '模块名'
    print(f"当前模块名: {__name__}")
    
    # 设置logger的级别
    logger.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到logger
    logger.addHandler(console_handler)
    
    # 使用logger记录日志
    logger.debug("这是debug信息 🐛")
    logger.info("这是info信息 ℹ️")
    logger.warning("这是warning信息 ⚠️")
    logger.error("这是error信息 ❌")
    logger.critical("这是critical信息 🔥")


# ============================================================
# 第四部分：日志写入文件
# ============================================================

def file_logging_example():
    """将日志同时输出到控制台和文件"""
    print("\n" + "="*60)
    print("4️⃣ 日志文件示例")
    print("="*60)
    
    # 创建logger
    logger = logging.getLogger("FileLogger")
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的处理器（避免重复）
    logger.handlers.clear()
    
    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上
    
    # 2. 文件处理器
    # API: logging.FileHandler(filename, mode='a', encoding=None)
    # 参数:
    #   filename: 日志文件路径
    #   mode: 'a'=追加, 'w'=覆盖
    #   encoding: 编码（中文需要'utf-8'）
    file_handler = logging.FileHandler(
        'safety_output/app.log',
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # 记录日志
    logger.debug("这条DEBUG只会写入文件")
    logger.info("这条INFO会显示在控制台和文件")
    logger.error("这条ERROR也会显示在控制台和文件")
    
    print(f"\n💾 日志已保存到: safety_output/app.log")


# ============================================================
# 第五部分：实际应用 - 改进你的代码
# ============================================================

# 全局logger（在文件顶部创建）
logger = logging.getLogger(__name__)

def setup_logging():
    """配置logging（在程序入口调用一次）"""
    # 设置根logger的级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler('safety_output/ventilation.log', encoding='utf-8')  # 输出到文件
        ]
    )


def read_docx_with_tables(file_path: str):
    """
    从Word文档中读取内容，将表格转换为Markdown格式
    （带日志记录的版本）
    
    Args:
        file_path: Word文档路径
    
    Returns:
        str: 转换后的文本内容，表格以Markdown格式呈现
    """
    logger.info(f"开始读取Word文档: {file_path}")
    
    try:
        # 尝试打开文档
        doc = Document(file_path)
        
        # Python的None检查（不需要括号和大括号）
        if doc is None:
            logger.error("Word文档对象为None")
            return ""
        
        logger.debug(f"文档加载成功，开始处理元素")
        
        content = []
        for idx, element in enumerate(doc.element.body):
            if element.tag.endswith('p'):
                para = [p for p in doc.paragraphs if p._element == element][0]
                if para.text.strip():
                    content.append(para.text)
                    logger.debug(f"处理段落 {idx}: {para.text[:30]}...")
            
            elif element.tag.endswith('tbl'):
                logger.debug(f"处理表格 {idx}")
                # ... 处理表格的代码
        
        logger.info(f"文档读取完成，共处理 {len(content)} 个元素")
        return "\n".join(content)
    
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"读取文档时发生错误: {type(e).__name__} - {str(e)}")
        logger.exception("详细错误堆栈:")  # 自动记录完整的异常堆栈
        return ""


# ============================================================
# 第六部分：异常日志（重要！）
# ============================================================

def exception_logging_example():
    """演示如何记录异常"""
    print("\n" + "="*60)
    print("5️⃣ 异常日志示例")
    print("="*60)
    
    logger = logging.getLogger("ExceptionLogger")
    logger.setLevel(logging.DEBUG)
    
    # 添加控制台处理器
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    try:
        # 故意触发一个错误
        result = 10 / 0
    except ZeroDivisionError as e:
        # ❌ 不推荐的方式
        logger.error(f"发生错误: {e}")
        
        # ✅ 推荐的方式（自动记录完整的堆栈信息）
        logger.exception("除零错误发生:")
        
        # 或者使用 exc_info=True
        logger.error("除零错误发生:", exc_info=True)


# ============================================================
# 第七部分：日志级别详解
# ============================================================

def logging_levels_explanation():
    """日志级别详细说明"""
    print("\n" + "="*60)
    print("6️⃣ 日志级别说明")
    print("="*60)
    
    print("""
日志级别（从低到高）:

1. DEBUG (10)    🐛
   - 作用: 详细的诊断信息
   - 使用场景: 开发调试、追踪变量值、函数调用
   - 示例: logger.debug(f"变量x的值: {x}")

2. INFO (20)     ℹ️
   - 作用: 确认程序按预期运行
   - 使用场景: 记录重要的业务流程、状态变化
   - 示例: logger.info("用户登录成功")

3. WARNING (30)  ⚠️
   - 作用: 发生了意外，但程序仍可继续
   - 使用场景: 配置缺失、使用了已弃用的API
   - 示例: logger.warning("配置文件缺失，使用默认值")

4. ERROR (40)    ❌
   - 作用: 由于严重问题，某些功能无法执行
   - 使用场景: 文件不存在、网络请求失败
   - 示例: logger.error("数据库连接失败")

5. CRITICAL (50) 🔥
   - 作用: 严重错误，程序可能无法继续运行
   - 使用场景: 系统崩溃、数据损坏
   - 示例: logger.critical("内存不足，程序即将退出")

级别数值:
- 只有级别 >= logger设置级别 的日志才会输出
- 例如: logger.setLevel(logging.WARNING)
  → DEBUG和INFO不会输出
  → WARNING、ERROR、CRITICAL会输出
    """)


# ============================================================
# 第八部分：格式化字符串详解
# ============================================================

def formatting_explanation():
    """日志格式化字符串说明"""
    print("\n" + "="*60)
    print("7️⃣ 日志格式化说明")
    print("="*60)
    
    print("""
常用的格式化占位符:

%(asctime)s        - 时间戳（如：2024-02-13 10:30:00）
%(name)s           - logger名称（如：__main__）
%(levelname)s      - 日志级别（如：ERROR）
%(message)s        - 日志消息
%(filename)s       - 源文件名（如：run_ventilation_agent.py）
%(lineno)d         - 行号（如：123）
%(funcName)s       - 函数名
%(process)d        - 进程ID
%(thread)d         - 线程ID

示例格式:
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
输出:
2024-02-13 10:30:00 - __main__ - ERROR - 文件不存在

详细格式（用于调试）:
'[%(asctime)s] %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
输出:
[2024-02-13 10:30:00] test.py:45 - ERROR - 文件不存在
    """)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n" + "🎓 Python logging 模块完整教程")
    
    # 1. 基础示例
    basic_logging_example()
    
    # 2. 配置示例
    # configured_logging_example()  # 注释掉，避免与basicConfig冲突
    
    # 3. Logger对象示例
    # logger_object_example()
    
    # 4. 文件日志示例
    file_logging_example()
    
    # 5. 异常日志示例
    exception_logging_example()
    
    # 6. 日志级别说明
    logging_levels_explanation()
    
    # 7. 格式化说明
    formatting_explanation()
    
    print("\n" + "="*60)
    print("✅ 教程完成！")
    print("="*60)
