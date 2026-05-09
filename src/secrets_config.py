"""
src/secrets_config.py
存放敏感配置（API key），此文件已加入 .gitignore，不会上传到 Gitee。
"""

LLM_API_KEY = "sk-ac1672bc3b2d481ebd286486dddc55f6"

# 用 DeepSeek 就用下面默认值；用其他厂改这两行：
#   通义千问: LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1", LLM_MODEL="qwen-plus"
#   智谱 GLM: LLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4", LLM_MODEL="glm-4-flash"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"