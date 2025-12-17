```python
python -m vllm.entrypoints.openai.api_server \
    --model /your/qwen3/model/path/Qwen3-32B-Instruct \
    --served-model-name "qwen3-32b" \
    --max_model_len 20176 \
    --tensor-parallel-size 8 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes



python -m vllm.entrypoints.openai.api_server --model   /gemini-1/space/space/private/pengjiaxin/base_ckpts/Qwen3-32B-Instruct --served-model-name "openchat" --max_model_len 20176 --tensor-parallel-size 2 --port 8015 --enable-auto-tool-choice --tool-call-parser hermes
```





为我修改这个代码，我的需求是：现在不要求生成稀少分类的角色，现在要求persona-to-persona，生成与提供的角色相关的十个角色，我会为你提供characters_file文件和categories_file文件，依然是在原本的分类体系，这个是不变的。只是我的需求更新了，每个角色生成10个相关角色





```json
        """生成用于产生用户query的提示词"""
        '''
        system_prompt = f"""你是一个专业的对话场景设计师，擅长创造富有真实感和深度的对话开场白，现在需要你为一个角色扮演对话生成合适的用户首轮询问。

角色名称：{role_name}
角色描述：{role_description}

请根据这个角色的特点、专业领域和角色设定，生成一个自然、合理、高质量、高信息量的用户首轮询问。这个询问应该：
1.  嵌入一个具体场景或个人困境: 不要泛泛而问。用户的提问应该基于一个具体的生活、工作或情感上的难题。
    - [优秀范例]: (对于营养师角色) “你好，我最近在为半程马拉松做准备，但训练到下午3点左右总是感觉精力崩溃，非常疲劳。想请问在饮食上有什么针对性的调整建议来帮助我维持体能吗？”
    - [应避免的范例]: (对于营养师角色) “如何健康饮食？”

2.  体现提问者的个人色彩: 让问题听起来像一个有着自己独特背景和困惑的真实个体提出的，而不是一个冰冷的测试指令。

3.  挖掘角色描述中的细节: 精准地利用角色描述中的关键词（如特定技能、经历、性格），让问题看起来是为这个角色“量身定做”的。

4.  包含足够的上下文: 提问应该是一句或两句完整的话，提供足够的背景信息，而不仅仅是一个简短的词组。这有助于开启一段有意义的对话。

5.  严格避免的模式: 严禁生成如“你好，能介绍下自己吗？”、“你是做什么的？”、“你有什么技能？”、“给我讲个故事吧”等任何宽泛、无具体指向性的“面试式”问题。

**输出指令:**
请直接输出最终的用户询问内容，不要包含任何额外的解释、标题或标记。"""
```

