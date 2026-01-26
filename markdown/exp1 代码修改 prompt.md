我将会输入我的推理 prompt 和打分 prompt，以及两个相关的 python 文件，你需要基于推理 prompt 和打分 prompt 对这两份代码进行修改：

这是我的推理 prompt：

```json
PROMPT_TEMPLATE_v9 = """人物信息背景如下：
{person_info}
你的核心任务是模拟<{speaker_tag}>角色，生成出在{last_timestamp}时刻要发送的消息内容
要求如下：1. 内容格式为：\n`{last_timestamp}<{speaker_tag}>-<文本>-<消息内容>`\n注意：只输出这1行文本，不加额外描述
2. 消息特性要求：\n-展现高情商的对话风格，使用口语化表达，必要时使用表情符号（用[xx]的文本表示）\n-内容符合<{speaker_tag}>的角色设定,需自然承接对话历史，保持上下文的连贯性\n-字数范围: 4-30个汉字
当前对话历史如下：
{dialogue_history}
"""
```

这是打分 prompt：

``` json 
# 🟦【System Prompt】

你是一名专业的 **对话评估裁判（Dialogue Intent Judge）**，擅长对回复进行评分。你应该基于给定的评判标准评估给定的回复。

结合对话历史和助手（Assistant）的多个候选回复，你需要参考 **[通用评估标准]** 对回复进行打分。

#### 📝 输入数据 ####

**人物设定 (Persona)**：
{{persona}}

**对话历史 (Dialogue History)**：
{{dialogue_history}}

**用户真实后续回复 (辅助语境，非标准答案)**：
{{user_true_reply}}

**待评分的候选回复列表**：
{{candidate_responses}}

---

#### 📤 输出格式要求 ####

请严格按照以下三行格式输出，不要包含任何其他开场白或结束语：

Specific Criteria: <列出针对本条对话生成的具体评估维度（如：情绪安抚、技术排查准确性等）及其权重>
Analysis: <基于上述标准，逐个分析每个候选回复的优缺点>
Scores: <按顺序给出所有回复的评分，在方框内用逗号分隔，例如 \boxed{8, 5, 2}>
```

首先，当接受到数据进行打分时，需要先对custom中的候选回复进行格式转换，满足以下这种格式：[The Begin of Response 1] {{response_1_content}} [The End of Response 1]，将这种形式传递给candidate_responses字段。response_1_content的内容需要剔除时间和角色，只保留说话的内容，也就是消息内容。然后进行打分，打分保留限速和现在的逻辑，最终的奖励包括格式奖励和内容奖励，内容奖励直接提取Scores中的值，相同的打分重复三次，对三次打分后的结果进行加权平均得到该条回复最终的得分。custom文件最终返回eturn reward_tensor, format_scores, content_scores, total_scores, 3个评判标准，3 个 critiques（这两部分的内容做到一一对应，一份标准对应一份评论）, gt_responses 即可