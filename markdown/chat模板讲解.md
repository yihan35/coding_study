```django
{# --- 第一部分：处理工具定义和系统提示 --- #}

{%- if tools -%} {# 判断在调用模板时，是否传入了 'tools' 这个变量 #}
	{{- "<|im_start|>system\n" -}} {# 如果传入了 'tools'，则开始一个 system 角色的消息 #}
	{%- if messages[0].role == "system" -%} {# 检查原始消息列表的第一条是否也是 system 角色，以便合并 #}
		{{- messages[0].content + "\n\n" -}} {# 如果是，就先把用户定义的通用 system 消息内容放进来 #}
	{%- endif -%} {# 结束对第一条消息是否为 system 的检查 #}
	{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" -}} {# 输出一段固定的、告诉模型如何使用工具的指令文本 #}
	{%- for tool in tools -%} {# 开始遍历 'tools' 列表中的每一个工具 #}
		{{- "\n" -}} {# 在每个工具定义前加一个换行符，让格式更清晰 #}
		{{- tool | tojson -}} {# 使用 tojson 过滤器将工具的Python字典对象转换为JSON字符串格式 #}
	{%- endfor -%} {# 结束工具列表的遍历 #}
	{{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" -}} {# 输出工具部分的结束标签和关于如何返回 tool_call 的指令，并结束这个 system 消息 #}
{%- elif messages[0].role == "system" -%} {# 如果没有传入 'tools'，但消息列表的第一条是 system 角色 #}
	{{- "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n" -}} {# 则只按标准格式处理这条普通的 system 消息 #}
{%- endif -%} {# 结束对 system 消息或 tools 的判断 #}

{# --- 第二部分：多步工具调用的逻辑准备 --- #}
        
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages | length - 1) -%} {# 创建一个 namespace 对象，用于在循环内外共享变量，并初始化默认值 #}
{%- for message in messages[::-1] -%} {# **倒序**遍历消息列表（从后往前），目的是寻找最后一个真正的用户提问 #}
	{%- set index = messages | length - 1 - loop.index0 -%} {# 计算当前消息在原始列表中的正序索引 #}
	{%- if ns.multi_step_tool and message.role == "user" and message.content is string and not (message.content.startswith("<tool_response>") and message.content.endswith("</tool_response>")) -%} {# 判断当前消息是否是一个普通的用户提问（而不是由工具响应伪装的） #}
		{%- set ns.multi_step_tool = false -%} {# 找到后，设置标志位为 false，表示不再需要继续寻找 #}
		{%- set ns.last_query_index = index -%} {# 在 namespace 中记录下这个“最后用户提问”的索引，供后续使用 #}
	{%- endif -%} {# 结束 if 判断 #}
{%- endfor -%} {# 结束倒序遍历 #}

{# --- 第三部分：主循环，逐条格式化消息 --- #}

{%- for message in messages -%} {# **正序**遍历消息列表，开始格式化输出每一条消息 #}
	{%- if message.content is string -%} {# 检查消息的 content 字段是否为字符串类型 #}
		{%- set content = message.content -%} {# 如果是字符串，则将其赋值给局部变量 'content' #}
	{%- else -%} {# 如果 content 不是字符串（例如 None 或不存在） #}
		{%- set content = "" -%} {# 则设置 'content' 为空字符串，以避免后续处理时出错 #}
	{%- endif -%} {# 结束对 content 类型的检查 #}

	{# --- 处理 User 和 System 角色 --- #}
	{%- if message.role == "user" or message.role == "system" and not loop.first -%} {# 判断角色是否为 'user'，或者是 'system' 但非第一条消息（第一条已在前面处理过） #}
		{{- "<|im_start|>" + message.role + "\n" + content + "<|im_end|>" + "\n" -}} {# 按标准格式输出 user 或 system 消息 #}

	{# --- 处理 Assistant 角色 --- #}
	{%- elif message.role == "assistant" -%} {# 如果角色是 'assistant'，则进入复杂的处理逻辑 #}
		{%- set reasoning_content = "" -%} {# 初始化一个用于存放思考过程的变量为空字符串 #}
		{%- if message.reasoning_content is string -%} {# (高级用法) 检查消息中是否直接提供了 'reasoning_content' 字段 #}
			{%- set reasoning_content = message.reasoning_content -%} {# 如果有，直接使用它的值 #}
		{%- elif "</think>" in content -%} {# 否则，检查 'content' 中是否包含 "</think>" 标签 #}
			{%- set reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n") -%} {# 如果包含，则从 'content' 中提取出 <think> 标签内的文本作为思考过程 #}
			{%- set content = content.split("</think>")[-1].lstrip("\n") -%} {# 同时，从 'content' 变量中移除 <think> 及其内容，只保留实际的回答部分 #}
		{%- endif -%} {# 结束对思考内容的提取 #}
		{%- if loop.index0 > ns.last_query_index -%} {# 判断当前消息是否在“最后用户提问”之后，即是否属于工具调用或总结环节 #}
			{%- if loop.last or not loop.last and reasoning_content -%} {# 判断是否为最后一条消息，或者不是最后一条但包含思考内容 #}
				{{- "<|im_start|>" + message.role + "\n<think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + content.lstrip("\n") -}} {# 如果满足条件，则将 <think> 标签和思考内容显式地打印出来 #}
			{%- else -%} {# 如果不满足上述条件（例如，是中间步骤且无思考内容） #}
				{{- "<|im_start|>" + message.role + "\n" + content -}} {# 则只打印被清理过的 'content'（即隐藏思考过程） #}
			{%- endif -%} {# 结束对是否显示 <think> 的判断 #}
		{%- else -%} {# 如果当前消息在“最后用户提问”之前 #}
			{{- "<|im_start|>" + message.role + "\n" + content -}} {# 则按普通方式打印 'content' #}
		{%- endif -%} {# 结束对消息位置的判断 #}
		{%- if message.tool_calls -%} {# 检查当前 assistant 消息中是否包含 'tool_calls' 字段 #}
			{%- for tool_call in message.tool_calls -%} {# 如果有，则遍历每一个工具调用请求 #}
				{%- if loop.first and content or not loop.first -%} {# 判断是否需要在 tool_call 前加换行符（为了格式美观） #}
					{{- "\n" -}} {# 添加换行符 #}
				{%- endif -%} {# 结束换行符判断 #}
				{%- if tool_call.function -%} {# 检查工具调用对象中是否有 'function' 键 #}
					{%- set tool_call = tool_call.function -%} {# 如果有，则使用 'function' 键对应的值作为工具调用的主体 #}
				{%- endif -%} {# 结束对 'function' 键的检查 #}
				{{- "<tool_call>\n{\"name\": \"" -}} {# 输出 <tool_call> 的起始标签和函数名的键 #}
				{{- tool_call.name -}} {# 输出函数名 #}
				{{- "\", \"arguments\": " -}} {# 输出参数的键 #}
				{%- if tool_call.arguments is string -%} {# 检查参数是否已经是字符串格式 #}
					{{- tool_call.arguments -}} {# 如果是，直接输出 #}
				{%- else -%} {# 如果参数是字典或其他格式 #}
					{{- tool_call.arguments | tojson -}} {# 则使用 tojson 过滤器将其转换为JSON字符串 #}
				{%- endif -%} {# 结束对参数格式的判断 #}
				{{- "}\n</tool_call>" -}} {# 输出 tool_call 的结束部分 #}
			{%- endfor -%} {# 结束对 'tool_calls' 的遍历 #}
		{%- endif -%} {# 结束对 'tool_calls' 的检查 #}
		{{- "<|im_end|>\n" -}} {# 输出 assistant 消息的结束标记 #}

	{# --- 处理 Tool 角色 --- #}
	{%- elif message.role == "tool" -%} {# 如果角色是 'tool'，处理工具返回的结果 #}
		{%- if loop.first or messages[loop.index0 - 1].role != "tool" -%} {# 判断是否需要开始一个新的 'user' 消息块（用于包裹tool_response） #}
			{{- "<|im_start|>user" -}} {# 如果是连续 tool 消息的第一条，就输出起始标记（注意这里伪装成 user 角色） #}
		{%- endif -%} {# 结束对起始标记的判断 #}
		{{- "\n<tool_response>\n" -}} {# 输出 <tool_response> 的起始标签 #}
		{{- content -}} {# 输出工具返回的具体内容 #}
		{{- "\n</tool_response>" -}} {# 输出 <tool_response> 的结束标签 #}
		{%- if loop.last or messages[loop.index0 + 1].role != "tool" -%} {# 判断是否需要结束当前的 'user' 消息块 #}
			{{- "<|im_end|>\n" -}} {# 如果这是连续 tool 消息的最后一条，则输出结束标记 #}
		{%- endif -%} {# 结束对结束标记的判断 #}
	{%- endif -%} {# 结束所有角色的判断 #}
{%- endfor -%} {# 结束主循环 #}


{# --- 第四部分：添加用于推理的生成提示 --- #}

{%- if add_generation_prompt -%} {# 判断是否需要在末尾添加引导生成的提示（通常在推理时为 True） #}
	{{- "<|im_start|>assistant\n" -}} {# 如果需要，则输出一个未闭合的 assistant 起始标记，提示模型开始生成 #}
	{%- if enable_thinking is defined and enable_thinking is false -%} {# (高级用法) 判断是否需要强制开启或关闭思考模式 #}
		{{- "<think>\n\n</think>\n\n" -}} {# 如果需要，输出一个空的思考块 #}
	{%- endif -%} {# 结束对 thinking 的判断 #}
{%- endif -%} {# 结束对生成提示的判断 #}
```





---

这个模板的核心设计思想是：它不仅是简单地拼接字符串，更像是一个小型的**预处理器**，它会**解析、重构**你输入的`messages`和`tools`，以生成最符合模型训练和推理规范的格式。

------

### **第一部分：工具定义与系统提示 (System Prompt)**

这部分代码的职责是生成对话的“开场白”，即告诉模型它的身份、能力（特别是可用的工具）和基本指令。

```Django
{%- if tools -%}
	{{- "<|im_start|>system\n" -}}
	{%- if messages[0].role == "system" -%}
		{{- messages[0].content + "\n\n" -}}
	{%- endif -%}
	{{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" -}}
	{%- for tool in tools -%}
		{{- "\n" -}}
		{{- tool | tojson -}}
	{%- endfor -%}
	{{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" -}}
{%- elif messages[0].role == "system" -%}
	{{- "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n" -}}
{%- endif -%}
```

**逐行解释:**

- `{%- if tools -%}`:
  - **含义**: 判断在调用`apply_chat_template`时是否传入了`tools`这个变量。
  - **作用**: 这是整个工具调用格式的入口。如果提供了工具，就走工具调用的专用系统提示逻辑。
- `{{- "<|im_start|>system\n" -}}`:
  - **含义**: 输出一个`system`角色的起始标记。
- `{%- if messages[0].role == "system" -%}`:
  - **含义**: 检查你的`messages`列表中的第一条消息是不是也是`system`角色。
  - **作用**: 这允许你将一个通用的系统提示（例如“你是一个乐于助人的助手”）和一个由工具生成的系统提示合并。如果存在，它会先把你的通用系统提示内容放进去。
- `{{- "# Tools\n\n..." -}}`:
  - **含义**: 输出一段固定的、硬编码的文本，作为给模型的指令。告诉它工具的定义在`<tools>`标签内，以及返回结果时要使用`<tool_call>`标签。
- `{%- for tool in tools -%}`:
  - **含义**: 开始遍历`tools`列表中的每一个工具定义。
- `{{- tool | tojson -}}`:
  - **含义**: 这是关键！`tool`是一个Python字典，`| tojson`过滤器会把它转换成一个紧凑的JSON字符串。
  - **作用**: 将工具的结构化数据序列化为文本，让模型可以读取。
- `{%- elif messages[0].role == "system" -%}`:
  - **含义**: 如果第一个`if tools`不满足（即没有提供工具），则检查第一条消息是否为`system`角色。
  - **作用**: 处理没有工具调用，但有普通系统提示的对话。

------

### **第二部分：多步工具调用逻辑准备**

这部分非常高级，它的作用是**提前分析对话历史**，为后面正确处理复杂的多步工具调用做准备。

```Django
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages | length - 1) -%}
{%- for message in messages[::-1] -%}
	{%- set index = messages | length - 1 - loop.index0 -%}
	{%- if ns.multi_step_tool and message.role == "user" and message.content is string and not (message.content.startswith("<tool_response>") and message.content.endswith("</tool_response>")) -%}
		{%- set ns.multi_step_tool = false -%}
		{%- set ns.last_query_index = index -%}
	{%- endif -%}
{%- endfor -%}
```

**逐行解释:**

- `{%- set ns = namespace(...) -%}`:
  - **含义**: 创建一个`namespace`对象。
  - **作用**: 在Jinja2的循环内部，你不能直接修改外部的变量。`namespace`就像一个特殊的容器，允许你在循环内部修改它的属性（如`ns.last_query_index`），并且在循环结束后这些修改依然有效。
- `{%- for message in messages[::-1] -%}`:
  - **含义**: **倒序遍历**`messages`列表！`[::-1]`是Python中列表反转的写法。
  - **作用**: 这是为了从对话的**结尾**开始，向前寻找最后一个**真正的用户问题**。因为在工具调用中，`tool`角色的返回结果可能会被格式化为`user`角色，所以需要一种方式来区分它们。
- `{%- if ... -%}`:
  - **含义**: 这个复杂的`if`条件正在寻找一个**不是工具返回结果的、普通的用户消息**。`not (message.content.startswith("<tool_response>"))`就是它的判断依据。
  - **作用**: 一旦从后往前找到了第一个这样的用户消息，就把它的索引记录在`ns.last_query_index`中，并停止继续寻找。这个索引对于后面判断哪部分是思考、哪部分是总结至关重要。

------



### **第三部分：主循环与消息格式化**



这是模板的主体，它按顺序遍历每一条消息，并根据其`role`应用不同的格式化规则。

```Django
{%- for message in messages -%}
	...
	{%- if message.role == "user" or message.role == "system" and not loop.first -%}
		{{- "<|im_start|>" + message.role + "\n" + content + "<|im_end|>" + "\n" -}}
        
	{%- elif message.role == "assistant" -%}
		... (处理 assistant 角色, 包括 <think> 和 tool_calls) ...
        
	{%- elif message.role == "tool" -%}
		... (处理 tool 角色, 包括 tool_response) ...
	{%- endif -%}
{%- endfor -%}
```

**分角色解释:**

#### **User / System 角色**

- `{%- if message.role == "user" or message.role == "system" and not loop.first -%}`:
  - **含义**: 如果角色是`user`，或者角色是`system`但不是第一条消息（第一条已在前面处理），则执行。
  - **作用**: 这是最简单的格式化，直接用`<|im_start|>`和`<|im_end|>`把角色和内容包裹起来。

#### **Assistant 角色**

- 这一块是模板最复杂的部分，我们之前已经详细讨论过。它的逻辑是：
  1. `{%- if "</think>" in content -%}`: 检查`content`中是否有`<think>`标签。
  2. `{%- set reasoning_content = ... %}`: 如果有，将思考内容**提取**到`reasoning_content`变量。
  3. `{%- set content = ... %}`: 同时，从`content`变量中**移除**思考部分，只留下实际回答。
  4. `{%- if loop.index0 > ns.last_query_index -%}`: 使用第二部分计算出的`last_query_index`来判断当前是否处于工具调用的思考/总结阶段。
  5. `{%- if loop.last or not loop.last and reasoning_content -%}`: 根据是否为最后一条消息、是否包含思考内容，来**决定**最终要不要把`<think>`标签和`reasoning_content`打印出来。
  6. `{%- for tool_call in message.tool_calls -%}`: 如果消息中有`tool_calls`，遍历它们并格式化为`<tool_call>{...}</tool_call>`。



#### **Tool 角色**

- `{%- elif message.role == "tool" -%}`:
  - `{%- if loop.first or messages[loop.index0 - 1].role != "tool" -%}`:
    - **含义**: 检查这是不是第一条消息，或者**它的前一条消息的角色不是`tool`**。
    - **作用**: 这是一个聪明的技巧，用于将**连续的多条`tool`消息**合并到同一个`user`消息块中。只有在连续`tool`消息的第一条前面，才会加上`<|im_start|>user`。
  - `{{- "\n<tool_response>\n" ... "\n</tool_response>" -}}`:
    - **含义**: 将`tool`消息的内容包裹在`<tool_response>`标签里。
  - `{%- if loop.last or messages[loop.index0 + 1].role != "tool" -%}`:
    - **含义**: 检查这是不是最后一条消息，或者**它的后一条消息的角色不是`tool`**。
    - **作用**: 只有在连续`tool`消息的最后一条后面，才会加上`<|im_end|>\n`，从而封闭整个`user`消息块。

------



### **第四部分：添加生成提示 (Generation Prompt)**

这部分用于推理，告诉模型接下来该它说话了。

Django

```Django
{%- if add_generation_prompt -%}
	{{- "<|im_start|>assistant\n" -}}
	{%- if enable_thinking is defined and enable_thinking is false -%}
		{{- "<think>\n\n</think>\n\n" -}}
	{%- endif -%}
{%- endif -%}
```

**逐行解释:**

- `{%- if add_generation_prompt -%}`:
  - **含义**: 判断在调用`apply_chat_template`时是否传入了`add_generation_prompt=True`。
  - **作用**: 控制是否在末尾添加引导模型生成的提示。在准备训练数据时通常为`False`，在进行推理时为`True`。
- `{{- "<|im_start|>assistant\n" -}}`:
  - **含义**: 输出一个`assistant`的起始标记，但不封闭。
  - **作用**: 提示模型，轮到你了，请以`assistant`的身份开始生成内容。