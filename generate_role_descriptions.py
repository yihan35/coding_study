#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from openai import OpenAI
from typing import List, Dict, Any, Set
import argparse
from collections import defaultdict

# ================= API配置 =================
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://10.244.78.132:8010/v1"  # 请替换为你的API地址

# ================= 初始化OpenAI客户端 =================
try:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    print("✅ OpenAI客户端初始化成功")
except Exception as e:
    print(f"❌ Error initializing OpenAI client: {e}")
    print("Please make sure the OPENAI_API_BASE URL is correct and the server is running.")
    client = None

def predict(query_messages: List[Dict[str, str]]) -> str:
    """调用大模型API生成响应"""
    if not client:
        print("Client not initialized. Cannot make prediction.")
        return "Error: Client not initialized."
    
    max_try = 5
    while max_try > 0:
        try:
            model_name = client.models.list().data[0].id
            print(f"---> [INFO] Using model '{model_name}' to generate response...")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=query_messages,
                temperature=0.8,  # 稍微提高温度以增加多样性
                top_p=0.9,
                max_tokens=1024,
                extra_body={
                    "repetition_penalty": 1.05,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            content = response.choices[0].message.content.strip()
            print(f"<--- [SUCCESS] Model returned content successfully.")
            return content
            
        except Exception as e:
            max_try -= 1
            print(f"[ERROR] API call failed: {e}. Retrying... ({max_try} attempts left)")
            if max_try == 0:
                return f"Error: API call failed after multiple retries. Last error: {e}"
    
    return "Error: Maximum retries exceeded."

def create_enrichment_prompt(role_name: str, original_description: str) -> List[Dict[str, str]]:
    """
    创建用于丰富描述的prompt
    """
    prompt = f"""你是一个专业的角色设定专家，需要为AI智能体创建丰富、详细且专业的角色描述。

任务要求：
1. 基于给定的角色名称和原始描述，创建一个更加丰富、详细的角色描述
2. 描述必须使用第三人称"你是..."的格式，不能使用"我是..."
3. 描述要包含角色的专业能力、服务范围、特点优势等多个维度
4. 语言要专业、准确、有吸引力
5. 描述长度控制在100-200字之间
6. 突出角色的独特性和专业性

角色信息：
- 角色名称：{role_name}
- 原始描述：{original_description}

请基于以上信息，生成一个丰富、专业的角色描述。只需要输出最终的描述内容，不要包含其他解释。

丰富后的描述："""

    return [{"role": "user", "content": prompt}]

def generate_enriched_description(role_name: str, original_description: str) -> str:
    """
    生成丰富的角色描述
    """
    messages = create_enrichment_prompt(role_name, original_description)
    response = predict(messages)
    
    if response.startswith("Error:"):
        print(f"⚠️  生成描述失败，使用原始描述: {response}")
        return original_description
    
    # 清理生成的描述
    enriched_description = response.strip()
    
    # 确保使用"你是"开头
    if not enriched_description.startswith("你是"):
        if enriched_description.startswith("我是"):
            enriched_description = enriched_description.replace("我是", "你是", 1)
        elif not enriched_description.startswith("你"):
            enriched_description = f"你是{enriched_description}"
    
    return enriched_description

def analyze_roles(input_file: str) -> Dict[str, Dict[str, Any]]:
    """
    分析文件中的所有角色，统计每个角色的信息
    """
    role_info = defaultdict(lambda: {
        'original_descriptions': set(),
        'count': 0,
        'sample_data': None
    })
    
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                role_name = data.get('role', '未知角色')
                description = data.get('description', '')
                
                role_info[role_name]['original_descriptions'].add(description)
                role_info[role_name]['count'] += 1
                
                # 保存第一个样本数据
                if role_info[role_name]['sample_data'] is None:
                    role_info[role_name]['sample_data'] = data
                    
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    print(f"📊 文件分析完成:")
    print(f"总数据条数: {total_lines}")
    print(f"发现角色数量: {len(role_info)}")
    
    return dict(role_info)

def generate_role_descriptions(role_info: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    为每个角色生成统一的丰富描述
    """
    enriched_descriptions = {}
    
    print(f"\n🚀 开始为 {len(role_info)} 个角色生成丰富描述...")
    
    for i, (role_name, info) in enumerate(role_info.items(), 1):
        print(f"\n[{i}/{len(role_info)}] 处理角色: {role_name}")
        print(f"数据条数: {info['count']}")
        
        # 选择最常见的原始描述，或者最长的描述
        original_descriptions = list(info['original_descriptions'])
        if len(original_descriptions) == 1:
            selected_description = original_descriptions[0]
        else:
            # 选择最长的描述作为基础
            selected_description = max(original_descriptions, key=len)
        
        print(f"原始描述: {selected_description}")
        
        try:
            enriched_desc = generate_enriched_description(role_name, selected_description)
            enriched_descriptions[role_name] = enriched_desc
            print(f"✅ 丰富描述: {enriched_desc[:80]}...")
            
        except Exception as e:
            print(f"❌ 生成描述失败: {e}")
            enriched_descriptions[role_name] = selected_description
    
    return enriched_descriptions

def process_file_with_enriched_descriptions(
    input_file: str, 
    output_file: str, 
    role_descriptions: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    使用丰富的描述处理整个文件
    """
    processed_data = []
    total_lines = 0
    success_count = 0
    error_count = 0
    
    print(f"\n📝 开始更新文件中的描述...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                role_name = data.get('role', '未知角色')
                
                # 更新描述
                if role_name in role_descriptions:
                    original_desc = data.get('description', '')
                    data['description'] = role_descriptions[role_name]
                    
                    # 添加处理信息
                    if 'processing_info' not in data:
                        data['processing_info'] = {}
                    data['processing_info']['description_enriched'] = True
                    data['processing_info']['original_description'] = original_desc
                    data['processing_info']['enrichment_line'] = line_num
                
                processed_data.append(data)
                success_count += 1
                
                if line_num % 100 == 0:
                    print(f"已处理 {line_num} 行...")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️  第{line_num}行JSON解析错误: {e}")
                error_count += 1
                continue
            except Exception as e:
                print(f"⚠️  第{line_num}行处理错误: {e}")
                error_count += 1
                continue
    
    print(f"\n📊 文件处理完成!")
    print(f"总行数: {total_lines}")
    print(f"成功处理: {success_count}")
    print(f"错误数量: {error_count}")
    
    # 保存结果
    save_results(processed_data, output_file)
    
    return processed_data

def save_results(data: List[Dict[str, Any]], output_file: str):
    """
    保存处理结果到JSONL文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"💾 结果已保存到: {output_file}")
        print(f"📄 共保存了 {len(data)} 条数据")
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

def preview_enriched_descriptions(role_descriptions: Dict[str, str], limit: int = 5):
    """
    预览生成的丰富描述
    """
    if not role_descriptions:
        print("📭 没有生成任何丰富描述")
        return
    
    print(f"\n🎯 丰富描述预览 (显示前{min(limit, len(role_descriptions))}个角色):")
    print("=" * 100)
    
    for i, (role_name, description) in enumerate(list(role_descriptions.items())[:limit], 1):
        print(f"\n[{i}] 角色: {role_name}")
        print(f"丰富描述: {description}")
        print("-" * 80)

def save_role_descriptions(role_descriptions: Dict[str, str], output_file: str):
    """
    单独保存角色描述映射表
    """
    try:
        descriptions_data = {
            "role_descriptions": role_descriptions,
            "total_roles": len(role_descriptions),
            "generation_info": "Generated by description enricher script"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(descriptions_data, ensure_ascii=False, indent=2))
        
        print(f"📋 角色描述映射表已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存角色描述映射表时出错: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='丰富和统一角色描述')
    parser.add_argument('input_file', help='输入的JSONL文件路径')
    parser.add_argument('output_file', help='输出的JSONL文件路径')
    parser.add_argument('--descriptions-output', help='角色描述映射表输出文件路径（可选）')
    parser.add_argument('--preview', type=int, default=5, help='预览角色描述的数量（默认5）')
    
    args = parser.parse_args()
    
    try:
        # 第一步：分析所有角色
        print("🔍 第一步：分析文件中的角色...")
        role_info = analyze_roles(args.input_file)
        
        # 第二步：为每个角色生成丰富描述
        print("\n🎨 第二步：为每个角色生成丰富描述...")
        role_descriptions = generate_role_descriptions(role_info)
        
        # 预览生成的描述
        preview_enriched_descriptions(role_descriptions, args.preview)
        
        # 保存角色描述映射表（如果指定了输出文件）
        if args.descriptions_output:
            save_role_descriptions(role_descriptions, args.descriptions_output)
        
        # 第三步：使用丰富描述更新整个文件
        print("\n📝 第三步：更新文件中的所有描述...")
        processed_data = process_file_with_enriched_descriptions(
            args.input_file, 
            args.output_file, 
            role_descriptions
        )
        
        print(f"\n🎉 处理完成!")
        print(f"共处理了 {len(role_descriptions)} 个不同角色")
        print(f"更新了 {len(processed_data)} 条数据的描述")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")

if __name__ == "__main__":
    # 如果直接运行脚本，可以在这里测试
    if len(os.sys.argv) == 1:
        # 测试模式
        print("🧪 测试模式 - 请提供输入和输出文件路径作为命令行参数")
        print("用法: python script.py <input_file> <output_file> [--descriptions-output mapping.json] [--preview N]")
        print("\n示例:")
        print("python script.py data.jsonl enriched_data.jsonl --descriptions-output role_mappings.json --preview 10")
        print("\n功能说明:")
        print("1. 分析文件中的所有角色")
        print("2. 为每个角色生成统一的丰富描述")
        print("3. 更新文件中所有相同角色的描述")
        print("4. 可选：保存角色描述映射表")
    else:
        main()


'''
python description_enricher.py \
    input.jsonl \
    output.jsonl \
    --descriptions-output role_mappings.json

'''