#!/usr/bin/env python3
"""
代码语法检查脚本
检查 decoder_only_transformer.py 的语法正确性
"""

import ast
import sys

def check_syntax(filename):
    """检查 Python 文件的语法"""
    print(f"检查文件: {filename}")
    print("=" * 60)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 解析 AST
        tree = ast.parse(code, filename=filename)
        
        # 统计信息
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}")
        
        print("✓ 语法检查通过!")
        print("\n统计信息:")
        print(f"  - 导入模块数: {len(set(imports))}")
        print(f"  - 类定义数: {len(classes)}")
        print(f"  - 函数定义数: {len(functions)}")
        
        print("\n类列表:")
        for cls in classes:
            print(f"  ✓ {cls}")
        
        print("\n顶层函数列表:")
        # 只显示非方法的函数
        module_functions = [f for f in functions if f not in ['__init__', 'forward', 'generate']]
        for func in set(module_functions[:20]):  # 限制显示前20个
            print(f"  ✓ {func}")
        
        # 检查关键组件
        print("\n关键组件检查:")
        required_classes = [
            'RMSNorm',
            'MultiHeadAttention',
            'SwiGLU_FFN',
            'DecoderBlock',
            'DecoderOnlyTransformer'
        ]
        
        required_functions = [
            'compute_freqs_cis',
            'apply_rope',
            'top_p_sampling'
        ]
        
        for cls in required_classes:
            if cls in classes:
                print(f"  ✓ {cls} 类已定义")
            else:
                print(f"  ✗ {cls} 类未找到")
                return False
        
        for func in required_functions:
            if func in functions:
                print(f"  ✓ {func} 函数已定义")
            else:
                print(f"  ✗ {func} 函数未找到")
                return False
        
        print("\n" + "=" * 60)
        print("所有检查通过! ✓")
        return True
        
    except SyntaxError as e:
        print(f"\n✗ 语法错误:")
        print(f"  文件: {e.filename}")
        print(f"  行号: {e.lineno}")
        print(f"  错误: {e.msg}")
        print(f"  代码: {e.text}")
        return False
    
    except Exception as e:
        print(f"\n✗ 检查失败: {e}")
        return False

if __name__ == "__main__":
    filename = "decoder_only_transformer.py"
    success = check_syntax(filename)
    sys.exit(0 if success else 1)
