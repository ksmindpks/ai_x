import ast

# 위 예시 코드를 문자열로 가져옵니다. (실제로는 파일을 읽어오게 됩니다)
with open("example.py", "r", encoding="utf-8") as f:
    source_code = f.read()

# 1. 코드를 AST 트리 구조로 파싱합니다.
tree = ast.parse(source_code)

print("--- Docstring 추출 결과 ---")

# 2. 모듈 레벨의 Docstring을 추출합니다.
module_docstring = ast.get_docstring(tree)
print(f"모듈 Docstring: \n{module_docstring}\n")


# 3. 트리를 순회하며 함수와 클래스를 찾습니다.
for node in tree.body:
    # 만약 노드가 함수 정의(FunctionDef)라면
    if isinstance(node, ast.FunctionDef):
        function_name = node.name
        docstring = ast.get_docstring(node)
        print(f"함수 '{function_name}'의 Docstring: \n{docstring}\n")
    
    # 만약 노드가 클래스 정의(ClassDef)라면
    elif isinstance(node, ast.ClassDef):
        class_name = node.name
        docstring = ast.get_docstring(node)
        print(f"클래스 '{class_name}'의 Docstring: \n{docstring}\n")
        
        # 클래스 내부의 메서드도 찾을 수 있습니다.
        for method_node in node.body:
            if isinstance(method_node, ast.FunctionDef):
                method_name = method_node.name
                method_docstring = ast.get_docstring(method_node)
                print(f"  - 메서드 '{method_name}'의 Docstring: \n    {method_docstring}\n")