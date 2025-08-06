import os
import requests
import ast
from datetime import datetime
from collections import defaultdict, Counter
import json
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedRepositoryAnalyzer:
    def __init__(self, github_user, github_repo, github_branch="main"):
        self.github_user = github_user
        self.github_repo = github_repo
        self.github_branch = github_branch
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        
        # 개선된 분석 결과 구조
        self.analysis_results = {
            "repository_info": {},
            "file_statistics": {},
            "directory_structure": {},
            "directory_tree": {},  # 새로 추가: 계층적 트리 구조
            "code_analysis": {},
            "documentation_analysis": {},
            "file_size_analysis": {},  # 새로 추가: 파일 크기 분석
            "summary": {}
        }
    
    def build_directory_tree(self, tree_items: List[Dict]) -> Dict:
        """계층적 디렉토리 트리 구조 생성"""
        tree_structure = {
            "name": "root",
            "type": "directory",
            "children": {},
            "file_count": 0,
            "total_size": 0,
            "file_types": Counter()
        }
        
        for item in tree_items:
            path_parts = item["path"].split("/")
            current_node = tree_structure
            
            # 경로를 따라 트리 구조 생성
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:  # 마지막 부분 (파일 또는 최종 디렉토리)
                    if item["type"] == "blob":  # 파일
                        current_node["file_count"] += 1
                        if "size" in item:
                            current_node["total_size"] += item.get("size", 0)
                        
                        # 파일 확장자 추출
                        if "." in part:
                            ext = part.split(".")[-1].lower()
                            current_node["file_types"][ext] += 1
                    else:  # 디렉토리
                        if part not in current_node["children"]:
                            current_node["children"][part] = {
                                "name": part,
                                "type": "directory",
                                "children": {},
                                "file_count": 0,
                                "total_size": 0,
                                "file_types": Counter()
                            }
                else:  # 중간 디렉토리
                    if part not in current_node["children"]:
                        current_node["children"][part] = {
                            "name": part,
                            "type": "directory",
                            "children": {},
                            "file_count": 0,
                            "total_size": 0,
                            "file_types": Counter()
                        }
                    current_node = current_node["children"][part]
        
        return tree_structure
    
    def analyze_file_structure_deep(self):
        """심층 파일 구조 및 통계 분석"""
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/git/trees/{self.github_branch}?recursive=1"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            tree = response.json().get("tree", [])
            
            # 기본 통계
            file_extensions = Counter()
            directory_depth = defaultdict(int)  # 깊이별 파일 수
            path_components = defaultdict(set)  # 각 레벨의 고유 디렉토리
            total_files = 0
            total_dirs = 0
            total_size = 0
            
            # 디렉토리별 상세 정보
            directory_details = defaultdict(lambda: {
                "file_count": 0,
                "subdirs": set(),
                "total_size": 0,
                "file_types": Counter(),
                "depth": 0
            })
            
            for item in tree:
                path = item["path"]
                
                if item["type"] == "blob":  # 파일
                    total_files += 1
                    size = item.get("size", 0)
                    total_size += size
                    
                    # 확장자 추출
                    if "." in path:
                        ext = path.split(".")[-1].lower()
                        file_extensions[ext] += 1
                    
                    # 경로 분석
                    parts = path.split("/")
                    depth = len(parts) - 1
                    directory_depth[depth] += 1
                    
                    # 각 디렉토리 레벨 분석
                    current_path = ""
                    for i, part in enumerate(parts[:-1]):  # 파일명 제외
                        parent_path = current_path
                        current_path = part if not current_path else f"{current_path}/{part}"
                        
                        directory_details[current_path]["file_count"] += 1
                        directory_details[current_path]["total_size"] += size
                        directory_details[current_path]["depth"] = i + 1
                        
                        if "." in parts[-1]:
                            ext = parts[-1].split(".")[-1].lower()
                            directory_details[current_path]["file_types"][ext] += 1
                        
                        # 부모-자식 관계 기록
                        if parent_path and parent_path != current_path:
                            directory_details[parent_path]["subdirs"].add(part)
                        
                        path_components[i].add(part)
                    
                    # 루트 레벨 파일
                    if depth == 0:
                        directory_details["root"]["file_count"] += 1
                        directory_details["root"]["total_size"] += size
                        if "." in path:
                            ext = path.split(".")[-1].lower()
                            directory_details["root"]["file_types"][ext] += 1
                
                elif item["type"] == "tree":  # 디렉토리
                    total_dirs += 1
            
            # 트리 구조 생성
            tree_structure = self.build_directory_tree(tree)
            
            # 가장 큰 디렉토리들 찾기
            sorted_dirs = sorted(
                [(path, info["file_count"], info["total_size"]) 
                 for path, info in directory_details.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]  # 상위 20개
            
            # 깊이 통계
            max_depth = max(directory_depth.keys()) if directory_depth else 0
            avg_depth = sum(d * count for d, count in directory_depth.items()) / total_files if total_files > 0 else 0
            
            self.analysis_results["file_statistics"] = {
                "total_files": total_files,
                "total_directories": total_dirs,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_extensions": dict(file_extensions.most_common(15)),
                "max_depth": max_depth,
                "average_depth": round(avg_depth, 2),
                "files_by_depth": dict(directory_depth)
            }
            
            self.analysis_results["directory_structure"] = {
                "top_directories": [
                    {
                        "path": path,
                        "file_count": count,
                        "size_mb": round(size / (1024 * 1024), 2)
                    }
                    for path, count, size in sorted_dirs
                ],
                "directory_count_by_level": {
                    f"level_{i}": len(dirs) 
                    for i, dirs in path_components.items()
                }
            }
            
            self.analysis_results["directory_tree"] = tree_structure
            
            # 디렉토리별 상세 정보 (dict를 JSON 직렬화 가능하게 변환)
            self.analysis_results["directory_details"] = {
                path: {
                    "file_count": info["file_count"],
                    "subdirs_count": len(info["subdirs"]),
                    "total_size_mb": round(info["total_size"] / (1024 * 1024), 2),
                    "depth": info["depth"],
                    "main_file_types": dict(info["file_types"].most_common(5))
                }
                for path, info in list(directory_details.items())[:50]  # 상위 50개만
            }
            
            logger.info(f"심층 파일 구조 분석 완료: {total_files}개 파일, {total_dirs}개 디렉토리, 최대 깊이 {max_depth}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"파일 구조 분석 실패: {e}")
    
    def generate_tree_visualization(self, node: Dict, prefix: str = "", is_last: bool = True, max_depth: int = 3, current_depth: int = 0) -> str:
        """디렉토리 트리 시각화 생성"""
        if current_depth > max_depth:
            return ""
        
        result = ""
        
        # 현재 노드 출력 (루트가 아닌 경우)
        if node["name"] != "root":
            connector = "└── " if is_last else "├── "
            result += f"{prefix}{connector}{node['name']}/ ({node['file_count']} files, {round(node['total_size'] / 1024, 1)}KB)\n"
            prefix += "    " if is_last else "│   "
        
        # 자식 노드들 정렬 및 출력
        children = sorted(node["children"].items(), key=lambda x: x[0])
        for i, (child_name, child_node) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            result += self.generate_tree_visualization(
                child_node, prefix, is_last_child, max_depth, current_depth + 1
            )
        
        return result
    
    def analyze_python_code_comprehensive(self, max_files: int = 200):
        """포괄적인 Python 코드 분석 (더 많은 파일 분석)"""
        if self.analysis_results["file_statistics"].get("file_extensions", {}).get("py", 0) == 0:
            logger.info("Python 파일이 없어 코드 분석을 건너뜁니다.")
            return
        
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/git/trees/{self.github_branch}?recursive=1"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            tree = response.json().get("tree", [])
            
            # 모든 Python 파일 찾기
            all_python_files = [f for f in tree if f["path"].endswith(".py")]
            analyzed_files = all_python_files[:max_files]
            
            # 디렉토리별 Python 파일 분포
            python_file_distribution = defaultdict(int)
            for file_info in all_python_files:
                parts = file_info["path"].split("/")
                if len(parts) > 1:
                    python_file_distribution[parts[0]] += 1
                else:
                    python_file_distribution["root"] += 1
            
            total_functions = 0
            total_classes = 0
            total_docstrings = 0
            documented_functions = 0
            documented_classes = 0
            total_lines = 0
            imports_count = Counter()
            
            complexity_stats = {
                "simple": 0,
                "medium": 0,
                "complex": 0,
                "very_complex": 0  # 100줄 이상
            }
            
            for file_info in analyzed_files:
                raw_url = f"https://raw.githubusercontent.com/{self.github_user}/{self.github_repo}/{self.github_branch}/{file_info['path']}"
                
                try:
                    file_response = requests.get(raw_url)
                    if file_response.status_code == 200:
                        code = file_response.text
                        total_lines += len(code.splitlines())
                        
                        try:
                            tree_ast = ast.parse(code)
                            
                            for node in ast.walk(tree_ast):
                                # Import 분석
                                if isinstance(node, (ast.Import, ast.ImportFrom)):
                                    if isinstance(node, ast.ImportFrom) and node.module:
                                        base_module = node.module.split('.')[0]
                                        imports_count[base_module] += 1
                                    elif isinstance(node, ast.Import):
                                        for alias in node.names:
                                            base_module = alias.name.split('.')[0]
                                            imports_count[base_module] += 1
                                
                                # 함수 분석
                                elif isinstance(node, ast.FunctionDef):
                                    total_functions += 1
                                    
                                    if ast.get_docstring(node):
                                        documented_functions += 1
                                        total_docstrings += 1
                                    
                                    # 복잡도 분석
                                    lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                                    if lines < 10:
                                        complexity_stats["simple"] += 1
                                    elif lines <= 50:
                                        complexity_stats["medium"] += 1
                                    elif lines <= 100:
                                        complexity_stats["complex"] += 1
                                    else:
                                        complexity_stats["very_complex"] += 1
                                
                                # 클래스 분석
                                elif isinstance(node, ast.ClassDef):
                                    total_classes += 1
                                    
                                    if ast.get_docstring(node):
                                        documented_classes += 1
                                        total_docstrings += 1
                        
                        except SyntaxError:
                            continue
                
                except requests.exceptions.RequestException:
                    continue
            
            # 문서화 비율 계산
            func_doc_rate = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            class_doc_rate = (documented_classes / total_classes * 100) if total_classes > 0 else 0
            
            # 가장 많이 사용된 import
            top_imports = imports_count.most_common(15)
            
            self.analysis_results["code_analysis"] = {
                "total_python_files": len(all_python_files),
                "analyzed_files": len(analyzed_files),
                "total_lines_of_code": total_lines,
                "avg_lines_per_file": round(total_lines / len(analyzed_files), 1) if analyzed_files else 0,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "total_docstrings": total_docstrings,
                "documented_functions": documented_functions,
                "documented_classes": documented_classes,
                "function_documentation_rate": round(func_doc_rate, 1),
                "class_documentation_rate": round(class_doc_rate, 1),
                "complexity_distribution": complexity_stats,
                "python_file_distribution": dict(sorted(python_file_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_imports": top_imports
            }
            
            logger.info(f"Python 코드 분석 완료: {len(analyzed_files)}개 파일 분석, 총 {total_lines}줄")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Python 코드 분석 실패: {e}")
    
    # 기존 메서드들은 그대로 유지
    def get_repository_info(self):
        """GitHub API로 저장소 기본 정보 수집"""
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            repo_data = response.json()
            
            self.analysis_results["repository_info"] = {
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description"),
                "language": repo_data.get("language"),
                "stars": repo_data.get("stargazers_count"),
                "forks": repo_data.get("forks_count"),
                "issues": repo_data.get("open_issues_count"),
                "size": repo_data.get("size"),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
                "url": repo_data.get("html_url"),
                "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None,
                "topics": repo_data.get("topics", []),
                "watchers": repo_data.get("watchers_count"),
                "default_branch": repo_data.get("default_branch")
            }
            
            logger.info(f"저장소 정보 수집 완료: {repo_data.get('full_name')}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"저장소 정보 수집 실패: {e}")
    
    def analyze_documentation(self):
        """문서화 파일 분석 (README, docs 등)"""
        doc_files = ["README.md", "README.rst", "README.txt", "CHANGELOG.md", 
                     "CONTRIBUTING.md", "LICENSE", "CODE_OF_CONDUCT.md", "SECURITY.md"]
        found_docs = []
        
        for doc_file in doc_files:
            url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/contents/{doc_file}"
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    found_docs.append(doc_file)
            except:
                continue
        
        # docs 디렉토리 확인
        docs_url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/contents/docs"
        has_docs_dir = False
        docs_file_count = 0
        
        try:
            response = requests.get(docs_url, headers=self.headers)
            if response.status_code == 200:
                has_docs_dir = True
                docs_content = response.json()
                docs_file_count = len([f for f in docs_content if f["type"] == "file"])
        except:
            pass
        
        self.analysis_results["documentation_analysis"] = {
            "documentation_files": found_docs,
            "has_docs_directory": has_docs_dir,
            "docs_file_count": docs_file_count,
            "documentation_score": len(found_docs) + (2 if has_docs_dir else 0) + (min(docs_file_count, 3))
        }
        
        logger.info(f"문서화 분석 완료: {len(found_docs)}개 문서 파일")
    
    def generate_summary(self):
        """전체 분석 결과 요약"""
        repo_info = self.analysis_results["repository_info"]
        file_stats = self.analysis_results["file_statistics"]
        code_analysis = self.analysis_results["code_analysis"]
        doc_analysis = self.analysis_results["documentation_analysis"]
        
        # 프로젝트 규모 평가
        file_count = file_stats.get("total_files", 0)
        if file_count < 10:
            project_size = "Small"
        elif file_count < 100:
            project_size = "Medium"
        elif file_count < 1000:
            project_size = "Large"
        else:
            project_size = "Very Large"
        
        # 문서화 품질 평가
        doc_score = doc_analysis.get("documentation_score", 0)
        if doc_score >= 8:
            doc_quality = "Excellent"
        elif doc_score >= 5:
            doc_quality = "Good"
        elif doc_score >= 2:
            doc_quality = "Basic"
        else:
            doc_quality = "Poor"
        
        # 코드 품질 평가
        code_quality = "N/A"
        if code_analysis.get("total_functions", 0) > 0:
            func_doc_rate = code_analysis.get("function_documentation_rate", 0)
            if func_doc_rate >= 80:
                code_quality = "Excellent"
            elif func_doc_rate >= 60:
                code_quality = "Good"
            elif func_doc_rate >= 30:
                code_quality = "Fair"
            else:
                code_quality = "Needs Improvement"
        
        # 프로젝트 복잡도 평가
        max_depth = file_stats.get("max_depth", 0)
        if max_depth <= 3:
            structure_complexity = "Simple"
        elif max_depth <= 5:
            structure_complexity = "Moderate"
        elif max_depth <= 7:
            structure_complexity = "Complex"
        else:
            structure_complexity = "Very Complex"
        
        self.analysis_results["summary"] = {
            "project_size": project_size,
            "structure_complexity": structure_complexity,
            "documentation_quality": doc_quality,
            "code_quality": code_quality,
            "main_language": repo_info.get("language", "Unknown"),
            "activity_level": "Active" if repo_info.get("updated_at") else "Unknown",
            "community_engagement": f"{repo_info.get('stars', 0)} stars, {repo_info.get('forks', 0)} forks, {repo_info.get('watchers', 0)} watchers"
        }
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        logger.info("개선된 저장소 분석 시작...")
        
        self.get_repository_info()
        self.analyze_file_structure_deep()  # 개선된 메서드 사용
        self.analyze_python_code_comprehensive()  # 개선된 메서드 사용
        self.analyze_documentation()
        self.generate_summary()
        
        logger.info("저장소 분석 완료!")
        return self.analysis_results


class ImprovedReadmeGenerator:
    def __init__(self, analysis_results):
        self.analysis = analysis_results
    
    def generate_badges(self):
        """배지 생성"""
        repo_info = self.analysis["repository_info"]
        code_analysis = self.analysis["code_analysis"]
        summary = self.analysis["summary"]
        
        badges = []
        
        # GitHub 배지들
        if repo_info.get("stars", 0) > 0:
            badges.append(f"![GitHub stars](https://img.shields.io/github/stars/{repo_info.get('full_name')}?style=flat-square)")
        
        if repo_info.get("forks", 0) > 0:
            badges.append(f"![GitHub forks](https://img.shields.io/github/forks/{repo_info.get('full_name')}?style=flat-square)")
        
        if repo_info.get("issues", 0) > 0:
            badges.append(f"![GitHub issues](https://img.shields.io/github/issues/{repo_info.get('full_name')}?style=flat-square)")
        
        # 언어 배지
        if repo_info.get("language"):
            badges.append(f"![Language](https://img.shields.io/badge/language-{repo_info.get('language')}-blue?style=flat-square)")
        
        # 라이센스 배지
        if repo_info.get("license"):
            license_name = repo_info.get("license").replace(" ", "%20")
            badges.append(f"![License](https://img.shields.io/badge/license-{license_name}-green?style=flat-square)")
        
        # 문서화 품질 배지
        doc_quality = summary.get("documentation_quality", "Unknown")
        color = {"Excellent": "brightgreen", "Good": "green", "Basic": "yellow", "Poor": "red"}.get(doc_quality, "lightgrey")
        badges.append(f"![Documentation](https://img.shields.io/badge/docs-{doc_quality}-{color}?style=flat-square)")
        
        # 코드 품질 배지
        if code_analysis.get("total_functions", 0) > 0:
            code_quality = summary.get("code_quality", "N/A")
            color = {"Excellent": "brightgreen", "Good": "green", "Fair": "yellow", "Needs Improvement": "red"}.get(code_quality, "lightgrey")
            badges.append(f"![Code Quality](https://img.shields.io/badge/code%20quality-{code_quality.replace(' ', '%20')}-{color}?style=flat-square)")
        
        return " ".join(badges)
    
    def generate_readme(self):
        """개선된 README.md 생성"""
        repo_info = self.analysis["repository_info"]
        file_stats = self.analysis["file_statistics"]
        dir_structure = self.analysis["directory_structure"]
        code_analysis = self.analysis["code_analysis"]
        doc_analysis = self.analysis["documentation_analysis"]
        summary = self.analysis["summary"]
        
        badges = self.generate_badges()
        
        readme_content = f"""# 📊 {repo_info.get('name', 'Repository')} Deep Analysis Report

{badges}

## 🏠 Repository Overview

**🔗 Repository:** [{repo_info.get('full_name')}]({repo_info.get('url')})  
**📝 Description:** {repo_info.get('description', 'No description provided')}  
**💻 Primary Language:** {repo_info.get('language', 'Unknown')}  
**⚖️ License:** {repo_info.get('license', 'Not specified')}  
**🌿 Default Branch:** {repo_info.get('default_branch', 'main')}  

### ⭐ Community Stats
- **🌟 Stars:** {repo_info.get('stars', 0):,}
- **🍴 Forks:** {repo_info.get('forks', 0):,}
- **👀 Watchers:** {repo_info.get('watchers', 0):,}
- **🐛 Open Issues:** {repo_info.get('issues', 0):,}
- **📦 Repository Size:** {repo_info.get('size', 0):,} KB

### 📅 Timeline
- **🎉 Created:** {repo_info.get('created_at', 'Unknown')[:10]}
- **🔄 Last Updated:** {repo_info.get('updated_at', 'Unknown')[:10]}

## 📁 Deep File Structure Analysis

### 📊 Overall Statistics
- **📄 Total Files:** {file_stats.get('total_files', 0):,}
- **📂 Total Directories:** {file_stats.get('total_directories', 0):,}
- **💾 Total Size:** {file_stats.get('total_size_mb', 0):,} MB
- **🏗️ Max Directory Depth:** {file_stats.get('max_depth', 0)}
- **📐 Average File Depth:** {file_stats.get('average_depth', 0)}

### 📊 Directory Depth Distribution
"""
        # 깊이별 파일 분포
        depth_dist = file_stats.get("files_by_depth", {})
        if depth_dist:
            for depth, count in sorted(depth_dist.items()):
                bar_length = min(int(count / max(depth_dist.values()) * 30), 30)
                bar = "█" * bar_length
                readme_content += f"- **Level {depth}:** {bar} {count:,} files\n"

        readme_content += f"""
### 🗂️ File Types Distribution
"""
        # 파일 확장자별 분포
        extensions = file_stats.get("file_extensions", {})
        total_files = file_stats.get('total_files', 1)
        for ext, count in extensions.items():
            percentage = (count / total_files) * 100
            icon = {"py": "🐍", "js": "📜", "html": "🌐", "css": "🎨", "md": "📝", 
                    "json": "📋", "txt": "📄", "yml": "⚙️", "yaml": "⚙️",
                    "java": "☕", "cpp": "⚡", "c": "🔷", "go": "🐹", "rs": "🦀"}.get(ext, "📄")
            readme_content += f"- **{icon} {ext.upper()}:** {count:,} files ({percentage:.1f}%)\n"

        readme_content += f"""
### 📂 Top Directories by File Count
"""
        # 상위 디렉토리들
        top_dirs = dir_structure.get("top_directories", [])[:10]
        for dir_info in top_dirs:
            path = dir_info["path"]
            icon = {"src": "📁", "lib": "📚", "docs": "📖", "test": "🧪", 
                    "tests": "🧪", "examples": "💡", "tools": "🔧", 
                    "scripts": "📜", "assets": "🎯", "static": "🗂️",
                    "bin": "⚙️", "dist": "📦", "build": "🏗️"}.get(path.split("/")[0].lower(), "📁")
            readme_content += f"- **{icon} /{path}:** {dir_info['file_count']:,} files ({dir_info['size_mb']:.1f} MB)\n"

        # Python 코드 분석 (개선된 버전)
        if code_analysis.get("total_functions", 0) > 0:
            readme_content += f"""
## 🐍 Comprehensive Python Code Analysis

### 📈 Code Statistics
- **📁 Total Python Files:** {code_analysis.get('total_python_files', 0):,}
- **🔍 Files Analyzed:** {code_analysis.get('analyzed_files', 0):,}
- **📝 Lines of Code:** {code_analysis.get('total_lines_of_code', 0):,}
- **📊 Average Lines per File:** {code_analysis.get('avg_lines_per_file', 0)}
- **⚡ Functions:** {code_analysis.get('total_functions', 0):,}
- **🏗️ Classes:** {code_analysis.get('total_classes', 0):,}
- **📚 Documentation Coverage:**
  - Functions: {code_analysis.get('documented_functions', 0):,}/{code_analysis.get('total_functions', 0):,} ({code_analysis.get('function_documentation_rate', 0)}%)
  - Classes: {code_analysis.get('documented_classes', 0):,}/{code_analysis.get('total_classes', 0):,} ({code_analysis.get('class_documentation_rate', 0)}%)

### 🔍 Code Complexity Distribution
"""
            complexity = code_analysis.get("complexity_distribution", {})
            total_funcs = sum(complexity.values())
            complexity_icons = {"simple": "🟢", "medium": "🟡", "complex": "🔴", "very_complex": "⚫"}
            
            if total_funcs > 0:
                for level, count in complexity.items():
                    percentage = (count / total_funcs) * 100
                    icon = complexity_icons.get(level, "⚪")
                    label = level.replace("_", " ").title()
                    readme_content += f"- **{icon} {label}:** {count:,} functions ({percentage:.1f}%)\n"

            # Python 파일 분포
            readme_content += f"""
### 📁 Python File Distribution
"""
            py_dist = code_analysis.get("python_file_distribution", {})
            for directory, count in list(py_dist.items())[:8]:
                readme_content += f"- **{directory}:** {count} Python files\n"

            # 가장 많이 사용된 imports
            if code_analysis.get("top_imports"):
                readme_content += f"""
### 📦 Most Used Imports
"""
                for module, count in code_analysis.get("top_imports", [])[:10]:
                    readme_content += f"- **{module}:** used {count} times\n"

        readme_content += f"""
## 📖 Documentation Analysis

### 📝 Documentation Files
"""
        doc_files = doc_analysis.get("documentation_files", [])
        doc_icons = {
            "README.md": "📋", "README.rst": "📋", "README.txt": "📋",
            "CHANGELOG.md": "📅", "CONTRIBUTING.md": "🤝", "LICENSE": "⚖️",
            "CODE_OF_CONDUCT.md": "📜", "SECURITY.md": "🔒"
        }
        
        if doc_files:
            for doc_file in doc_files:
                icon = doc_icons.get(doc_file, "📄")
                readme_content += f"- ✅ {icon} {doc_file}\n"
        else:
            readme_content += "- ❌ No standard documentation files found\n"

        readme_content += f"""
- **📚 Documentation Directory:** {'✅ Present' if doc_analysis.get('has_docs_directory') else '❌ Not found'}
- **📄 Files in docs/:** {doc_analysis.get('docs_file_count', 0)}
- **📊 Documentation Score:** {doc_analysis.get('documentation_score', 0)}/10

## 🎯 Project Assessment

### 📏 Project Metrics
- **📐 Project Size:** {summary.get('project_size')}
- **🏗️ Structure Complexity:** {summary.get('structure_complexity')}
- **📖 Documentation Quality:** {summary.get('documentation_quality')}
- **💎 Code Quality:** {summary.get('code_quality')}
- **🔄 Activity Level:** {summary.get('activity_level')}

### 💡 Recommendations
"""

        # 개선된 추천사항
        recommendations = []
        
        if code_analysis.get("function_documentation_rate", 0) < 50:
            recommendations.append(f"📝 Improve function documentation (currently {code_analysis.get('function_documentation_rate', 0)}%)")
        
        if not doc_analysis.get("has_docs_directory"):
            recommendations.append("📚 Add a dedicated documentation directory")
        
        if "README.md" not in doc_analysis.get("documentation_files", []):
            recommendations.append("📋 Add a comprehensive README.md file")
        
        if not repo_info.get("license"):
            recommendations.append("⚖️ Add a license file")
        
        if file_stats.get("max_depth", 0) > 7:
            recommendations.append("🏗️ Consider simplifying deep directory structures")
        
        if code_analysis.get("avg_lines_per_file", 0) > 500:
            recommendations.append("✂️ Consider breaking down large files")
        
        if not recommendations:
            recommendations.append("🎉 Excellent! The repository follows best practices")
        
        for rec in recommendations:
            readme_content += f"- {rec}\n"

        readme_content += f"""

---

*This deep analysis was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*

## 🔧 Analysis Details

This comprehensive analysis examined:
- 🏗️ Complete directory tree structure up to {file_stats.get('max_depth', 0)} levels deep
- 💎 {code_analysis.get('analyzed_files', 0)} Python files for code quality metrics
- 📊 File distribution across {file_stats.get('total_directories', 0)} directories
- 🎯 Documentation completeness and quality

---

**⚡ Generated by Improved Repository Analyzer Tool v2.0**
"""

        return readme_content
    
    def save_readme(self, filename="ANALYSIS_README.md"):
        """README 파일로 저장"""
        content = self.generate_readme()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"개선된 README 파일 생성 완료: {filename}")
        return filename


# 사용 예시
def analyze_repository_improved(github_user, github_repo, github_branch="main"):
    """개선된 저장소 분석 및 README 생성"""
    
    # 1. 개선된 분석기로 저장소 분석
    analyzer = ImprovedRepositoryAnalyzer(github_user, github_repo, github_branch)
    analysis_results = analyzer.run_full_analysis()
    
    # 2. 개선된 README 생성
    readme_gen = ImprovedReadmeGenerator(analysis_results)
    readme_file = readme_gen.save_readme(f"{github_repo}_deep_analysis.md")
    
    # 3. 분석 결과를 JSON으로도 저장
    with open(f"{github_repo}_deep_analysis.json", 'w', encoding='utf-8') as f:
        # Counter와 set 객체를 JSON 직렬화 가능하게 변환
        def convert_for_json(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_safe_results = convert_for_json(analysis_results)
        json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
    
    # 4. 트리 구조 시각화 파일 생성
    tree_viz = analyzer.generate_tree_visualization(analysis_results["directory_tree"])
    with open(f"{github_repo}_tree.txt", 'w', encoding='utf-8') as f:
        f.write(f"Directory Structure for {github_user}/{github_repo}\n")
        f.write("=" * 50 + "\n\n")
        f.write(tree_viz)
    
    return readme_file, analysis_results


if __name__ == "__main__":
    print("\n=== 개선된 GitHub Repository Analyzer ===")
    print("실행 방법을 선택하세요:")
    print("1. 대화형 메뉴로 선택")
    print("2. 테스트 저장소 자동 분석")
    
    choice = input("\n선택 (1-2): ").strip()
    
    if choice == "1":
        # 원본 코드의 선택 함수 사용 (이 함수도 복사해와야 함)
        print("\n분석할 저장소를 입력하세요:")
        github_user = input("GitHub 사용자명/조직명: ").strip()
        github_repo = input("저장소 이름: ").strip()
        github_branch = input("브랜치 (기본값: main): ").strip() or "main"
        
        if github_user and github_repo:
            print(f"\n{github_user}/{github_repo} 분석 시작...")
            try:
                readme_file, results = analyze_repository_improved(github_user, github_repo, github_branch)
                print(f"✅ 분석 완료: {readme_file}")
                print(f"   - 총 파일: {results['file_statistics']['total_files']:,}")
                print(f"   - 최대 깊이: {results['file_statistics']['max_depth']}")
                print(f"   - 총 크기: {results['file_statistics']['total_size_mb']:.1f} MB")
            except Exception as e:
                print(f"❌ 분석 실패: {e}")
    
    elif choice == "2":
        # 기존 테스트 코드
        test_repos = [
            ("django", "django"),
            ("tensorflow", "tensorflow"),
            ("facebook", "react")
        ]
        
        for user, repo in test_repos[:1]:  # 첫 번째만 테스트
            print(f"\n분석 중: {user}/{repo}")
            try:
                readme_file, results = analyze_repository_improved(user, repo)
                print(f"✅ 분석 완료: {readme_file}")
                print(f"   - 총 파일: {results['file_statistics']['total_files']:,}")
                print(f"   - 최대 깊이: {results['file_statistics']['max_depth']}")
                print(f"   - 총 크기: {results['file_statistics']['total_size_mb']:.1f} MB")
            except Exception as e:
                print(f"❌ 분석 실패: {e}")