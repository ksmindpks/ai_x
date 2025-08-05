import os
import requests
import ast
from datetime import datetime
from collections import defaultdict, Counter
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RepositoryAnalyzer:
    def __init__(self, github_user, github_repo, github_branch="main"):
        self.github_user = github_user
        self.github_repo = github_repo
        self.github_branch = github_branch
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        
        # 분석 결과 저장
        self.analysis_results = {
            "repository_info": {},
            "file_statistics": {},
            "directory_structure": {},
            "code_analysis": {},
            "documentation_analysis": {},
            "summary": {}
        }
    
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
                "license": repo_data.get("license", {}).get("name") if repo_data.get("license") else None
            }
            
            logger.info(f"저장소 정보 수집 완료: {repo_data.get('full_name')}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"저장소 정보 수집 실패: {e}")
    
    def analyze_file_structure(self):
        """파일 구조 및 통계 분석"""
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/git/trees/{self.github_branch}?recursive=1"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            tree = response.json().get("tree", [])
            
            # 파일 확장자별 통계
            file_extensions = Counter()
            directory_files = defaultdict(int)
            total_files = 0
            total_dirs = 0
            
            for item in tree:
                if item["type"] == "blob":  # 파일
                    total_files += 1
                    path = item["path"]
                    
                    # 확장자 추출
                    if "." in path:
                        ext = path.split(".")[-1].lower()
                        file_extensions[ext] += 1
                    
                    # 디렉토리별 파일 수
                    if "/" in path:
                        main_dir = path.split("/")[0]
                        directory_files[main_dir] += 1
                    else:
                        directory_files["root"] += 1
                        
                elif item["type"] == "tree":  # 디렉토리
                    total_dirs += 1
            
            self.analysis_results["file_statistics"] = {
                "total_files": total_files,
                "total_directories": total_dirs,
                "file_extensions": dict(file_extensions.most_common(10)),
                "directory_files": dict(sorted(directory_files.items(), key=lambda x: x[1], reverse=True)[:15])
            }
            
            logger.info(f"파일 구조 분석 완료: {total_files}개 파일, {total_dirs}개 디렉토리")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"파일 구조 분석 실패: {e}")
    
    def analyze_python_code(self, max_files=50):
        """Python 코드 분석 (함수, 클래스, docstring 등)"""
        if self.analysis_results["file_statistics"].get("file_extensions", {}).get("py", 0) == 0:
            logger.info("Python 파일이 없어 코드 분석을 건너뜁니다.")
            return
        
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/git/trees/{self.github_branch}?recursive=1"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            tree = response.json().get("tree", [])
            
            python_files = [f for f in tree if f["path"].endswith(".py")][:max_files]
            
            total_functions = 0
            total_classes = 0
            total_docstrings = 0
            documented_functions = 0
            documented_classes = 0
            
            complexity_stats = {
                "simple": 0,    # < 10 lines
                "medium": 0,    # 10-50 lines  
                "complex": 0    # > 50 lines
            }
            
            for file_info in python_files:
                raw_url = f"https://raw.githubusercontent.com/{self.github_user}/{self.github_repo}/{self.github_branch}/{file_info['path']}"
                
                try:
                    file_response = requests.get(raw_url)
                    if file_response.status_code == 200:
                        code = file_response.text
                        
                        # AST 파싱
                        try:
                            tree_ast = ast.parse(code)
                            
                            for node in ast.walk(tree_ast):
                                if isinstance(node, ast.FunctionDef):
                                    total_functions += 1
                                    
                                    # docstring 확인
                                    if ast.get_docstring(node):
                                        documented_functions += 1
                                        total_docstrings += 1
                                    
                                    # 복잡도 분석 (라인 수 기준)
                                    lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                                    if lines < 10:
                                        complexity_stats["simple"] += 1
                                    elif lines <= 50:
                                        complexity_stats["medium"] += 1
                                    else:
                                        complexity_stats["complex"] += 1
                                
                                elif isinstance(node, ast.ClassDef):
                                    total_classes += 1
                                    
                                    # docstring 확인
                                    if ast.get_docstring(node):
                                        documented_classes += 1
                                        total_docstrings += 1
                        
                        except SyntaxError:
                            # 구문 오류가 있는 파일은 건너뛰기
                            continue
                
                except requests.exceptions.RequestException:
                    continue
            
            # 문서화 비율 계산
            func_doc_rate = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            class_doc_rate = (documented_classes / total_classes * 100) if total_classes > 0 else 0
            
            self.analysis_results["code_analysis"] = {
                "analyzed_files": len(python_files),
                "total_functions": total_functions,
                "total_classes": total_classes,
                "total_docstrings": total_docstrings,
                "documented_functions": documented_functions,
                "documented_classes": documented_classes,
                "function_documentation_rate": round(func_doc_rate, 1),
                "class_documentation_rate": round(class_doc_rate, 1),
                "complexity_distribution": complexity_stats
            }
            
            logger.info(f"Python 코드 분석 완료: {total_functions}개 함수, {total_classes}개 클래스")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Python 코드 분석 실패: {e}")
    
    def analyze_documentation(self):
        """문서화 파일 분석 (README, docs 등)"""
        doc_files = ["README.md", "README.rst", "README.txt", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"]
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
        
        try:
            response = requests.get(docs_url, headers=self.headers)
            if response.status_code == 200:
                has_docs_dir = True
        except:
            pass
        
        self.analysis_results["documentation_analysis"] = {
            "documentation_files": found_docs,
            "has_docs_directory": has_docs_dir,
            "documentation_score": len(found_docs) + (2 if has_docs_dir else 0)
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
        else:
            project_size = "Large"
        
        # 문서화 품질 평가
        doc_score = doc_analysis.get("documentation_score", 0)
        if doc_score >= 5:
            doc_quality = "Excellent"
        elif doc_score >= 3:
            doc_quality = "Good"
        elif doc_score >= 1:
            doc_quality = "Basic"
        else:
            doc_quality = "Poor"
        
        # 코드 품질 평가 (Python 프로젝트의 경우)
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
        
        self.analysis_results["summary"] = {
            "project_size": project_size,
            "documentation_quality": doc_quality,
            "code_quality": code_quality,
            "main_language": repo_info.get("language", "Unknown"),
            "activity_level": "Active" if repo_info.get("updated_at") else "Unknown",
            "community_engagement": f"{repo_info.get('stars', 0)} stars, {repo_info.get('forks', 0)} forks"
        }
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        logger.info("저장소 분석 시작...")
        
        self.get_repository_info()
        self.analyze_file_structure()
        self.analyze_python_code()
        self.analyze_documentation()
        self.generate_summary()
        
        logger.info("저장소 분석 완료!")
        return self.analysis_results

class ReadmeGenerator:
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
        
        # 코드 품질 배지 (Python 프로젝트인 경우)
        if code_analysis.get("total_functions", 0) > 0:
            code_quality = summary.get("code_quality", "N/A")
            color = {"Excellent": "brightgreen", "Good": "green", "Fair": "yellow", "Needs Improvement": "red"}.get(code_quality, "lightgrey")
            badges.append(f"![Code Quality](https://img.shields.io/badge/code%20quality-{code_quality.replace(' ', '%20')}-{color}?style=flat-square)")
        
        return " ".join(badges)
    
    def generate_readme(self):
        """분석 결과를 바탕으로 README.md 생성"""
        repo_info = self.analysis["repository_info"]
        file_stats = self.analysis["file_statistics"]
        code_analysis = self.analysis["code_analysis"]
        doc_analysis = self.analysis["documentation_analysis"]
        summary = self.analysis["summary"]
        
        # 배지 생성
        badges = self.generate_badges()
        
        readme_content = f"""# 📊 {repo_info.get('name', 'Repository')} Analysis Report

{badges}

## 🏠 Repository Overview

**🔗 Repository:** [{repo_info.get('full_name')}]({repo_info.get('url')})  
**📝 Description:** {repo_info.get('description', 'No description provided')}  
**💻 Primary Language:** {repo_info.get('language', 'Unknown')}  
**⚖️ License:** {repo_info.get('license', 'Not specified')}  

### ⭐ Community Stats
- **🌟 Stars:** {repo_info.get('stars', 0):,}
- **🍴 Forks:** {repo_info.get('forks', 0):,}
- **🐛 Open Issues:** {repo_info.get('issues', 0):,}
- **📦 Repository Size:** {repo_info.get('size', 0):,} KB

### 📅 Timeline
- **🎉 Created:** {repo_info.get('created_at', 'Unknown')[:10]}
- **🔄 Last Updated:** {repo_info.get('updated_at', 'Unknown')[:10]}

## 📁 File Structure Analysis

### 📊 File Statistics
- **📄 Total Files:** {file_stats.get('total_files', 0):,}
- **📂 Total Directories:** {file_stats.get('total_directories', 0):,}

### 🗂️ File Types Distribution
"""

        # 파일 확장자별 분포
        extensions = file_stats.get("file_extensions", {})
        for ext, count in extensions.items():
            percentage = (count / file_stats.get('total_files', 1)) * 100
            # 파일 타입별 아이콘
            icon = {"py": "🐍", "js": "📄", "html": "🌐", "css": "🎨", "md": "📝", "json": "📋", "txt": "📄", "yml": "⚙️", "yaml": "⚙️"}.get(ext, "📄")
            readme_content += f"- **{icon} {ext.upper()}:** {count:,} files ({percentage:.1f}%)\n"

        readme_content += f"""
### 📂 Directory Structure
"""

        # 디렉토리별 파일 분포
        directories = file_stats.get("directory_files", {})
        for dir_name, count in directories.items():
            # 디렉토리별 아이콘
            icon = {"src": "📁", "lib": "📚", "docs": "📖", "test": "🧪", "tests": "🧪", "examples": "💡", "tools": "🔧", "scripts": "📜", "assets": "🎯", "static": "🗂️"}.get(dir_name.lower(), "📁")
            readme_content += f"- **{icon} {dir_name}:** {count:,} files\n"

        # Python 코드 분석 (있는 경우)
        if code_analysis.get("total_functions", 0) > 0:
            readme_content += f"""
## 🐍 Python Code Analysis

### 📈 Code Statistics
- **⚡ Functions:** {code_analysis.get('total_functions', 0):,}
- **🏗️ Classes:** {code_analysis.get('total_classes', 0):,}
- **📚 Documented Functions:** {code_analysis.get('documented_functions', 0):,} ({code_analysis.get('function_documentation_rate', 0)}%)
- **📖 Documented Classes:** {code_analysis.get('documented_classes', 0):,} ({code_analysis.get('class_documentation_rate', 0)}%)

### 🔍 Code Complexity Distribution
"""
            complexity = code_analysis.get("complexity_distribution", {})
            total_funcs = sum(complexity.values())
            complexity_icons = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}
            
            if total_funcs > 0:
                for level, count in complexity.items():
                    percentage = (count / total_funcs) * 100
                    icon = complexity_icons.get(level, "⚪")
                    readme_content += f"- **{icon} {level.title()}:** {count:,} functions ({percentage:.1f}%)\n"

        readme_content += f"""
## 📖 Documentation Analysis

### 📝 Documentation Files Found
"""
        doc_files = doc_analysis.get("documentation_files", [])
        doc_icons = {
            "README.md": "📋", "README.rst": "📋", "README.txt": "📋",
            "CHANGELOG.md": "📅", "CONTRIBUTING.md": "🤝", "LICENSE": "⚖️"
        }
        
        if doc_files:
            for doc_file in doc_files:
                icon = doc_icons.get(doc_file, "📄")
                readme_content += f"- ✅ {icon} {doc_file}\n"
        else:
            readme_content += "- ❌ No standard documentation files found\n"

        readme_content += f"""
- **📚 Documentation Directory:** {'✅ Present' if doc_analysis.get('has_docs_directory') else '❌ Not found'}
- **📊 Documentation Score:** {doc_analysis.get('documentation_score', 0)}/10

## 🎯 Project Assessment

### 📏 Project Metrics
- **📐 Project Size:** {summary.get('project_size')}
- **📖 Documentation Quality:** {summary.get('documentation_quality')}
- **💎 Code Quality:** {summary.get('code_quality')}
- **🔄 Activity Level:** {summary.get('activity_level')}

### 💡 Recommendations

"""

        # 추천사항 생성
        recommendations = []
        
        if code_analysis.get("function_documentation_rate", 0) < 50:
            recommendations.append("📝 Consider improving function documentation coverage")
        
        if not doc_analysis.get("has_docs_directory"):
            recommendations.append("📚 Add a dedicated documentation directory")
        
        if "README.md" not in doc_analysis.get("documentation_files", []):
            recommendations.append("📋 Add a comprehensive README.md file")
        
        if not repo_info.get("license"):
            recommendations.append("⚖️ Consider adding a license file")
        
        if not recommendations:
            recommendations.append("🎉 Great job! The repository follows good practices")
        
        for rec in recommendations:
            readme_content += f"- {rec}\n"

        readme_content += f"""

---

*This analysis was generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*

## 🔧 How to Use This Analysis

This analysis provides insights into:
- 🏗️ Repository structure and organization
- 💎 Code quality and documentation coverage
- 📊 Community engagement metrics
- 🎯 Areas for improvement

Use these insights to:
- 📚 Improve project documentation
- 💎 Enhance code quality
- 🗂️ Better organize your repository
- 🤝 Increase community engagement

---

**⚡ Generated by Repository Analyzer Tool**
"""

        return readme_content
    
    def save_readme(self, filename="ANALYSIS_README.md"):
        """README 파일로 저장"""
        content = self.generate_readme()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"README 파일 생성 완료: {filename}")
        return filename

# 실행 예시 및 저장소 선택 도우미
def get_popular_repositories():
    """인기 있는 Python 저장소 목록 반환"""
    popular_repos = {
        "web_frameworks": [
            ("django", "django", "Django 웹 프레임워크"),
            ("pallets", "flask", "Flask 마이크로 웹 프레임워크"), 
            ("tiangolo", "fastapi", "FastAPI 현대적 웹 API 프레임워크")
        ],
        "data_science": [
            ("pandas-dev", "pandas", "데이터 분석 라이브러리"),
            ("numpy", "numpy", "과학 컴퓨팅 기본 패키지"),
            ("matplotlib", "matplotlib", "데이터 시각화 라이브러리")
        ],
        "machine_learning": [
            ("scikit-learn", "scikit-learn", "머신러닝 라이브러리"),
            ("tensorflow", "tensorflow", "구글의 머신러닝 플랫폼"),
            ("pytorch", "pytorch", "페이스북의 딥러닝 프레임워크")
        ],
        "tools": [
            ("python", "cpython", "Python 인터프리터"),
            ("psf", "requests", "HTTP 라이브러리"),
            ("pytest-dev", "pytest", "테스팅 프레임워크")
        ],
        "automation": [
            ("ansible", "ansible", "자동화 플랫폼"),
            ("scrapy", "scrapy", "웹 크롤링 프레임워크"),
            ("celery", "celery", "분산 작업 큐")
        ]
    }
    return popular_repos

def select_repository_interactive():
    """대화형 저장소 선택"""
    print("\n=== GitHub Repository Analyzer ===")
    print("분석할 저장소를 선택하세요:")
    print("1. 직접 입력")
    print("2. 인기 저장소에서 선택")
    
    choice = input("\n선택 (1-2): ").strip()
    
    if choice == "1":
        github_user = input("GitHub 사용자명/조직명: ").strip()
        github_repo = input("저장소 이름: ").strip()
        github_branch = input("브랜치 (기본값: main): ").strip() or "main"
        return github_user, github_repo, github_branch
    
    elif choice == "2":
        popular_repos = get_popular_repositories()
        
        print("\n카테고리를 선택하세요:")
        categories = list(popular_repos.keys())
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category.replace('_', ' ').title()}")
        
        cat_choice = input(f"\n카테고리 선택 (1-{len(categories)}): ").strip()
        
        try:
            cat_index = int(cat_choice) - 1
            selected_category = categories[cat_index]
            repos = popular_repos[selected_category]
            
            print(f"\n{selected_category.replace('_', ' ').title()} 저장소:")
            for i, (user, repo, desc) in enumerate(repos, 1):
                print(f"{i}. {user}/{repo} - {desc}")
            
            repo_choice = input(f"\n저장소 선택 (1-{len(repos)}): ").strip()
            repo_index = int(repo_choice) - 1
            
            github_user, github_repo, _ = repos[repo_index]
            github_branch = input("브랜치 (기본값: main): ").strip() or "main"
            
            return github_user, github_repo, github_branch
            
        except (ValueError, IndexError):
            print("잘못된 선택입니다.")
            return None, None, None
    
    else:
        print("잘못된 선택입니다.")
        return None, None, None

def analyze_multiple_repositories(repo_list):
    """여러 저장소 일괄 분석"""
    results = {}
    
    for i, (user, repo, branch) in enumerate(repo_list, 1):
        print(f"\n[{i}/{len(repo_list)}] 분석 중: {user}/{repo}")
        
        try:
            analyzer = RepositoryAnalyzer(user, repo, branch)
            analysis_results = analyzer.run_full_analysis()
            
            readme_gen = ReadmeGenerator(analysis_results)
            readme_file = readme_gen.save_readme(f"{repo}_analysis.md")
            
            results[f"{user}/{repo}"] = {
                "analysis": analysis_results,
                "readme_file": readme_file
            }
            
            # 분석 결과를 JSON으로도 저장
            with open(f"{repo}_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            print(f"완료: {readme_file}")
            
        except Exception as e:
            logger.error(f"{user}/{repo} 분석 실패: {e}")
            results[f"{user}/{repo}"] = {"error": str(e)}
    
    return results

# 사용 예시
def analyze_repository(github_user, github_repo, github_branch="main"):
    """저장소 분석 및 README 생성"""
    
    # 1. 저장소 분석
    analyzer = RepositoryAnalyzer(github_user, github_repo, github_branch)
    analysis_results = analyzer.run_full_analysis()
    
    # 2. README 생성
    readme_gen = ReadmeGenerator(analysis_results)
    readme_file = readme_gen.save_readme(f"{github_repo}_analysis.md")
    
    # 3. 분석 결과를 JSON으로도 저장
    with open(f"{github_repo}_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    return readme_file, analysis_results

if __name__ == "__main__":
    # 대화형 실행
    github_user, github_repo, github_branch = select_repository_interactive()
    
    if github_user and github_repo:
        print(f"\n{github_user}/{github_repo} 분석 시작...")
        readme_file, results = analyze_repository(github_user, github_repo, github_branch)
        print(f"분석 완료! README 파일: {readme_file}")
    else:
        print("분석이 취소되었습니다.")
    
    # 또는 여러 저장소 일괄 분석 예시
    # batch_repos = [
    #     ("django", "django", "main"),
    #     ("pallets", "flask", "main"), 
    #     ("tiangolo", "fastapi", "master")
    # ]
    # analyze_multiple_repositories(batch_repos)