import os
import io

def generate_tree(startpath, ignore_dirs=None, ignore_files=None):
    if ignore_dirs is None:
        ignore_dirs = set(['.git', '__pycache__', '.ipynb_checkpoints', '.venv'])
    if ignore_files is None:
        ignore_files = set(['.gitignore', '.gitattributes'])

    tree = []
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        tree.append(f'{indent}{os.path.basename(root)}/')
        subindent = '│   ' * level + '├── '
        for f in sorted(files):
            if f not in ignore_files:
                tree.append(f'{subindent}{f}')
    return '\n'.join(tree)

def update_readme(readme_path, tree):
    try:
        with io.open(readme_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        content = "# TikTok EDA Project\n\n## Overview\nThis project involves an extensive exploratory data analysis (EDA) and machine learning model development on a TikTok dataset. The goal is to uncover insights from the data and build predictive models.\n\n"

    # Find the start and end of the directory structure
    start = content.find("## Project Structure")
    if start == -1:
        # If "Project Structure" section doesn't exist, add it
        updated_content = content + "\n## Project Structure\n\nThe project is organized into the following directories and scripts:\n\n```\n" + tree + "\n```\n"
    else:
        # Find the end of the Project Structure section
        next_section = content.find("##", start + 1)
        if next_section == -1:
            next_section = len(content)
        
        # Replace only the Project Structure section
        updated_content = content[:start] + "## Project Structure\n\nThe project is organized into the following directories and scripts:\n\n```\n" + tree + "\n```\n\n" + content[next_section:]

    with io.open(readme_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    print("README.md updated successfully.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    tree = generate_tree(project_root)
    readme_path = os.path.join(project_root, 'README.md')
    update_readme(readme_path, tree)
