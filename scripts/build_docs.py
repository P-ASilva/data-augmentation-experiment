import os
import importlib.util
import jinja2
from pathlib import Path

def load_template(template_path):
    """Load Jinja2 template from file"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return jinja2.Template(f.read())

def run_report_generator(module_path):
    """Dynamically import and run report generator module"""
    spec = importlib.util.spec_from_file_location("report_gen", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()

def build_study_reports():
    """Build all study reports by executing their generators"""
    docs_dir = Path("docs")
    studies_dir = docs_dir / "studies"
    
    for study_folder in studies_dir.iterdir():
        print(study_folder)
        if not study_folder.is_dir():
            continue
            
        template_file = study_folder / "report.md.j2"
        generator_file = study_folder / "main.py"
        
        if not template_file.exists() or not generator_file.exists():
            continue
        
        print(f"Building report for {study_folder.name}...")
        
        # Run the generator to get template variables
        template_vars = run_report_generator(generator_file)
        
        # Render template with variables
        template = load_template(template_file)
        rendered_content = template.render(**template_vars)
        
        # Write final markdown file
        output_file = study_folder / "report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rendered_content)
        
        print(f"✓ Generated {output_file}")

if __name__ == "__main__":
    build_study_reports()