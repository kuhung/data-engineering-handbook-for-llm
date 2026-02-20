import yaml
import os

def generate_pdf_config():
    with open("mkdocs.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Replace arithmatex with mdx_math for server-side MathML rendering
    if "markdown_extensions" in config:
        new_extensions = []
        for ext in config["markdown_extensions"]:
            if isinstance(ext, str):
                if ext == "pymdownx.arithmatex":
                    continue
                new_extensions.append(ext)
            elif isinstance(ext, dict):
                if "pymdownx.arithmatex" in ext:
                    continue
                new_extensions.append(ext)
        
        # Add mdx_math
        new_extensions.append({
            "mdx_math": {
                "enable_dollar_delimiter": True
            }
        })
        config["markdown_extensions"] = new_extensions

    # 2. Add PDF-specific CSS
    if "extra_css" not in config:
        config["extra_css"] = []
    config["extra_css"].append("stylesheets/pdf_extra.css")

    # 3. Remove MathJax scripts (not needed for server-side rendering)
    if "extra_javascript" in config:
        config["extra_javascript"] = [
            js for js in config["extra_javascript"] 
            if "mathjax" not in js and "cdn.jsdelivr.net" not in js
        ]

    # 4. Save as mkdocs-pdf.yml
    with open("mkdocs-pdf.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print("Generated mkdocs-pdf.yml for PDF build.")

if __name__ == "__main__":
    generate_pdf_config()
