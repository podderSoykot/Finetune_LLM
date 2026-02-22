import json
import re

def remove_comments_from_line(line):
    """Remove comments from a line, preserving strings."""
    if not line or not line.strip():
        return line
    
    # Remove trailing newline for processing
    has_newline = line.endswith('\n')
    line = line.rstrip('\n')
    
    stripped = line.lstrip()
    
    # Check if entire line is a comment (starts with # after whitespace)
    if stripped.startswith('#'):
        return None
    
    # Check if # is inside a string
    in_single_quote = False
    in_double_quote = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    
    while i < len(line):
        char = line[i]
        
        # Check for triple quotes
        if i + 2 < len(line):
            triple = line[i:i+3]
            if triple == '"""' and not in_single_quote and not in_triple_single:
                in_triple_double = not in_triple_double
                i += 3
                continue
            if triple == "'''" and not in_double_quote and not in_triple_double:
                in_triple_single = not in_triple_single
                i += 3
                continue
        
        # Check for single/double quotes (only if not in triple quotes)
        if not in_triple_single and not in_triple_double:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == '#' and not in_single_quote and not in_double_quote:
                # Found a comment outside of strings
                # Remove everything from # onwards, but keep the line if there's content before #
                before_comment = line[:i].rstrip()
                if before_comment:
                    return before_comment + ('\n' if has_newline else '')
                else:
                    return None
        
        i += 1
    
    return line + ('\n' if has_newline else '')

def process_notebook(notebook_path):
    """Remove all comments from code cells in a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    total_comments_removed = 0
    
    for cell in data['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            original_source = cell['source']
            
            # Handle both string and list formats
            if isinstance(original_source, str):
                lines = original_source.splitlines(keepends=True)
            else:
                lines = original_source
            
            new_lines = []
            for line in lines:
                processed = remove_comments_from_line(line)
                if processed is not None:
                    new_lines.append(processed)
                else:
                    total_comments_removed += 1
            
            # Reconstruct source in the same format as original
            if isinstance(original_source, str):
                new_source = ''.join(new_lines)
            else:
                new_source = new_lines
            
            if new_source != original_source:
                cell['source'] = new_source
                modified = True
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"Comments removed from {notebook_path}")
        print(f"Total comment lines removed: {total_comments_removed}")
    else:
        print(f"No comments found in {notebook_path}")

if __name__ == '__main__':
    notebook_path = 'finetune-llama.ipynb'
    process_notebook(notebook_path)
