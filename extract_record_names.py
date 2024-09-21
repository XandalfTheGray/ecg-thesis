import os
import argparse
import io

def get_directory_structure(directory):
    """
    Returns a string representation of the directory structure.
    """
    output = []
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
        output.append(f'{indent}{os.path.basename(root)}/')
        subindent = '│   ' * level + '├── '
        for file in files:
            output.append(f'{subindent}{file}')
    return '\n'.join(output)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract directory structure and file names.")
    parser.add_argument("directory", help="Path to the directory to analyze.")
    parser.add_argument("--output", default="directory_structure.txt", help="Output file name (default: directory_structure.txt)")

    # Parse arguments
    args = parser.parse_args()

    # Get directory structure
    structure = get_directory_structure(args.directory)

    # Write to file using UTF-8 encoding
    with io.open(args.output, "w", encoding="utf-8") as f:
        f.write(structure)

    print(f"Directory structure has been written to {args.output}")

if __name__ == "__main__":
    main()