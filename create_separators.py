def create_section_separator(title, char="*", width=120):
    """Create a decorative section separator with a title."""
    side_padding = (width - len(title) - 2) // 2
    separator = f"{char * width}\n"
    title_line = f"{char * side_padding} {title} {char * (width - side_padding - len(title) - 2)}\n"
    return f'"""\n{separator}{title_line}{separator}"""'

# Example usage:
print(create_section_separator("TREE LOADING AND PROCESSSING"))
print("\n# Your code here\n")
print(create_section_separator("FASTA LOADING"))
print("\n# Your code here\n")
print(create_section_separator("MAFFT"))
print("\n# Your code here\n")
print(create_section_separator("SCORING"))
print("\n# Your code here\n")
print(create_section_separator("RECURSIVE ALIGNMENT"))
print("\n# Your code here\n")
print(create_section_separator("UTILS"))
print("\n# Your code here\n")
print(create_section_separator("AA TO 3Di Converter"))
print("\n# Your code here\n")
print(create_section_separator("MAIN WORKFLOW"))

