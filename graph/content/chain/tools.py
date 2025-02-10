import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def write_markdown_file(content, filename):
  """Writes the given content as a markdown file to the local directory.

  Args:
    content: The string content to write to the file.
    filename: The filename to save the file as.
  """
  main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  dir = os.path.join(main_dir, "result")
  if not os.path.exists(dir): os.makedirs(dir)

  file_path = os.path.join(dir, f"{filename}.md")
  with open(file_path, "w") as f:
    f.write(content)