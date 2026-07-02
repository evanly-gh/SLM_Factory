# agent/tools/__init__.py
from agent.tools.bash_tool import bash
from agent.tools.file_tools import read_file, edit_file
from agent.tools.web_search import web_search

COLD_START_TOOLS = [bash, read_file, edit_file, web_search]
