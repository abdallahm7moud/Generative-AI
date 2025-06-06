import re
from typing import Dict, Optional, List, Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool  # Use crewai.tools, not langchain.tools
from langchain_community.llms import Ollama

# --------- Input Schemas ---------

class CodeGenArgs(BaseModel):
    specification: str = Field(description="Description of what the code should do")


class CodeAnalyzeArgs(BaseModel):
    code_input: str = Field(description="Snippet ID or direct code to analyze")


# --------- CodeGeneratorTool ---------

class CodeGeneratorTool(BaseTool):
    name: str = "generate_code"
    description: str = "Generate code based on detailed specifications."
    args_schema: Type[CodeGenArgs] = CodeGenArgs
    _llm: Ollama = PrivateAttr()
    _code_snippets: Dict[str, str] = PrivateAttr()
    _snippet_counter: int = PrivateAttr()

    def __init__(self, code_snippets: Optional[Dict[str, str]] = None):
        super().__init__()
        self._llm = Ollama(
            model="ollama/codellama:34b",
            temperature=0.2,
            base_url="http://localhost:11434"
        )
        self._code_snippets = code_snippets if code_snippets is not None else {}
        self._snippet_counter = len(self._code_snippets)

    def _run(self, specification: str) -> str:
        """
        Generate code based on detailed specifications.
        """
        try:
            prompt = f"""
                Generate Python code based on the following specification.
                Include:
                1. Well-structured code that follows best practices
                2. Comprehensive comments explaining the logic
                3. Error handling where appropriate
                4. Examples of how to use the code

                SPECIFICATION:
                {specification}

                Return only valid code blocks (Python) formatted using triple backticks.
                """
            response = self._llm.invoke(prompt)

            # Extract code blocks using regex
            code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
            code = "\n\n# ----------\n\n".join(code_blocks) if code_blocks else response.strip()

            # Store and index the code
            self._snippet_counter += 1
            snippet_id = f"snippet_{self._snippet_counter}"
            self._code_snippets[snippet_id] = code

            return f"""
✅ Code generated successfully (ID: {snippet_id}):

{code}

➡️ To analyze this code, use: analyze_code with `{snippet_id}`
➡️ To refine this code, use: refine_code with `{snippet_id}` and your improvements.
"""
        except Exception as e:
            return f"❌ Error generating code: {str(e)}"


# --------- CodeAnalyzerTool ---------

class CodeAnalyzerTool(BaseTool):
    name: str = "code_analyzer"
    description: str = "Analyze code snippets and detect potential issues."
    args_schema: Type[CodeAnalyzeArgs] = CodeAnalyzeArgs
    _llm: Ollama = PrivateAttr()
    _code_snippets: Dict[str, str] = PrivateAttr()

    def __init__(self, code_snippets: Optional[Dict[str, str]] = None):
        super().__init__()
        self._code_snippets = code_snippets if code_snippets is not None else {}
        self._llm = Ollama(
            model="ollama/codellama:34b",
            temperature=0.2,
            base_url="http://localhost:11434"
        )

    def _run(self, code_input: str) -> str:
        """
        Analyze code and provide feedback.
        """
        try:
            code = self._code_snippets.get(code_input, code_input)

            prompt = f"""
                Analyze the following Python code. Provide feedback on:
                1. What the code does
                2. Bugs or logical issues
                3. Performance improvements
                4. Style and readability
                5. Security concerns

                CODE:
                ```python
                {code}
                ```
                Format your response in clear sections.
                """
            analysis = self._llm.invoke(prompt)
            return f"Code Analysis:\n\n{analysis}"
        except Exception as e:
            return f"Error analyzing code: {str(e)}"


# --------- Tool Factory ---------

def get_code_tools() -> List[BaseTool]:
    """
    Get a collection of code-related tools that share state.
    Returns:
        List of tool instances
    """
    shared_snippets = {}
    generator = CodeGeneratorTool(code_snippets=shared_snippets)
    analyzer = CodeAnalyzerTool(code_snippets=shared_snippets)
    return [generator, analyzer]
