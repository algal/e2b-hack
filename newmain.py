import base64
from dotenv import load_dotenv
from anthropic import Anthropic
from typing import List, Tuple
from e2b_code_interpreter import CodeInterpreter, Result
from e2b_code_interpreter.models import Logs

# from e2b_hello_world.model import MODEL_NAME, SYSTEM_PROMPT, tools
# from e2b_hello_world.code_interpreter import code_interpret

# Load the .env file
load_dotenv()


MODEL_NAME = "claude-3-opus-20240229"

SYSTEM_PROMPT = """
## your job & context
You are the world's best storyteller and data analyst, you will always start with a single CSV and go from there:
- Your goal is to take in a CSV, process it and understand what domain we are working with then come up with a plan to process it and tell a story
- You will always come up with a plan, and ask me for me for confirmation before executing it, wait for my response and then start executing
- After you have done that, we will go about that plan step by step and execute every step, get my confirmation and continue with the next
- the python code runs in jupyter notebook.
- every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
- you can run any python code you want, everything is running in a secure sandbox environment.
"""

tools = [
    {
        "name": "execute_python",
        "description": "Execute python code in a Jupyter notebook cell and returns any result, stdout, stderr, display_data, and error.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The python code to execute in a single cell."
                }
            },
            "required": ["code"]
        }
    }
]

def code_interpret(code_interpreter: CodeInterpreter, code: str):
  "Runs `code` using `code_interpreter`"
  print(f"\n{'='*50}\n> Running following AI-generated code:\n{code}\n{'='*50}")
  exec = code_interpreter.notebook.exec_cell(
    code,
    # You can stream logs from the code interpreter
    # on_stderr=lambda stderr: print("\n[Code Interpreter stdout]", stderr),
    # on_stdout=lambda stdout: print("\n[Code Interpreter stderr]", stdout),
    #
    # You can also stream additional results like charts, images, etc.
    # on_result=...
  )

  if exec.error:
    print("[Code Interpreter error]", exec.error) # Runtime error
  else:
    return exec.results, exec.logs

client = Anthropic()

class BlockStep:
  def __init__(self):
    self.input_text = None
    self.output_code = None
    self.output_image_b64 = None


def steps_to_messages(steps:List[BlockStep]) -> List[dict]:
  past_messages = []
  for step in steps:
     d = {"role":"user","content": step.input_text}
     dr = {"role":"assistant","content": step.output_code}
     past_messages += [d]
     past_messages += [dr]
  return past_messages


def make_next_BlockStep(input_text:str, code_interpreter_ctx:CodeInterpreter, history:List[BlockStep] = []) -> BlockStep:
    """
    Sends a message to the model.
    Prints model response until model wants to execute code.
    Feeds that code to code_interpreter to run
    Returns code_interpeter's retvalue, a Tuple(List[e2b_code_interpreter.Result],Logs]
    """
    user_message = input_text
    code_interpreter = code_interpreter_ctx
  
    print(f"\n{'='*50}\nUser Message: {input_text}\n{'='*50}")

    ret_blockstep = BlockStep()
    ret_blockstep.input_text = input_text
    
    older_messages = steps_to_messages(history)
    preceding_messages = older_messages + [{"role": "user", "content": user_message}]

    message = client.messages.create(
        model=MODEL_NAME,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        messages=preceding_messages,
        tools=tools,
    )

    print(f"\n{'='*50}\nModel response: {message.content}\n{'='*50}")

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        # determine which tool Claude wants to use and what inputs it wants to feed it
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\n{'='*50}\nUsing tool: {tool_name}\n{'='*50}")

        if tool_name == "execute_python":
            # look for code reutnred by the model
            ret_blockstep.output_code = tool_input["code"]
            retval = code_interpret(code_interpreter, tool_input["code"])
            # look for first generated image in code exec output
            (results, logs) = retval
            first_image_result = next(b for b in results if b.png)
            if first_image_result:
              ret_blockstep.output_image_b64 = first_image_result.png
              # assert: we have a complete BlockStep
        else:
            print("Claude wants an unknown tool. logic error.")
            return ret_blockstep
    else:
        print(f"Claude did not stop replying bc of tool use")
        return ret_blockstep
    return ret_blockstep

def chat(code_interpreter: CodeInterpreter, user_message: str) -> Tuple[List[Result], Logs]:
    """
    Sends a message to the model.
    Prints model response until model wants to execute code.
    Feeds that code to code_interpreter to run
    Returns code_interpeter's retvalue, a Tuple(List[e2b_code_interpreter.Result],Logs]
    """
    print(f"\n{'='*50}\nUser Message: {user_message}\n{'='*50}")

    message = client.messages.create(
        model=MODEL_NAME,
        system=SYSTEM_PROMPT,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    print(f"\n{'='*50}\nModel response: {message.content}\n{'='*50}")

    # Assert: any generated code is now available to us

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\n{'='*50}\nUsing tool: {tool_name}\n{'='*50}")

        if tool_name == "execute_python":
            return code_interpret(code_interpreter, tool_input["code"])

    return None, None

class Session:
  def __init__(self):
    self.ci = CodeInterpreter()
    self.history = []

  def add_next_input(self,input_str):
    retries = 3
    while retries > 0:
      bs = make_next_BlockStep(input_str, self.ci, self.history)
      if bs.output_image_b64 is not None:
        break
      retries -= 1
      print(f"BlockStep had no image. Retrying... retries left: {retries}")

    if bs.output_image_b64 is None:
      raise ValueError("Max retries reached with no image")
    # assert: success
    self.history.append(bs)
    return bs

def main():
  sess = Session()
  #next_input_text = "Estimate a distribution of height of men without using external data sources. Also print the median value."

  next_input_text = None
  i = 0
  while i < 3:
    if next_input_text is None:
      next_input_text = input("Enter the next advice for more analysis: ")
    bs = sess.add_next_input(next_input_text)
    next_input_text = None
    # print file
    fname = f"chart{i}.png"
    with open(fname,"wb") as f:
      f.write(base64.b64decode(bs.output_image_b64))
      print(f"saved {fname}")
    i += 1

if __name__ == '__main__':
    main()
    
