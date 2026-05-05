# import json
# import re
# from prompt import system_prompt
# from reactxen.utils.model_inference import watsonx_llm


# def get_llm_answer_from_json(data: dict, model_id) -> str:
#     """
#     Given a parsed JSON dict with keys 'task', 'trajectory', and 'final_answer',
#     formats the content and returns the LLM's response.
#     """
#     try:
#         trajectory = data.get("trajectory", [])
#         question = data.get("text", "[No question provided]")
#         if len(trajectory) > 0:
#             final_answer = trajectory[-1].get('final_answer', "[No final answer provided]")
#         else:
#             final_answer = "[No final answer provided]"

#         import time
#         print(f"final_answer: {final_answer}", flush=True)
#         # time.sleep(10)

#         formatted_steps = [f"Question: {question}"]
#         for idx, step in enumerate(trajectory, 1):
#             thought = step.get("task_description", "[No thought]")
#             action = step.get("agent_name", "[No action]")
#             observation = step.get("response", "[No observation]")

#             step_text = (
#                 f"Thought {idx}: {thought}\n"
#                 f"Action {idx}: {action}\n"
#                 f"Observation {idx}: {observation}\n"
#             )
#             formatted_steps.append(step_text)

#         formatted_steps.append(f"Answer: {final_answer}")

#         # Combine all steps into a single formatted prompt
#         final_prompt_string = "\n" + "-" * 40 + "\n".join(formatted_steps)
#         prompt = system_prompt.format(trace=final_prompt_string)

#         # Call the model inference
#         # ans = watsonx_llm(prompt=prompt, model_id=16)
#         ans = watsonx_llm(prompt=prompt, model_id=model_id)
#         return ans

#     except Exception as e:
#         return f"Error while processing input data: {e}"

# def extract_json_from_response(response_text: str) -> dict:
#     """
#     Extract and parse a JSON object from LLM-generated response text,
#     even if it's wrapped in text or markdown formatting.
#     """
#     # Try to find a JSON block inside markdown-style code fences
#     match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
#     if match:
#         json_str = match.group(1)
#     else:
#         # Fallback: find the first {...} block in the response
#         match = re.search(r"(\{.*\})", response_text, re.DOTALL)
#         if match:
#             json_str = match.group(1)
#         else:
#             raise ValueError("No valid JSON found in the response text.")

#     try:
#         return json.loads(json_str)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"JSON decoding failed: {e}")



import json
import re
from prompt import system_prompt
from reactxen.utils.model_inference import watsonx_llm

def get_llm_answer_from_json(data: dict, model_id):
    """
    Formats one trajectory JSON dict into a prompt and calls LLM.
    Returns: whatever watsonx_llm returns (likely dict).
    """
    try:
        # ---- Basic shape debug ----
        print("[get_llm_answer_from_json] data type:", type(data), flush=True)
        if isinstance(data, dict):
            print("[get_llm_answer_from_json] top-level keys:", list(data.keys())[:50], flush=True)

        trajectory = data.get("trajectory", [])
        question = data.get("text", "[No question provided]")

        print("[get_llm_answer_from_json] question (head):", str(question)[:200], flush=True)
        print("[get_llm_answer_from_json] trajectory type:", type(trajectory), "len:", (len(trajectory) if isinstance(trajectory, list) else "N/A"), flush=True)

        # NOTE: your sample file has top-level "final_answer"
        # while your older logic tried to read final_answer from the last trajectory step.
        final_answer = data.get("final_answer", "[No final answer provided]")
        print("[get_llm_answer_from_json] final_answer (head):", str(final_answer)[:200], flush=True)

        # ---- Build prompt ----
        formatted_steps = [f"Question: {question}"]
        if isinstance(trajectory, list):
            for idx, step in enumerate(trajectory, 1):
                if not isinstance(step, dict):
                    print(f"[get_llm_answer_from_json] step {idx} is not dict: {type(step)}", flush=True)
                    continue

                thought = step.get("task_description", "[No thought]")
                action = step.get("agent_name", "[No action]")
                observation = step.get("response", "[No observation]")

                step_text = (
                    f"Thought {idx}: {thought}\n"
                    f"Action {idx}: {action}\n"
                    f"Observation {idx}: {observation}\n"
                )
                formatted_steps.append(step_text)
        else:
            print("[get_llm_answer_from_json] trajectory is not a list. Using empty steps.", flush=True)

        formatted_steps.append(f"Answer: {final_answer}")

        final_prompt_string = "\n" + "-" * 40 + "\n".join(formatted_steps)
        prompt = system_prompt.format(trace=final_prompt_string)

        print("[get_llm_answer_from_json] prompt chars:", len(prompt), flush=True)
        print("[get_llm_answer_from_json] prompt head:\n", prompt[:400], flush=True)
        print("[get_llm_answer_from_json] prompt tail:\n", prompt[-400:], flush=True)

        # ---- Call LLM ----
        ans = watsonx_llm(prompt=prompt, model_id=model_id)

        # ---- LLM return debug ----
        print("[get_llm_answer_from_json] watsonx_llm return type:", type(ans), flush=True)
        if isinstance(ans, dict):
            print("[get_llm_answer_from_json] watsonx_llm keys:", list(ans.keys())[:50], flush=True)
            gt = ans.get("generated_text", None)
            print("[get_llm_answer_from_json] generated_text exists:", gt is not None, flush=True)
            if gt is not None:
                print("[get_llm_answer_from_json] generated_text head:\n", str(gt)[:600], flush=True)
                print("[get_llm_answer_from_json] generated_text tail:\n", str(gt)[-600:], flush=True)
        else:
            print("[get_llm_answer_from_json] watsonx_llm raw (head):", str(ans)[:600], flush=True)

        return ans

    except Exception as e:
        # IMPORTANT: do NOT return string silently; re-raise or return structured error
        print("[get_llm_answer_from_json] ERROR:", repr(e), flush=True)
        raise

def extract_json_from_response(response_text: str) -> dict:
    """
    Extract and parse a JSON object from LLM-generated response text,
    even if it's wrapped in text or markdown formatting.
    """
    print("[extract_json_from_response] response_text type:", type(response_text), flush=True)
    if response_text is None:
        raise ValueError("response_text is None")

    text = str(response_text)
    print("[extract_json_from_response] chars:", len(text), flush=True)
    print("[extract_json_from_response] head:\n", text[:800], flush=True)
    print("[extract_json_from_response] tail:\n", text[-800:], flush=True)

    # 1) code fence ```json ... ```
    fence_pat = r"```json\s*(\{.*?\})\s*```"
    match = re.search(fence_pat, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        print("[extract_json_from_response] matched code-fence JSON. len:", len(json_str), flush=True)
    else:
        print("[extract_json_from_response] no code-fence JSON match", flush=True)

        # 2) naive { ... } (non-greedy) first try
        # NOTE: make it NON-greedy to reduce over-capture
        naive_pat = r"(\{.*?\})"
        match = re.search(naive_pat, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            print("[extract_json_from_response] matched naive first {...} (non-greedy). len:", len(json_str), flush=True)
        else:
            print("[extract_json_from_response] no naive {...} match", flush=True)

            # 3) balanced brace extraction fallback
            start = text.find("{")
            if start == -1:
                raise ValueError("No valid JSON found in the response text. (No '{' at all)")

            depth = 0
            end = None
            for i in range(start, len(text)):
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end is None:
                raise ValueError("Found '{' but could not find balanced closing '}'")

            json_str = text[start:end]
            print("[extract_json_from_response] matched balanced {...}. len:", len(json_str), flush=True)

    # show the extracted candidate
    print("[extract_json_from_response] json_str head:\n", json_str[:800], flush=True)
    print("[extract_json_from_response] json_str tail:\n", json_str[-800:], flush=True)

    try:
        obj = json.loads(json_str)
        print("[extract_json_from_response] json.loads SUCCESS. type:", type(obj), flush=True)
        if isinstance(obj, dict):
            print("[extract_json_from_response] keys:", list(obj.keys())[:50], flush=True)
        return obj
    except json.JSONDecodeError as e:
        # include context around error position
        pos = e.pos
        lo = max(0, pos - 120)
        hi = min(len(json_str), pos + 120)
        print("[extract_json_from_response] json.loads FAILED:", str(e), flush=True)
        print("[extract_json_from_response] error pos:", pos, flush=True)
        print("[extract_json_from_response] context around pos:\n", json_str[lo:hi], flush=True)
        raise ValueError(f"JSON decoding failed: {e}")
