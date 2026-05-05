from reactxen.utils.model_inference import watsonx_llm
from agent_hive.agents.plan_reviewer_prompt import review_plan_system_prompt_template
import json
import re
from agent_hive.agents.base_agent import BaseAgent
from typing import List, Dict
import time
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)


class PlanReviewerAgent(BaseAgent):
    """
    This class is responsible for evaluating the generated plan based on the given criteria.
    It uses a language model to generate a review of the plan and then parses the JSON output from the model.
    """
    name = "PlanReviewerAgent"
    description = "This agent evaluates the generated plan based on predefined criteria."
    memory = []
    tools = []

    def __init__(self, llm="mistralai/mistral-large", max_retries=3):
        self.llm = llm
        self.max_retries = max_retries

    def extract_and_parse_json_using_manual_parser(self, response):

        cleaned_json_str = (
            response.strip().replace("\n", " ").replace("\\n", " ").replace("\\", "")
        )

        # Define regular expressions to extract each part:
        status_regex = r'"status":\s*"([^"]+)"'
        reasoning_regex = r'"reasoning":\s*"([^"]+)"'
        suggestions_regex = r'"suggestions":\s*"([^"]+)"'

        # Extract the values using regex
        status_match = re.search(status_regex, cleaned_json_str)
        reasoning_match = re.search(reasoning_regex, cleaned_json_str)
        suggestions_match = re.search(suggestions_regex, cleaned_json_str)

        # Extract and display the results if found
        if status_match and reasoning_match and suggestions_match:
            status = status_match.group(1)
            reasoning = reasoning_match.group(1)
            suggestions = suggestions_match.group(1)
            return {
                "status": status,
                "reasoning": reasoning,
                "suggestions": suggestions,
            }
        else:
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed.",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block.",
            }

    def extract_and_parse_json(self, response):
        """
        Extract and parse JSON from the response.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            dict: Parsed JSON object or an error report.
        """
        try:
            # Extract JSON block enclosed in ```json ... ```
            # match = re.search(r"```json(.*?)```", response, re.DOTALL)
            match = re.search(r"\{.*\}", response.strip(), re.DOTALL)
            if match:
                json_block = match.group(0).strip()  # Extract and clean the JSON block
            else:
                json_block = response.strip()

            if not json_block:
                raise ValueError("Extracted JSON block is empty.")

            parsed_json = json.loads(json_block)
            return parsed_json

        except json.JSONDecodeError as ex:
            return {
                "status": "Error",
                "reasoning": f"The extracted JSON block could not be parsed. {ex}",
                "suggestions": "Ensure the LLM outputs valid JSON inside the ```json``` block.",
            }

        except ValueError as ex:
            # print(f"Value Error: {ex}")
            return {
                "status": "Error",
                "reasoning": str(ex),
                "suggestions": "Check if the extracted JSON block is empty or improperly formatted.",
            }

    def execute_task(self, question: str, agent_descriptions: str, plan: str):
        """
        Evaluate the plan based on the question and agent expertise.

        Args:
            question (str): The user's question.
            agent_descriptions (str): Descriptions of the agents involved.
            plan (str): The plan to evaluate.

        Returns:
            tuple[dict, int, int]:
                - parsed review result
                - total input token count
                - total generated token count
        """
        input_tokens_count = 0
        generated_tokens_count = 0

        prompt = review_plan_system_prompt_template.format(
            question=question,
            agent_expertise=agent_descriptions,
            plan=plan,
        )
        logger.info(f"Review Prompt: {prompt}")

        for it_index in range(self.max_retries):
            resp = watsonx_llm(
                prompt,
                model_id=self.llm,
                stop=["\n(END OF RESPONSE)"],
            )
            llm_response = resp.get("generated_text", "")
            in_tok = resp.get("input_token_count", 0)
            out_tok = resp.get("generated_token_count", 0)

            print(f"in_tok2: {in_tok}")
            print(f"type(in_tok): {type(in_tok)}")
            print(f"out_tok2: {out_tok}")
            print(f"type(out_tok): {type(out_tok)}")

            input_tokens_count += in_tok
            generated_tokens_count += out_tok
            logger.info(f"Plan: \n{llm_response}")

            review_result = llm_response

            parsed_result = self.extract_and_parse_json(review_result)
            if parsed_result.get("status") != "Error":
                return parsed_result, input_tokens_count, generated_tokens_count

            parsed_result = self.extract_and_parse_json_using_manual_parser(review_result)
            if parsed_result.get("status") != "Error":
                return parsed_result, input_tokens_count, generated_tokens_count

        review = {
            "status": "Error",
            "reasoning": f"Failed to produce valid JSON after {self.max_retries} attempts.",
            "suggestions": "Review the prompt and refine the LLM response strategy.",
        }
        return review, input_tokens_count, generated_tokens_count
