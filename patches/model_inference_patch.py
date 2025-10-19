from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
import os

from ibm_watsonx_ai.foundation_models.schema import TextGenParameters, TextGenDecodingMethod, TextChatParameters

load_dotenv()

def langchain_watsonx_llm(
        model_id="mistralai/mistral-medium-2505",  # ← デフォルトモデルを安全なものに変更
        decoding_method=TextGenDecodingMethod.GREEDY,
        max_new_tokens=10000,
        min_new_tokens=0,
        stop=None,
):
    api_key = os.environ["WATSONX_APIKEY"]
    project_id = os.environ["WATSONX_PROJECT_ID"]
    url = os.environ["WATSONX_URL"]

    parameters = {
        "max_tokens": max_new_tokens,
        "stop": stop,
        'temperature': 0.0
    }

    model = ChatWatsonx(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=parameters,
        api_key=api_key,
    )
    return model
