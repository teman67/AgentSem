import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import os
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import re
import requests
from dotenv import load_dotenv
load_dotenv()

def extract_rdf_shacl_improved(response_text):
    """Improved extraction with better error handling and debugging"""
    if not response_text or not response_text.strip():
        st.error("❌ Empty response from LLM")
        return "", ""
    
    parts = response_text.split("```")
    rdf_code = ""
    shacl_code = ""

    code_blocks = []

    for i, part in enumerate(parts):
        if i % 2 == 1:
            part_clean = part.strip()
            if part_clean.startswith("turtle"):
                part_clean = part_clean[6:].strip()
            elif part_clean.startswith("ttl"):
                part_clean = part_clean[3:].strip()
            if "@prefix" in part_clean or "sh:" in part_clean:
                code_blocks.append(part_clean)

    if len(code_blocks) >= 1:
        rdf_code = code_blocks[0]
    if len(code_blocks) >= 2:
        shacl_code = code_blocks[1]

    return rdf_code, shacl_code


def call_llm(prompt, system_prompt, model_info):
    """
    Calls the specified LLM with the given prompt and system prompt.
    Args:
        prompt (str): The user prompt to send to the LLM.
        system_prompt (str): The system prompt to set the context for the LLM.
        model_info (dict): Information about the model, including provider, model name, and API key.
    Returns:
        str: The response from the LLM.
    """
    provider = model_info["provider"]
    model = model_info["model"]
    temperature = model_info.get("temperature", 0.3)

    if provider == "OpenAI":
        client = OpenAI(api_key=model_info["api_key"])
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif provider == "Anthropic":
        client = Anthropic(api_key=model_info["api_key"])
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif provider == "Ollama":
        endpoint = model_info["endpoint"]
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
        }
        try:
            response = requests.post(f"{endpoint}/api/chat", json=data)
            response.raise_for_status()  # raise error if status code is 4xx/5xx

            json_data = response.json()
            if "message" in json_data and "content" in json_data["message"]:
                return json_data["message"]["content"]
            elif "error" in json_data:
                raise ValueError(f"Ollama API error: {json_data['error']}")
            else:
                raise ValueError("Unexpected Ollama response format.")

        except Exception as e:
            raise RuntimeError(f"Failed to call Ollama API: {str(e)}")
    return response.json()["message"]["content"]
