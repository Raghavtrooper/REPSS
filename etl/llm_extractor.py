import json
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

def initialize_ollama_llm(model_name):
    """
    Initializes and returns the Ollama LLM for structured extraction.
    """
    try:
        llm = Ollama(model=model_name, temperature=0.0) # Keep temperature low for deterministic JSON
        # Test if the LLM is reachable
        llm.invoke("Hello", stream=False) # Simple invoke to check connectivity
        print(f"Ollama LLM '{model_name}' initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Ollama LLM '{model_name}'. "
              f"Make sure Ollama is running and the model is pulled (`ollama pull {model_name}`). Error: {e}")
        raise

def get_extraction_prompt():
    """
    Defines and returns the ChatPromptTemplate for structured data extraction.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at extracting structured information from resume text. "
         "Extract the following details in JSON format. If a field is not explicitly found or is empty, use 'Not Found' for strings or an empty list [] for skills. "
         "The output MUST be a VALID AND COMPLETE JSON object. Pay EXTREME attention to JSON syntax, including commas and closing braces. "
         "The JSON object MUST have the following keys:\n"
         "- 'name': Full name of the candidate (string).\n"
         "- 'title': Current or most recent job title (string).\n"
         "- 'department': Candidate's department or primary field (e.g., Engineering, Marketing) (string).\n"
         "- 'skills': A JSON array of key technical or soft skills (list of strings). If no skills, provide an empty array `[]`.\n"
         "- 'experience': **CRITICAL**: Provide a SINGLE, CONCISE TEXT SUMMARY of ALL work experience. DO NOT provide a list of experience objects or detailed bullet points; flatten all experience into ONE descriptive string.\n"
         "- 'education': **CRITICAL**: Provide a SINGLE, CONCISE TEXT SUMMARY of ALL education history. DO NOT provide a list of education objects; flatten all education into ONE descriptive string.\n"
         "- 'contact_info': Candidate's primary contact information (email, phone, LinkedIn if present) (string), e.g., 'john.doe@example.com | 123-456-7890'.\n\n"
         "Your response MUST start with the opening curly brace `{{` and end with the closing curly brace `}}`. "
         "Example JSON structure (observe the single string for experience/education):\n"
         "```json\n"
         "{{\n"
         "  \"name\": \"John Doe\",\n"
         "  \"title\": \"Software Engineer\",\n"
         "  \"department\": \"Engineering\",\n"
         "  \"skills\": [\"Python\", \"AWS\", \"Docker\"],\n"
         "  \"experience\": \"5 years in software development at Example Corp, responsible for backend services and cloud deployments, including a previous role at ABC Inc. as a Junior Developer.\",\n"
         "  \"education\": \"B.Sc. in Computer Science from University X (2018-2022) and an online certificate in Data Science from XYZ Academy.\",\n"
         "  \"contact_info\": \"john.doe@example.com | +1-555-123-4567\"\n"
         "}}\n"
         "```\n"
         "Ensure ALL string values within the JSON are correctly escaped if they contain double quotes or backslashes. DO NOT include any other text, preambles, or explanations outside the JSON object. Just the JSON."
        ),
        ("user", "Extract details from this resume text:\n{resume_text}")
    ])

def process_resume_file(file_path, llm, extraction_prompt, load_document_content_func):
    """
    Reads a raw resume file, extracts structured data using LLM, and returns it.
    """
    raw_text = load_document_content_func(file_path)
    if raw_text is None or not raw_text.strip():
        print(f"  Skipping {os.path.basename(file_path)}: Could not load content or content is empty.")
        return None

    print(f"  Processing: {os.path.basename(file_path)}...")
    
    try:
        # Invoke LLM for extraction
        llm_response = llm.invoke(extraction_prompt.format_messages(resume_text=raw_text))
        
        if hasattr(llm_response, 'content'):
            extraction_raw_text = llm_response.content
        else:
            extraction_raw_text = str(llm_response) # Assume it's a string directly if no .content

        # Clean up the raw text to ensure it's valid JSON
        cleaned_json_string = extraction_raw_text.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()
        
        # Attempt JSON parsing with basic healing
        try:
            extracted_data = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"    WARNING: JSONDecodeError for {os.path.basename(file_path)}. Error: {e}")
            print(f"    Raw LLM output causing error (first 500 chars): {cleaned_json_string[:500]}...")
            # Simple JSON healing: if it looks like it's cut off at the end, try adding '}'
            if not cleaned_json_string.endswith('}'):
                print("    Attempting to heal JSON by adding '}'...")
                try:
                    extracted_data = json.loads(cleaned_json_string + '}')
                except json.JSONDecodeError:
                    print("    JSON healing failed. Returning None.")
                    return None # Return None if parsing still fails
            else:
                return None # If it's not a simple missing brace, return None

        # Clean up "Not Found" or empty values for better handling
        for key, value in extracted_data.items():
            if value == "Not Found":
                extracted_data[key] = None
            elif isinstance(value, list) and not value:
                extracted_data[key] = []
            elif value is None:
                extracted_data[key] = None
            
            # For 'experience' and 'education' which are now forced to be single strings,
            # if the LLM still returns a list, convert it to a string.
            if key in ['experience', 'education'] and isinstance(value, list):
                extracted_data[key] = " ".join(map(str, value)) if value else None
            # Ensure skills is a list of strings
            if key == 'skills':
                if isinstance(value, str):
                    extracted_data[key] = [s.strip() for s in value.split(',') if s.strip()] if value != "Not Found" else []
                elif not isinstance(value, list):
                    extracted_data[key] = []

        # Generate a simple ID for the profile, useful for tracking
        if 'id' not in extracted_data or not extracted_data['id']:
             extracted_data['id'] = os.path.basename(file_path).split('.')[0] # Use filename as ID
        
        print(f"  Successfully extracted data from {os.path.basename(file_path)}.")
        return extracted_data

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None
