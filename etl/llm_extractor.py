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
    Updated to extract 'email_id' and 'phone_number' separately.
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
         "- 'email_id': Candidate's primary email address (string). Use 'Not Found' if not explicitly found.\n" # NEW FIELD
         "- 'phone_number': Candidate's primary phone number (string). Use 'Not Found' if not explicitly found.\n\n" # NEW FIELD
         "Your response MUST start with the opening curly brace `{{` and end with the closing curly brace `}}`. "
         "Example JSON structure (observe the single string for experience/education, and separate contact fields):\n"
         "```json\n"
         "{{\n"
         "  \"name\": \"John Doe\",\n"
         "  \"title\": \"Software Engineer\",\n"
         "  \"department\": \"Engineering\",\n"
         "  \"skills\": [\"Python\", \"AWS\", \"Docker\"],\n"
         "  \"experience\": \"5 years in software development at Example Corp, responsible for backend services and cloud deployments, including a previous role at ABC Inc. as a Junior Developer.\",\n"
         "  \"education\": \"B.Sc. in Computer Science from University X (2018-2022) and an online certificate in Data Science from XYZ Academy.\",\n"
         "  \"email_id\": \"john.doe@example.com\",\n" # NEW EXAMPLE
         "  \"phone_number\": \"+1-555-123-4567\"\n" # NEW EXAMPLE
         "}}\n"
         "```\n"
         "Ensure ALL string values within the JSON are correctly escaped if they contain double quotes or backslashes. DO NOT include any other text, preambles, or explanations outside the JSON object. Just the JSON."
        ),
        ("user", "Extract details from this resume text:\n{resume_text}")
    ])

def process_resume_file(file_path, llm, extraction_prompt, load_document_content_func):
    """
    Reads a raw resume file, extracts structured data using LLM, and returns it.
    Updated to handle new 'email_id' and 'phone_number' fields.
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
            extraction_raw_text = llm_response.content # Corrected typo here
        else:
            extraction_raw_text = str(llm_response) # Assume it's a string directly if no .content

        # Clean up the raw text to ensure it's valid JSON
        cleaned_json_string = extraction_raw_text.strip()
        
        # New robust cleaning: find the first '{' and last '}'
        start_index = cleaned_json_string.find('{')
        end_index = cleaned_json_string.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            cleaned_json_string = cleaned_json_string[start_index : end_index + 1]
        else:
            print(f"    WARNING: Could not find valid JSON boundaries for {os.path.basename(file_path)}. Raw LLM output: {cleaned_json_string}")
            return None

        # Attempt JSON parsing with basic healing
        try:
            extracted_data = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"    WARNING: JSONDecodeError for {os.path.basename(file_path)}. Error: {e}")
            print(f"    Raw LLM output causing error (first 500 chars): {cleaned_json_string[:500]}...")
            if not cleaned_json_string.endswith('}'):
                print("    Attempting to heal JSON by adding '}'...")
                try:
                    extracted_data = json.loads(cleaned_json_string + '}')
                except json.JSONDecodeError:
                    print("    JSON healing failed. Returning None.")
                    return None
            else:
                return None

        # Clean up "Not Found" or empty values for better handling
        # Also ensure 'email_id' and 'phone_number' are handled
        fields_to_check = ['name', 'department', 'experience', 'education', 'email_id', 'phone_number']
        for key in fields_to_check:
            if key in extracted_data and (extracted_data[key] == "Not Found" or extracted_data[key] is None or (isinstance(extracted_data[key], str) and extracted_data[key].strip() == '')):
                extracted_data[key] = None
            
        # For 'experience' and 'education' which are now forced to be single strings,
        # if the LLM still returns a list, convert it to a string.
        for key in ['experience', 'education']:
            if key in extracted_data and isinstance(extracted_data[key], list):
                extracted_data[key] = " ".join(map(str, extracted_data[key])) if extracted_data[key] else None
        
        # Ensure skills is a list of strings
        if 'skills' in extracted_data:
            value = extracted_data['skills']
            if isinstance(value, str):
                extracted_data['skills'] = [s.strip() for s in value.split(',') if s.strip()] if value != "Not Found" else []
            elif not isinstance(value, list):
                extracted_data['skills'] = []


        # Generate a simple ID for the profile, useful for tracking
        if 'id' not in extracted_data or not extracted_data['id']:
             extracted_data['id'] = os.path.basename(file_path).split('.')[0] # Use filename as ID
        
        # Remove the old 'contact_info' if it somehow still exists from an old prompt/model behavior
        extracted_data.pop('contact_info', None)

        print(f"  Successfully extracted data from {os.path.basename(file_path)}.")
        return extracted_data

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        return None