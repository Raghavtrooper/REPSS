import json
import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    Updated to explicitly use 'resume_text' as the input variable and
    escape curly braces in the example JSON.
    Added 'companies_worked_with_duration' field.
    """
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at extracting structured information from resume text. "
         "Extract the following details in JSON format. If a field is not explicitly found or is empty, use 'Not Found' for strings or an empty list [] for skills. "
         "The output MUST be a VALID AND COMPLETE JSON object. Pay EXTREME attention to JSON syntax, including commas and closing braces. "
         "The JSON object MUST have the following keys, listed in this order:\n"
         "- 'name': Full name of the candidate (string).\n"
         "- 'email_id': Candidate's primary email address (string). Use 'Not Found' if not explicitly found.\n"
         "- 'phone_number': Candidate's primary phone number (string). Use 'Not Found' if not explicitly found.\n"
         "- 'location': Candidate's living address or general location (string). Use 'Not Found' if not explicitly found.\n"
         "- 'objective': Candidate's career objective or summary statement (string). Use 'Not Found' if not explicitly found.\n"
         "- 'skills': A JSON array of key technical or soft skills (list of strings). If no skills, provide an empty array `[]`.\n"
         "- 'qualifications_summary': **CRITICAL**: Provide a SINGLE, CONCISE TEXT SUMMARY of ALL education history and qualifications, including degree(s) and year(s) of passing. DO NOT provide a list of education objects; flatten all education into ONE descriptive string.\n"
         "- 'experience_summary': **CRITICAL**: Provide a SINGLE, COMPREHENSIVE TEXT SUMMARY of ALL work experience. This summary MUST include details about each company (e.g., Company A, Company B, Company C), their respective dates (e.g., from 01-10-2007 to 21st-03-2010), project descriptions, and roles and responsibilities. DO NOT provide a list of experience objects or detailed bullet points; flatten all experience into ONE descriptive string.\n"
         "- 'companies_worked_with_duration': A JSON array of strings, where each string represents a company and the duration worked there, extracted from the experience summary. Format each entry as 'Company Name (Start Date - End Date)' or 'Company Name (Total Duration)'. If no companies found, provide an empty array `[]`.\n\n"
         "Your response MUST start with the opening curly brace `{{` and end with the closing curly brace `}}`. "
         "Example JSON structure (observe the single string for experience/qualifications, and separate contact fields):\n"
         "```json\n"
         "{{\n" # Escaped
         "  \"name\": \"John Doe\",\n"
         "  \"email_id\": \"john.doe@example.com\",\n"
         "  \"phone_number\": \"+1-555-123-4567\",\n"
         "  \"location\": \"New York, NY\",\n"
         "  \"objective\": \"Highly motivated software engineer seeking challenging opportunities in cloud development.\",\n"
         "  \"skills\": [\"Python\", \"AWS\", \"Docker\"],\n"
         "  \"qualifications_summary\": \"B.Sc. in Computer Science from University X (2018-2022) and an online certificate in Data Science from XYZ Academy.\",\n"
         "  \"experience_summary\": \"5 years in software development. At Company A (01-10-2007 to 21-03-2010), developed backend services for e-commerce. At Company B (01-04-2010 to 21-03-2017), led a team for cloud deployments. At Company C (01-04-2017 to till date), responsible for AI integration projects.\",\n"
         "  \"companies_worked_with_duration\": [\"Company A (01-10-2007 - 21-03-2010)\", \"Company B (01-04-2010 - 21-03-2017)\", \"Company C (01-04-2017 - till date)\"]\n"
         "}}\n" # Escaped
         "```\n"
         "Ensure ALL string values within the JSON are correctly escaped if they contain double quotes or backslashes. DO NOT include any other text, preambles, or explanations outside the JSON object. Just the JSON."
        ),
        ("user", "Extract details from this resume text:\n{resume_text}")
    ])

def process_resume_file(llm, extraction_prompt, document_content, original_filename, doc_metadata=None):
    """
    Reads a raw resume file, extracts structured data using LLM, and returns it.
    Updated to handle new 'location', 'objective', 'qualifications_summary', and
    'experience_summary' fields, 'has_photo' metadata, and the new
    'companies_worked_with_duration' field.
    """
    if document_content is None or not document_content.strip():
        print(f"  Skipping {os.path.basename(original_filename)}: Could not load content or content is empty.")
        return None

    print(f"  Processing: {os.path.basename(original_filename)}...")
    
    try:
        # Construct a runnable chain for explicit input variable mapping
        # The prompt expects an input variable named 'resume_text'
        chain = {"resume_text": RunnablePassthrough()} | extraction_prompt | llm

        # Invoke the chain, passing the document_content as the value for 'resume_text'
        llm_response = chain.invoke({"resume_text": document_content})
        
        if hasattr(llm_response, 'content'):
            extraction_raw_text = llm_response.content
        else:
            extraction_raw_text = str(llm_response)

        # Clean up the raw text to ensure it's valid JSON
        cleaned_json_string = extraction_raw_text.strip()
        
        # New robust cleaning: find the first '{' and last '}'
        start_index = cleaned_json_string.find('{')
        end_index = cleaned_json_string.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            cleaned_json_string = cleaned_json_string[start_index : end_index + 1]
        else:
            print(f"    WARNING: Could not find valid JSON boundaries for {os.path.basename(original_filename)}. Raw LLM output: {cleaned_json_string}")
            return None

        # Attempt JSON parsing with basic healing
        try:
            extracted_data = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            print(f"    WARNING: JSONDecodeError for {os.path.basename(original_filename)}. Error: {e}")
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
        fields_to_check = ['name', 'email_id', 'phone_number', 'location', 'objective', 
                           'experience_summary', 'qualifications_summary'] # Updated fields
        for key in fields_to_check:
            if key in extracted_data and (extracted_data[key] == "Not Found" or extracted_data[key] is None or (isinstance(extracted_data[key], str) and extracted_data[key].strip() == '')):
                extracted_data[key] = None
            
        # For 'experience_summary' and 'qualifications_summary' which are now forced to be single strings,
        # if the LLM still returns a list, convert it to a string.
        for key in ['experience_summary', 'qualifications_summary']:
            if key in extracted_data and isinstance(extracted_data[key], list):
                extracted_data[key] = " ".join(map(str, extracted_data[key])) if extracted_data[key] else None
        
        # Ensure skills is a list of strings
        if 'skills' in extracted_data:
            value = extracted_data['skills']
            if isinstance(value, str):
                extracted_data['skills'] = [s.strip() for s in value.split(',') if s.strip()] if value != "Not Found" else []
            elif not isinstance(value, list):
                extracted_data['skills'] = []

        # Ensure 'companies_worked_with_duration' is a list of strings
        if 'companies_worked_with_duration' in extracted_data:
            value = extracted_data['companies_worked_with_duration']
            if isinstance(value, str):
                # If LLM returns a string (e.g., "Company A (2020-2022), Company B (2018-2020)"), split it
                extracted_data['companies_worked_with_duration'] = [s.strip() for s in value.split(',') if s.strip()] if value != "Not Found" else []
            elif not isinstance(value, list):
                extracted_data['companies_worked_with_duration'] = []
        else:
            extracted_data['companies_worked_with_duration'] = [] # Ensure it always exists as a list

        # Merge document-level metadata (like has_photo) into the structured data
        if doc_metadata:
            extracted_data.update(doc_metadata)

        # Generate a simple ID for the profile, useful for tracking
        if 'id' not in extracted_data or not extracted_data['id']:
             extracted_data['id'] = os.path.basename(original_filename).split('.')[0] # Use filename as ID
        
        # Remove old field names that might still be generated by older prompt versions or model quirks
        extracted_data.pop('contact_info', None)
        extracted_data.pop('experience', None) 
        extracted_data.pop('education', None)
        extracted_data.pop('title', None)      # Remove 'title'
        extracted_data.pop('department', None) # Remove 'department'

        print(f"  Successfully extracted data from {os.path.basename(original_filename)}.")
        return extracted_data

    except Exception as e:
        print(f"An unexpected error occurred while processing {original_filename}: {e}")
        return None
