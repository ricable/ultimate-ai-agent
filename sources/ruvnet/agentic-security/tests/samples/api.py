import json
import requests
from typing import Union, Dict, List

def fetch_user_data(user_id):
    """Secure API endpoint"""
    # Validate user_id input
    if not isinstance(user_id, int) or user_id < 0:
        raise ValueError("Invalid user_id")

    # Use HTTPS and verify SSL certificate
    response = requests.get(f"https://api.example.com/users/{user_id}", verify=True)
    response.raise_for_status()
    return response.json()

from defusedxml import ElementTree as ET
import re
import logging

def parse_xml_data(xml_string):
    """Secure XML parsing with XXE protection and strict validation"""
    if not isinstance(xml_string, str):
        logging.error("Invalid input type for XML parsing")
        raise TypeError("XML input must be a string")

    # Disable external entity resolution to prevent XXE attacks
    parser = ET.XMLParser(resolve_entities=False)

    # Validate and sanitize input XML string
    try:
        xml_string = xml_string.strip()
        if not xml_string:
            raise ValueError("Invalid XML string: Empty input")
        
        # Comprehensive input validation
        malicious_patterns = [
            r'<!ENTITY',
            r'<!DOCTYPE',
            r'<!ELEMENT',
            r'<!ATTLIST',
            r'<\?xml-stylesheet',
            r'data:',
            r'file:',
            r'gopher:',
            r'http:',
            r'ftp:'
        ]
        for pattern in malicious_patterns:
            if re.search(pattern, xml_string, re.IGNORECASE):
                raise ValueError(f"Invalid XML string: Potentially malicious content detected: {pattern}")
                
    except ValueError as e:
        logging.error(f"XML Validation Error: {str(e)}")
        raise

    # Parse XML string securely
    try:
        tree = ET.fromstring(xml_string, parser=parser)
    except ET.ParseError as e:
        # Handle XML parsing errors with proper error logging
        logging.error(f"Error: {e}")
        return None

    # Additional input sanitization
    # Remove potential malicious nodes from the parsed tree
    sanitize_tree(tree)

    return tree

def sanitize_tree(tree: ET.Element) -> None:
    """Sanitize the parsed XML tree to remove potential malicious nodes"""
    if not isinstance(tree, ET.Element):
        raise TypeError("Expected ElementTree.Element")
        
    # Comprehensive list of potentially dangerous tags
    malicious_tags = {
        'ENTITY', 'DOCTYPE', 'ELEMENT', 'ATTLIST',
        'NOTATION', 'CDATA', 'SYSTEM', 'PUBLIC'
    }
    
    # Remove nodes with potentially malicious content
    for elem in tree.iter():
        if elem.tag in malicious_tags:
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
                
        # Check for suspicious attributes
        for attr in list(elem.attrib.keys()):
            if any(x in attr.lower() for x in ['script', 'on', 'xmlns']):
                del elem.attrib[attr]

def send_request(url: str, data: Union[Dict, List]) -> str:
    """Secure request handling with validation and proper SSL verification"""
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    # Validate URL format and scheme
    if not re.match(r'^https?://[\w\-\.]+(:\d+)?(/[\w\-\./]*)?$', url):
        raise ValueError("Invalid URL format")
    if not url.startswith('https://'):
        raise ValueError("Only HTTPS URLs are allowed")
        
    # Validate data
    if not isinstance(data, (dict, list)):
        raise ValueError("Data must be a dictionary or list")
        
    try:
        response = requests.post(
            url,
            json=data,
            verify=True,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        raise

def process_response(response_data):
    """Secure response handling with validation"""
    if not isinstance(response_data, str):
        logging.error("Invalid response data type")
        raise TypeError("Response data must be a string")

    try:
        # Set strict parsing options
        parsed_data = json.loads(
            response_data,
            parse_float=decimal.Decimal,  # Use Decimal for precise floating point
            parse_constant=lambda x: ValueError(f'Invalid constant {x}')  # Reject inf/nan
        )
        
        # Validate parsed data structure
        if not isinstance(parsed_data, (dict, list)):
            raise ValueError("Response must be a JSON object or array")
            
        return parsed_data
        
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"JSON Processing Error: {str(e)}")
        raise
import json
import decimal
import logging
import requests
import subprocess
import shlex
import re
from defusedxml import ElementTree as ET
from typing import Union, Dict, List, Any

import shlex

def execute_command(cmd):
    """Secure command execution with strict validation"""
    if not isinstance(cmd, str):
        logging.error("Invalid command type")
        raise TypeError("Command must be a string")

    # Whitelist of allowed commands
    ALLOWED_COMMANDS = {'ls', 'dir', 'echo', 'pwd'}
    
    try:
        # Safely split command
        cmd_args = shlex.split(cmd)
        if not cmd_args:
            raise ValueError("Empty command")
            
        # Validate base command
        base_cmd = cmd_args[0].lower()
        if base_cmd not in ALLOWED_COMMANDS:
            raise ValueError(f"Command not allowed: {base_cmd}")
            
        # Additional argument validation
        for arg in cmd_args[1:]:
            if arg.startswith('-'):
                raise ValueError(f"Command flags not allowed: {arg}")
            if any(c in arg for c in ';&|$()`'):
                raise ValueError(f"Invalid character in argument: {arg}")
                
        # Execute with restricted permissions
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,  # Prevent shell injection
            timeout=10,   # Prevent hanging
            check=True    # Raise on non-zero exit
        )
        return result.stdout
        
    except (ValueError, subprocess.SubprocessError) as e:
        logging.error(f"Command Execution Error: {str(e)}")
        raise
