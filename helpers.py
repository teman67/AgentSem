import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from anthropic import AuthenticationError as AnthropicAuthError
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import streamlit.web.server.websocket_headers as ws_headers
import uuid
import requests
import openai
import hashlib
import re
import os
import requests
import json
from supabase import create_client, Client
import datetime

def api_call():
    """
    Placeholder for an API call function.
    This should be replaced with actual API interaction logic.
    """
    provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Ollama"], key="provider_select")
    if provider == "OpenAI":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        model = st.sidebar.selectbox("OpenAI Model", model_options, index=0, key="openai_model")
        st.sidebar.markdown("Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
    elif provider == "Anthropic":
        # model_options = ["claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"]
        model_options = ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest", "claude-opus-4-20250514"]
        model = st.sidebar.selectbox("Claude Model", model_options, index=0, key="anthropic_model")
        st.sidebar.markdown("Get your Anthropic API key from [Anthropic Console](https://www.merge.dev/blog/anthropic-api-key)")
    else:
        model_options = ["llama3.3:70b-instruct-q8_0", "qwen3:32b-q8_0", "phi4-reasoning:14b-plus-fp16" , "mistral-small3.1:24b-instruct-2503-q8_0"]
        model = st.sidebar.selectbox("Ollama Model", model_options, index=0, key="ollama_model")

    return provider, model

@staticmethod
def get_input_data() -> str:
    """Handle file upload and text input"""
    import streamlit as st

    try:
        with open("BAM_Creep.txt", "r") as file:
            file_content = file.read()
    except FileNotFoundError:
        file_content = "Example file not found"
    
    st.subheader("🔬 Input Test Data")
    uploaded_file = st.file_uploader("Upload a file with mechanical test data", type=["txt", "csv", "json", "lis"])
    example = st.checkbox("Use example input")

    user_input = ""

    if uploaded_file is not None:
        file_data = uploaded_file.read().decode("utf-8")
        user_input = st.text_area("Mechanical Test Description (from uploaded file):", value=file_data, height=300)
    elif example:
        user_input = st.text_area("Mechanical Test Description (example):", value=file_content, height=300)
    else:
        user_input = st.text_area("Mechanical Test Description:", placeholder="Paste mechanical test data here...", height=200)

    return file_content, user_input


def extract_core_error(report):
    """
    Extract the core error message, removing variable elements like timestamps,
    line numbers, and URIs that might change between attempts
    """
    # Remove common variable elements - more comprehensive patterns
    cleaned = re.sub(r'line \d+', 'line X', report)
    cleaned = re.sub(r'column \d+', 'column X', cleaned)
    #Remove the codes to make the app Private
    
   
    #Remove the codes to make the app Private
    
    return ' '.join(unique_errors)

def is_syntax_error(report):
    """
    More precise detection of actual Turtle syntax errors vs SHACL validation errors
    """
    syntax_indicators = [
        "Bad syntax at line",
        "objectList expected",
        "Expected one of",
        "Unexpected end of file",
        "Invalid escape sequence",
        "Malformed URI",
        "Invalid character"
    ]
    
    # Must have syntax indicator AND not be a SHACL constraint violation
    has_syntax_indicator = any(indicator in report for indicator in syntax_indicators)
    has_shacl_violation = "Constraint Violation" in report or "sh:" in report
    
    return has_syntax_indicator and not has_shacl_violation

def should_retry_correction(report, previous_core_errors, max_same_error=2):
    """
    Determine if we should retry correction based on error patterns
    """
    core_error = extract_core_error(report)
    
    # Count how many times we've seen this core error
    same_error_count = previous_core_errors.count(core_error)
    
    return same_error_count < max_same_error

def safe_execute(func, *args, **kwargs):
    """Safely execute a function and handle all exceptions"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Log the error for debugging (optional)
        print(f"Error: {e}")  # This will only show in your console, not to users
        raise e  # Re-raise to be caught by your main try-catch
    
import re

def basic_syntax_cleanup(rdf_text, shacl_text):
    """
    Perform extensive Turtle syntax cleanup that LLMs commonly mess up.
    """
    fixes = [
        
        # Remove leading b' or b" markers from LLM output
        (r"b'([^']*)'", r"\1"),
        (r'b"([^"]*)"', r'\1'),
        
        # Fix '^b' artifacts
        #Remove the codes to make the app Private
        # Remove triple endings with '^b'
        #Remove the codes to make the app Private

        # Fix patterns like: ex:SomeTerm'^b'
        #Remove the codes to make the app Private

        # Fix broken string literals across lines
        #Remove the codes to make the app Private

        # Fix malformed URI fragments
        #Remove the codes to make the app Private

    ]

    def clean(text):
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text.strip()

    return clean(rdf_text), clean(shacl_text)


def validate_turtle_syntax(turtle_text):
    """
    Check if Turtle syntax is valid by trying to parse it
    """
    try:
        temp_graph = Graph()
        temp_graph.parse(data=turtle_text, format="turtle")
        return True, "Valid syntax"
    except Exception as e:
        return False, str(e)

def validate_api_key(provider, api_key, endpoint=None):
    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
        elif provider == "Anthropic":
            client = Anthropic(api_key=api_key)
            client.models.list()
        elif provider == "Ollama":
            # For Ollama, we can just check if the endpoint is reachable
            if not endpoint:
                raise ValueError("No endpoint provided for Ollama.")
            response = requests.get(f"{endpoint}/v1/models", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama responded with status code {response.status_code}")
        return True, ""
    except openai.AuthenticationError:
        return False, "❌ **Invalid OpenAI API Key**\n\nPlease verify your API key at https://platform.openai.com/account/api-keys"
    except AnthropicAuthError:
        return False, "❌ **Invalid Anthropic API Key**\n\nPlease verify your API key at https://console.anthropic.com/settings/keys"
    except requests.exceptions.RequestException as e:
        return False, f"❌ **Cannot Connect to Ollama**\n\nFailed to connect to Ollama at `{endpoint}`. Please make sure Ollama is running and accessible. Error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error while validating API key: {str(e)}"
    


def visualize_rdf(rdf_text):
    """
    Visualize RDF data using NetworkX and Pyvis with improved styling and layout.
    """
    try:
        g = Graph().parse(data=rdf_text, format="turtle")
        nx_graph = nx.DiGraph()

        #Remove the codes to make the app Private
    except Exception as e:
        st.error(f"Failed to parse RDF: {e}")
        return None
    

def extract_local_name(uri: str) -> str:
    """Extract the local name from a URI"""
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    return uri

def count_term_occurrences(text: str, term: str) -> int:
    """Count occurrences of a term in text using word boundaries"""
    import re
    pattern = rf'\b{re.escape(term)}\b'
    return len(re.findall(pattern, text, re.IGNORECASE))

def find_term_context(text: str, term: str, max_examples: int = 5) -> list[tuple[int, str]]:
    """Find lines containing the term with line numbers"""
    import re
    lines = text.split('\n')
    pattern = rf'\b{re.escape(term)}\b'
    
    matches = []
    for i, line in enumerate(lines, 1):
        if re.search(pattern, line, re.IGNORECASE):
            matches.append((i, line))
            if len(matches) >= max_examples:
                break
    
    return matches


def get_and_increment_app_counter():
    """Get current app opening count and increment it using Supabase"""
    try:
        # Get Supabase credentials from environment variables or Streamlit secrets
        supabase_url = os.getenv('SUPABASE_URL') or st.secrets.get('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY') or st.secrets.get('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in your environment or secrets.")
            return 1
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Try to get current counter
        response = supabase.table('app_counter').select('*').eq('app_name', 'agentsem').execute()
        
        if response.data:
            # Counter exists, increment it
            current_count = response.data[0]['count']
            new_count = current_count + 1
            
            # Update counter in database
            update_response = supabase.table('app_counter').update({
                'count': new_count,
                'last_opened': datetime.datetime.now().isoformat()
            }).eq('app_name', 'agentsem').execute()
            
            return new_count
        else:
            # Counter doesn't exist, create it
            insert_response = supabase.table('app_counter').insert({
                'app_name': 'agentsem',
                'count': 1,
                'last_opened': datetime.datetime.now().isoformat()
            }).execute()
            
            return 1
            
    except Exception as e:
        st.error(f"Error accessing Supabase: {str(e)}")


def get_user_location():
    """Get user's IP address and country information - Cloud deployment compatible"""
    try:
        ip_address = None
        
        # Method 1: Try to get real user IP from headers (works in cloud deployments)
        try:
            # Check for forwarded headers that contain the real user IP
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                headers = st.context.headers
                
                # Common headers that contain real user IP
                possible_ip_headers = [
                    'X-Forwarded-For',
                    'X-Real-IP', 
                    'X-Client-IP',
                    'CF-Connecting-IP',  # Cloudflare
                    'X-Forwarded-Proto',
                    'HTTP_X_FORWARDED_FOR',
                    'HTTP_X_REAL_IP',
                    'HTTP_CLIENT_IP'
                ]
                
                for header in possible_ip_headers:
                    if header in headers:
                        ip_address = headers[header].split(',')[0].strip()
                        if ip_address and ip_address != '127.0.0.1':
                            break
        except:
            pass
        
        # Method 2: Use JavaScript to get user IP (more reliable for cloud)
        if not ip_address:
            # Create a custom component to get user IP using JavaScript
            ip_address = get_user_ip_via_js()
        
        # Method 3: Fallback to external service (this will still show server IP on cloud)
        if not ip_address:
            try:
                ip_response = requests.get('https://api.ipify.org?format=json', timeout=5)
                ip_data = ip_response.json()
                ip_address = ip_data.get('ip')
            except:
                ip_address = 'unknown'
        
        # Get country information from IP
        country_info = {
            'country': 'Unknown',
            'country_code': 'XX',
            'city': 'Unknown',
            'region': 'Unknown'
        }
        
        if ip_address and ip_address != 'unknown':
            try:
                # Use multiple geolocation services for better accuracy
                country_info = get_location_from_ip(ip_address)
            except:
                pass
        
        return ip_address, country_info
    
    except Exception as e:
        return 'unknown', {
            'country': 'Unknown',
            'country_code': 'XX',
            'city': 'Unknown',
            'region': 'Unknown'
        }

def get_location_from_ip(ip_address):
    """Get location info from IP using multiple services"""
    
    # List of free geolocation services
    services = [
        {
            'url': f'http://ip-api.com/json/{ip_address}',
            'parser': lambda data: {
                'country': data.get('country', 'Unknown'),
                'country_code': data.get('countryCode', 'XX'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('regionName', 'Unknown')
            } if data.get('status') == 'success' else None
        },
        {
            'url': f'https://ipapi.co/{ip_address}/json/',
            'parser': lambda data: {
                'country': data.get('country_name', 'Unknown'),
                'country_code': data.get('country_code', 'XX'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown')
            } if not data.get('error') else None
        },
        {
            'url': f'https://ipinfo.io/{ip_address}/json',
            'parser': lambda data: {
                'country': data.get('country', 'XX'),
                'country_code': data.get('country', 'XX'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown')
            } if 'country' in data else None
        }
    ]
    
    for service in services:
        try:
            response = requests.get(service['url'], timeout=5)
            data = response.json()
            result = service['parser'](data)
            
            if result and result['country'] != 'Unknown':
                return result
                
        except Exception as e:
            continue
    
    # If all services fail, return unknown
    return {
        'country': 'Unknown',
        'country_code': 'XX',
        'city': 'Unknown',
        'region': 'Unknown'
    }

def get_user_location_alternative():
    """Alternative method using browser-based detection"""
    try:
        # Use a more reliable browser-based approach
        # This creates a temporary HTML component that detects user location
        
        location_html = """
        <div id="location-detector" style="display: none;">
            <script>
            // Function to get user's approximate location
            function detectUserLocation() {
                // Try to get timezone info
                const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
                
                // Try to get language/locale info
                const language = navigator.language || navigator.userLanguage;
                
                // Try to get more detailed location if available
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        function(position) {
                            // Send location data back (if user allows)
                            const locationData = {
                                latitude: position.coords.latitude,
                                longitude: position.coords.longitude,
                                timezone: timezone,
                                language: language
                            };
                            
                            // You could reverse geocode this to get country
                            console.log('Location detected:', locationData);
                        },
                        function(error) {
                            // Fallback to timezone-based detection
                            const locationData = {
                                timezone: timezone,
                                language: language
                            };
                            console.log('Fallback location:', locationData);
                        }
                    );
                } else {
                    // Basic fallback
                    const locationData = {
                        timezone: timezone,
                        language: language
                    };
                    console.log('Basic location:', locationData);
                }
            }
            
            // Run detection
            detectUserLocation();
            </script>
        </div>
        """
        
        # You would need to implement a way to capture this data back to Streamlit
        # For now, we'll use the IP-based approach with better error handling
        
        return get_user_location()
        
    except:
        return get_user_location()

# MAIN SOLUTION: Enhanced IP detection for cloud deployment
def get_user_location_cloud_compatible():
    """
    Cloud-compatible IP detection that handles proxy headers correctly
    """
    try:
        ip_address = None
        
        # Method 1: Check if we can access request headers through Streamlit
        try:
            # This is a workaround - Streamlit doesn't directly expose request headers            
            # Try to get headers from Streamlit's internal request context
            headers = ws_headers.get_websocket_headers()
            
            if headers:
                # Common headers that contain real user IP
                ip_headers = [
                    'x-forwarded-for',
                    'x-real-ip',
                    'x-client-ip',
                    'cf-connecting-ip',
                    'x-forwarded-proto'
                ]
                
                for header in ip_headers:
                    if header in headers:
                        ip_address = headers[header].split(',')[0].strip()
                        if ip_address and ip_address not in ['127.0.0.1', 'localhost']:
                            break
        except:
            pass
        
        # Method 2: Use external service that returns client IP
        if not ip_address:
            try:
                # This service specifically returns the client's IP as seen by the server
                services = [
                    'https://httpbin.org/ip',  # Returns {"origin": "ip"}
                    'https://api.ipify.org?format=json',  # Returns {"ip": "ip"}
                    'https://ipapi.co/ip/',  # Returns just the IP
                ]
                
                for service in services:
                    try:
                        response = requests.get(service, timeout=5)
                        if service == 'https://httpbin.org/ip':
                            data = response.json()
                            ip_address = data.get('origin', '').split(',')[0].strip()
                        elif service == 'https://api.ipify.org?format=json':
                            data = response.json()
                            ip_address = data.get('ip')
                        else:
                            ip_address = response.text.strip()
                        
                        if ip_address:
                            break
                    except:
                        continue
            except:
                ip_address = 'unknown'
        
        # Get country information from IP
        country_info = {
            'country': 'Unknown',
            'country_code': 'XX',
            'city': 'Unknown',
            'region': 'Unknown'
        }
        
        if ip_address and ip_address != 'unknown':
            country_info = get_location_from_ip(ip_address)
        
        return ip_address, country_info
    
    except Exception as e:
        return 'unknown', {
            'country': 'Unknown',
            'country_code': 'XX',
            'city': 'Unknown',
            'region': 'Unknown'
        }

def get_user_ip_via_js():
    """Get user's real IP using JavaScript injection"""
    try:
        # Create a JavaScript component to get the user's IP
        js_code = """
        <script>
        async function getUserIP() {
            try {
                // Try multiple IP detection services
                const services = [
                    'https://api.ipify.org?format=json',
                    'https://ipapi.co/json/',
                    'https://ipinfo.io/json'
                ];
                
                for (const service of services) {
                    try {
                        const response = await fetch(service);
                        const data = await response.json();
                        const ip = data.ip || data.query;
                        if (ip) {
                            // Send IP back to Streamlit
                            window.parent.postMessage({type: 'USER_IP', ip: ip}, '*');
                            return ip;
                        }
                    } catch (e) {
                        continue;
                    }
                }
            } catch (error) {
                console.error('Error getting IP:', error);
            }
        }
        
        // Run when page loads
        getUserIP();
        </script>
        """
        
        return None
        
    except:
        return None
    

def get_country_flag(country_code):
    """Get flag emoji for country code"""
    if country_code == 'XX' or len(country_code) != 2:
        return '🌍'
    
    # Convert country code to flag emoji
    try:
        return ''.join(chr(ord(c) + 127397) for c in country_code.upper())
    except:
        return '🌍'

def log_app_access_and_increment_counter():
    """Log app access with IP/country info and increment counter"""
    try:
        # Get Supabase credentials
        supabase_url = os.getenv('SUPABASE_URL') or st.secrets.get('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY') or st.secrets.get('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            st.error("Supabase credentials not found.")
            return 1, {}
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get user location
        ip_address, country_info = get_user_location()
        
        # Log the access
        access_log = {
            'app_name': 'agentsem',
            'ip_address': ip_address,
            'country': country_info['country'],
            'country_code': country_info['country_code'],
            'city': country_info['city'],
            'region': country_info['region'],
            'accessed_at': datetime.datetime.now().isoformat()
        }
        
        # Insert access log
        supabase.table('app_access_logs').insert(access_log).execute()
        
        # Update/increment main counter
        response = supabase.table('app_counter').select('*').eq('app_name', 'agentsem').execute()
        
        if response.data:
            current_count = response.data[0]['count']
            new_count = current_count + 1
            
            supabase.table('app_counter').update({
                'count': new_count,
                'last_opened': datetime.datetime.now().isoformat()
            }).eq('app_name', 'agentsem').execute()
        else:
            new_count = 1
            supabase.table('app_counter').insert({
                'app_name': 'agentsem',
                'count': 1,
                'last_opened': datetime.datetime.now().isoformat()
            }).execute()
        
        return new_count, country_info
        
    except Exception as e:
        st.error(f"Error logging access: {str(e)}")

def get_country_statistics():
    """Get country statistics from database"""
    try:
        supabase_url = os.getenv('SUPABASE_URL') or st.secrets.get('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY') or st.secrets.get('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            return {}
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Get country statistics
        response = supabase.table('app_access_logs').select('country, country_code').eq('app_name', 'agentsem').execute()
        
        if response.data:
            country_counts = {}
            for record in response.data:
                country = record['country']
                country_code = record['country_code']
                if country != 'Unknown':
                    if country in country_counts:
                        country_counts[country]['count'] += 1
                    else:
                        country_counts[country] = {
                            'count': 1,
                            'country_code': country_code
                        }
            
            # Sort by count (descending)
            sorted_countries = dict(sorted(country_counts.items(), key=lambda x: x[1]['count'], reverse=True))
            return sorted_countries
        
        return {}
        
    except Exception as e:
        return {}