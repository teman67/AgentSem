import streamlit as st
from controller import SemanticPipelineAgent
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from anthropic import AuthenticationError as AnthropicAuthError
from rdflib import Graph
from pyshacl import validate
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import uuid
import requests
import openai
from agents import OntologyMatcherAgent
from helpers import *
import re
import os
from supabase import create_client, Client
import datetime

load_dotenv()

st.set_page_config(page_title="AgentSem", layout="wide")

if 'app_counter' not in st.session_state:
    st.session_state.app_counter, st.session_state.user_country_info = log_app_access_and_increment_counter()

st.title("🧠 AgentSem: Agent-Based Semantic Data Generator")

# Initialize session state variables
if 'rdf_generated' not in st.session_state:
    st.session_state.rdf_generated = False
if 'final_rdf_code' not in st.session_state:
    st.session_state.final_rdf_code = ""
if 'final_shacl_code' not in st.session_state:
    st.session_state.final_shacl_code = ""
if 'validation_status' not in st.session_state:
    st.session_state.validation_status = False
if 'validation_report' not in st.session_state:
    st.session_state.validation_report = ""
if 'replacement_status' not in st.session_state:
    st.session_state.replacement_status = ""
if 'replacement_count' not in st.session_state:
    st.session_state.replacement_count = 0
if 'user_input_text' not in st.session_state:
    st.session_state.user_input_text = ""
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = ""

# Sidebar: API Configuration
st.sidebar.header("🔐 API Configuration")
provider, model = api_call()
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

api_key = ""
endpoint = ""

if provider in ["OpenAI", "Anthropic"]:
    api_key = st.sidebar.text_input("API Key", type="password")
else:
    endpoint = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")

st.sidebar.header("🔍 Ontology Matching Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold", 
    0.0, 1.0, 1.0, 0.1,
    help="Minimum similarity score for ontology term matching (0.0 = very loose, 1.0 = exact match only)"
)

max_opt = st.sidebar.number_input("How many attempt to optimize RDF/SHACL data?", 0, 10, 1, help="Number of times the LLM should attempt to optimize RDF/SHACL")
max_corr = st.sidebar.number_input("How many attempt to correct RDF/SHACL data to pass the validation process?", 0, 10, 8, help="Number of times the LLM should attempt to fix RDF/SHACL after validation fails")


# Enhanced sidebar with detection quality info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 App Statistics")
    
    # Main counter
    st.metric("Total Opens", st.session_state.app_counter)
    
    # User's location with enhanced info
    user_country = st.session_state.get('user_country_info', {})
    if user_country.get('country') != 'Unknown':
        country_flag = get_country_flag(user_country.get('country_code', 'XX'))
        confidence = user_country.get('confidence', 0)
        source = user_country.get('source', 'unknown')
        
        st.markdown(f"**Your Location:** {country_flag} {user_country.get('country')}")
        
        if user_country.get('city') != 'Unknown':
            st.caption(f"📍 {user_country.get('city')}, {user_country.get('region')}")
        
    else:
        st.caption("🔴 Location detection failed")
    
    
    # Country statistics
    st.markdown("### 🌍 Countries")
    
    with st.spinner("Loading country statistics..."):
        country_stats = get_country_statistics()
    
    if country_stats:
        # Show top countries
        total_countries = len(country_stats)
        total_visits = sum(stats['count'] for stats in country_stats.values())
        
        st.caption(f"Visitors from {total_countries} countries")
        
        # Display top 10 countries
        for i, (country, stats) in enumerate(list(country_stats.items())[:10]):
            flag = get_country_flag(stats['country_code'])
            percentage = (stats['count'] / total_visits) * 100
            st.markdown(f"{flag} **{country}** - {stats['count']} visits ({percentage:.1f}%)")
        
        # Show "and X more" if there are more than 10 countries
        if total_countries > 10:
            st.caption(f"...and {total_countries - 10} more countries")
    else:
        st.caption("No country data available yet")

# Input Section - Store in session state
file_content, user_input = get_input_data()

# Update session state with current input
if user_input.strip():
    st.session_state.user_input_text = user_input
if file_content:
    st.session_state.uploaded_file_content = file_content

# Use session state values for processing
current_user_input = st.session_state.user_input_text or user_input
current_file_content = st.session_state.uploaded_file_content or file_content

if st.button("Generate RDF & SHACL"):
    # Check for input first
    if not current_user_input.strip():
        st.error("Please provide some input data to generate RDF and SHACL.")
        st.stop()
    if provider == "OpenAI" and not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif provider == "Anthropic" and not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif provider == "Ollama" and not endpoint:
        st.error("Please enter your Ollama API endpoint in the sidebar.")
    else:
        with st.spinner("Running Agent-Based Pipeline..."):
            model_info = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "api_key": api_key,
                "endpoint": endpoint
            }

            # 🔐 Validate API key
            if provider in ["OpenAI", "Anthropic", "Ollama"]:
                valid, message = validate_api_key(provider, api_key, endpoint)
                if not valid:
                    st.error(f"❌ {message}")
                    st.stop()
                    
            agent = SemanticPipelineAgent(model_info, max_opt, max_corr)

            # Show generation process in an expander to keep it organized
            with st.expander("🛠️ Generation & Optimization Process", expanded=False):
                st.subheader("Initial Generation")
                rdf_code, shacl_code = agent.generator.run(current_user_input)
                st.markdown("**Initial RDF Output:**")
                st.code(rdf_code, language="turtle")
                st.markdown("**Initial SHACL Output:**")
                st.code(shacl_code, language="turtle")

                # Optimization passes
                for i in range(max_opt):
                    st.markdown(f"### 🔄 Optimization Pass {i+1}")
                    explanation = agent.critic.run(rdf_code, shacl_code)
                    st.markdown(f"**Critique Explanation:**")
                    st.info(explanation)
                    rdf_code, shacl_code = agent.generator.run(
                        current_user_input, f"{rdf_code} {shacl_code} {explanation}"
                    )
                    st.markdown(f"**Optimized RDF (Pass {i+1}):**")
                    st.code(rdf_code, language="turtle")
                    st.markdown(f"**Optimized SHACL (Pass {i+1}):**")
                    st.code(shacl_code, language="turtle")

            # Validation and correction process
            st.subheader("🔍 Validation & Correction Process")

            # First, try basic syntax cleanup
            rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)

            # Check basic Turtle syntax first
            rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
            shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)

            if not rdf_syntax_valid:
                st.error(f"🚫 RDF Syntax Error: {rdf_syntax_error}")
            if not shacl_syntax_valid:
                st.error(f"🚫 SHACL Syntax Error: {shacl_syntax_error}")

            # Only proceed with SHACL validation if basic syntax is valid
            if rdf_syntax_valid and shacl_syntax_valid:
                valid, report = agent.validator.run(rdf_code, shacl_code)
            else:
                valid = False
                report = f"Syntax errors prevent SHACL validation:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"

            correction_attempt = 0
            correction_history = []
            previous_core_errors = []  # Track core error patterns instead of full reports
            consecutive_failures = 0  # Track consecutive correction failures

            while not valid and correction_attempt < max_corr:
                correction_attempt += 1
                
                # Extract core error for loop detection
                core_error = extract_core_error(report)
                
                # More sophisticated retry logic
                if not should_retry_correction(report, previous_core_errors, max_same_error=2):
                    st.warning(f"🔄 **Similar Error Pattern Detected** after {correction_attempt-1} attempts.")
                    st.info("💡 **Trying alternative correction approach...**")
                    
                    # Try one more time with additional context
                    if consecutive_failures < 2:  # Allow 2 alternative attempts
                        consecutive_failures += 1
                        st.info(f"🛠️ Alternative correction attempt #{consecutive_failures}")
                        
                        # Add more context to help the LLM understand the pattern
                        enhanced_report = f"""
                        REPEATED ERROR PATTERN DETECTED: {core_error}
                        
                        Previous attempts have failed to fix this issue. Please:
                        1. Focus on the ROOT CAUSE of this specific error type
                        2. Consider completely different approaches to modeling this data
                        3. Simplify the constraints if they are too restrictive
                        4. Check for fundamental modeling issues
                        
                        Original error report:
                        {report}
                        """
                        
                        try:
                            rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, enhanced_report)
                            rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)
                            
                            # Validate syntax first
                            rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
                            shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)
                            
                            if rdf_syntax_valid and shacl_syntax_valid:
                                valid, report = agent.validator.run(rdf_code, shacl_code)
                                if valid:
                                    st.success("✨ Alternative correction approach successful!")
                                    break
                            else:
                                valid = False
                                report = f"Syntax errors after correction:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"
                                
                        except Exception as e:
                            st.error(f"Error during alternative correction: {str(e)}")
                            break
                    else:
                        st.error("❌ Multiple correction approaches failed. Breaking correction loop.")
                        st.info("""
                        **💡 Suggestions for manual review:**
                        - The error pattern suggests a fundamental modeling issue
                        - Consider simplifying the SHACL constraints
                        - Review the RDF structure for compliance with expected patterns
                        - Check if the input data matches the intended semantic model
                        """)
                        break
                
                # Track the core error pattern
                previous_core_errors.append(core_error)
                
                # Determine error type for better user feedback
                error_type = 'syntax' if is_syntax_error(report) else 'validation'
                
                if error_type == 'syntax':
                    st.warning(f"🔤 Syntax Error Detected. Attempting correction #{correction_attempt}/{max_corr}")
                else:
                    st.warning(f"📋 SHACL Validation Failed. Attempting correction #{correction_attempt}/{max_corr}")
                
                # Store correction history with better categorization
                correction_history.append({
                    'attempt': correction_attempt,
                    'rdf': rdf_code,
                    'shacl': shacl_code,
                    'report': report,
                    'error_type': error_type,
                    'core_error': core_error
                })
                
                # Show validation report with better formatting
                with st.expander(f"📋 Validation Report (Attempt {correction_attempt})", expanded=False):
                    st.markdown(f"**Error Type:** {error_type.title()}")
                    st.markdown(f"**Core Error Pattern:** `{core_error}`")
                    st.code(report)
                    
                    # Show problematic sections for syntax errors
                    if error_type == 'syntax' and "line" in report:
                        try:
                            line_match = re.search(r'line (\d+)', report)
                            if line_match:
                                line_num = int(line_match.group(1))
                                lines = rdf_code.split('\n')
                                if line_num <= len(lines):
                                    st.markdown(f"**Problematic line {line_num}:**")
                                    # Show context around the problematic line
                                    start_line = max(0, line_num - 3)
                                    end_line = min(len(lines), line_num + 2)
                                    context = '\n'.join([f"{i+1:3d}: {lines[i]}" for i in range(start_line, end_line)])
                                    st.code(context)
                        except:
                            pass
                
                # Attempt correction with progress feedback
                try:
                    with st.spinner(f"Correcting {error_type} error (attempt {correction_attempt})..."):
                        rdf_code, shacl_code = agent.corrector.run(rdf_code, shacl_code, report)
                        
                        # Apply cleanup after correction
                        rdf_code, shacl_code = basic_syntax_cleanup(rdf_code, shacl_code)
                        
                        # Validate syntax first before SHACL validation
                        rdf_syntax_valid, rdf_syntax_error = validate_turtle_syntax(rdf_code)
                        shacl_syntax_valid, shacl_syntax_error = validate_turtle_syntax(shacl_code)
                        
                        if rdf_syntax_valid and shacl_syntax_valid:
                            valid, report = agent.validator.run(rdf_code, shacl_code)
                            if valid:
                                consecutive_failures = 0  # Reset failure counter on success
                        else:
                            valid = False
                            report = f"Syntax errors after correction:\nRDF: {rdf_syntax_error if not rdf_syntax_valid else 'OK'}\nSHACL: {shacl_syntax_error if not shacl_syntax_valid else 'OK'}"
                            
                except Exception as e:
                    st.error(f"Error during correction attempt {correction_attempt}: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        st.error("Multiple correction failures. Stopping correction process.")
                        break

            # Enhanced final validation status
            if valid:
                st.success("✅ Final Validation: PASSED")
                if correction_attempt > 0:
                    st.info(f"🎉 Successfully corrected after {correction_attempt} attempt(s)!")
                    
                    # Show what types of errors were fixed
                    error_types_fixed = list(set([c['error_type'] for c in correction_history]))
                    if len(error_types_fixed) == 1:
                        st.info(f"✨ Fixed {error_types_fixed[0]} errors")
                    else:
                        st.info(f"✨ Fixed multiple error types: {', '.join(error_types_fixed)}")
            else:
                st.error("❌ Final Validation: FAILED")
                if correction_attempt >= max_corr:
                    st.warning(f"⚠️ The generated RDF/SHACL did not pass validation after {max_corr} correction attempts.")
                    
                    # Provide specific guidance based on the final error pattern
                    final_core_error = extract_core_error(report)
                    if is_syntax_error(report):
                        st.info("""
                        **💡 Persistent Syntax Error Detected:**
                        
                        The issue appears to be malformed Turtle syntax. Common solutions:
                        - Check for unescaped quotes or special characters
                        - Ensure proper URI formatting (use <> for full URIs)
                        - Verify all statements end with proper punctuation (. ; ,)
                        - Check for missing prefixes or namespace declarations
                        """)
                    elif "constraint violation" in final_core_error:
                        st.info(f"""
                        **💡 Persistent SHACL Constraint Violation:**
                        
                        Core error pattern: `{final_core_error}`
                        
                        Possible solutions:
                        - Review if the SHACL constraints are too restrictive for your data
                        - Check if the RDF structure matches the expected semantic model  
                        - Consider simplifying complex constraints
                        - Verify that required properties are present in the RDF data
                        """)
                    else:
                        st.info(f"""
                        **💡 Persistent Validation Error:**
                        
                        Core error pattern: `{final_core_error}`
                        
                        This suggests a fundamental mismatch between the RDF data structure and SHACL constraints.
                        Consider manual review of both files.
                        """)
                else:
                    st.warning("⚠️ Validation failed due to syntax or other errors.")

            # visualize final RDF
            st.subheader("🌐 RDF Graph Visualization")
            html_content = visualize_rdf(rdf_code)
            if html_content:
                components.html(html_content, height=1000, width=1200, scrolling=True)
            else:
                st.error("Could not generate RDF visualization")
            
            # Ontology Mappings
            st.subheader("🔎 Suggested Ontology Terms")
            mappings = agent.ontology_mapper.run(current_user_input)
            st.markdown(mappings)

            # NEW: Ontology Matching Analysis
            st.subheader("🎯 Ontology Term Matching Analysis")
            st.markdown("**Analysis of RDF terms against local ontology files:**")

            with st.spinner("Analyzing ontology matches..."):
                try:
                    ontology_analysis = agent.ontology_matcher.run(rdf_code, similarity_threshold)
                    st.markdown(ontology_analysis)
                except Exception as e:
                    st.error(f"Error during ontology matching: {str(e)}")
                    st.info("💡 **Tip:** Make sure to place your ontology files (.ttl or .owl) in an 'ontologies/' directory in the same location as this application.")

            # Store original RDF/SHACL codes for comparison later
            original_rdf = rdf_code
            original_shacl = shacl_code

            # NEW: Apply Ontology Term Replacements
            st.subheader("🔄 Ontology Term Replacement")
            st.markdown("**Applying exact matches from ontology analysis:**")

            with st.spinner("Applying ontology term replacements..."):
                try:
                    # Apply replacements
                    replaced_rdf, replaced_shacl, replacement_report, replacement_validation = agent.apply_ontology_replacements(
                        rdf_code, shacl_code, similarity_threshold
                    )
                    
                    # Show replacement report
                    st.markdown(replacement_report)
                    
                    # If replacements were made, update the final codes and validate
                    if replacement_validation and replacement_validation["replacements_made"] > 0:
                        st.success(f"✅ Applied {replacement_validation['replacements_made']} exact term replacements!")
                        
                        # Update the final codes
                        rdf_code = replaced_rdf
                        shacl_code = replaced_shacl
                        
                        # Store replacement info in session state
                        st.session_state.replacement_count = replacement_validation['replacements_made']
                        st.session_state.replacement_status = f" (with {replacement_validation['replacements_made']} ontology term replacements)"
                        
                        # Show validation results for replaced version
                        if replacement_validation["conforms"]:
                            st.success("✅ Replaced RDF/SHACL passes validation!")
                            valid = True  # Update validation status
                        else:
                            st.warning("⚠️ Replaced RDF/SHACL has validation issues:")
                            with st.expander("View Replacement Validation Report"):
                                st.code(replacement_validation["report"])
                            
                            # Offer to use original or replaced version
                            use_replaced = st.radio(
                                "Which version would you like to use as final?",
                                ["Use replaced version (with ontology terms)", "Use original version (before replacement)"],
                                index=0
                            )
                            
                            if use_replaced == "Use original version (before replacement)":
                                # Restore original codes
                                rdf_code = original_rdf
                                shacl_code = original_shacl
                                st.session_state.replacement_count = 0
                                st.session_state.replacement_status = ""
                                st.info("Using original version as final output.")
                    else:
                        st.info("ℹ️ No exact matches found for replacement. Using original RDF/SHACL.")
                        st.session_state.replacement_count = 0
                        st.session_state.replacement_status = ""
                        
                except Exception as e:
                    st.error(f"Error during ontology replacement: {str(e)}")
                    st.info("Using original RDF/SHACL without replacements.")
                    st.session_state.replacement_count = 0
                    st.session_state.replacement_status = ""

            # Store final results in session state
            st.session_state.final_rdf_code = rdf_code
            st.session_state.final_shacl_code = shacl_code
            st.session_state.validation_status = valid
            st.session_state.validation_report = report
            st.session_state.rdf_generated = True

            # Show before/after comparison if replacements were made
            if st.session_state.replacement_count > 0:
                st.subheader("📊 Before/After Comparison")
                
                # Create tabs for comparison
                tab1, tab2 = st.tabs(["🔍 RDF Comparison", "🛡️ SHACL Comparison"])
                
                with tab1:
                    st.markdown("**RDF Changes:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("*Before Replacement:*")
                        st.code(original_rdf, language="turtle")
                    
                    with col2:
                        st.markdown("*After Replacement:*")
                        st.code(rdf_code, language="turtle")
                
                with tab2:
                    st.markdown("**SHACL Changes:**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("*Before Replacement:*")
                        st.code(original_shacl, language="turtle")
                    
                    with col2:
                        st.markdown("*After Replacement:*")
                        st.code(shacl_code, language="turtle")

# Display results if they exist in session state
if st.session_state.rdf_generated:
    # FINAL RESULTS SECTION - Make this very clear
    st.markdown("---")
    st.header("🎯 FINAL VALIDATED RESULTS")

    # Validation status box
    if st.session_state.validation_status:
        st.success(f"✅ **STATUS: VALIDATION PASSED{st.session_state.replacement_status}** - These are your final, validated RDF and SHACL files.")
    else:
        st.error(f"❌ **STATUS: VALIDATION FAILED{st.session_state.replacement_status}** - These files contain validation errors.")

    # Show replacement summary if applicable
    if st.session_state.replacement_count > 0:
        st.info(f"🔄 **Ontology Integration:** {st.session_state.replacement_count} terms were replaced with ontology equivalents")

    # Final RDF output with clear labeling
    st.subheader("📄 Final RDF Output")
    st.markdown("**This is your final RDF file" + (" (with ontology term replacements)" if st.session_state.replacement_status else "") + ":**")
    st.code(st.session_state.final_rdf_code, language="turtle")

    # Final SHACL output with clear labeling  
    st.subheader("🛡️ Final SHACL Output")
    st.markdown("**This is your final SHACL shapes file" + (" (with ontology term replacements)" if st.session_state.replacement_status else "") + ":**")
    st.code(st.session_state.final_shacl_code, language="turtle")

    # Final validation report
    st.subheader("📋 Final Validation Report")
    with st.expander("View Final Validation Details", expanded=st.session_state.validation_status is False):
        st.code(st.session_state.validation_report if st.session_state.validation_report else "All SHACL constraints satisfied.")

    # Download buttons with clear labeling
    st.subheader("⬇️ Download Final Files")
    col1, col2 = st.columns(2)
    with col1:
        suffix = "with_ontology_terms" if st.session_state.replacement_status else ("validated" if st.session_state.validation_status else "with_errors")
        download_filename_rdf = f"final_rdf_{suffix}.ttl"
        st.download_button(
            "📥 Download Final RDF", 
            st.session_state.final_rdf_code, 
            download_filename_rdf, 
            "text/turtle",
            help=f"Download the final RDF file {st.session_state.replacement_status}",
            key="download_rdf"  # Add unique key to prevent state conflicts
        )
    with col2:
        download_filename_shacl = f"final_shacl_{suffix}.ttl"
        st.download_button(
            "📥 Download Final SHACL", 
            st.session_state.final_shacl_code, 
            download_filename_shacl, 
            "text/turtle",
            help=f"Download the final SHACL file {st.session_state.replacement_status}",
            key="download_shacl"  # Add unique key to prevent state conflicts
        )

    # Visualize FINAL RDF
    st.subheader("🌐 Final RDF Graph Visualization")
    st.markdown("**Visualization of your final RDF data:**")
       
    html_content = visualize_rdf(st.session_state.final_rdf_code)
    if html_content:
        components.html(html_content, height=1000, width=1200, scrolling=True)

        # Add instructions for graph interaction
        st.markdown("""
        ### Graph Navigation Instructions:
        - **Zoom**: Use mouse wheel or pinch gesture
        - **Pan**: Click and drag empty space
        - **Move nodes**: Click and drag nodes to rearrange
        - **View details**: Hover over nodes or edges for full information
        - **Select multiple**: Hold Ctrl or Cmd while clicking nodes
        - **Reset view**: Double-click on empty space
        """)
    else:
        st.error("Could not generate RDF visualization")

    # Add a button to clear results and start over
    if st.button("🔄 Start New Generation"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key.startswith(('rdf_', 'final_', 'validation_', 'replacement_', 'user_input_', 'uploaded_file_')):
                del st.session_state[key]
        st.session_state.rdf_generated = False
        st.rerun()