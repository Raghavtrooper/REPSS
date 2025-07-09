import streamlit as st
import time


def clear_chat_callback():
    """
    Clears the chat history in Streamlit's session state.
    """
    st.session_state.messages = []
    st.rerun()

def display_profile_details(profile_metadata: dict, use_inner_expanders: bool = True):
    """
    Displays the details of an employee profile in a neatly formatted way.
    Aims to prevent nested expander issues by controlling inner expanders.
    Updated to display new fields: location, objective, qualifications_summary,
    experience_summary, and has_photo. Removes old title/department fields.

    Args:
        profile_metadata (dict): The dictionary containing the profile data.
        use_inner_expanders (bool): If True, uses st.expander for sections like
                                     Skills, Experience, Education. If False,
                                     displays content directly (useful when
                                     this function is already nested in an expander).
    """
    st.subheader(profile_metadata.get('name', 'N/A'))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Email:** {profile_metadata.get('email_id', 'N/A')}")
        st.write(f"**Phone:** {profile_metadata.get('phone_number', 'N/A')}")
    with col2:
        st.write(f"**Location:** {profile_metadata.get('location', 'N/A')}")
        st.write(f"**Has Photo:** {'Yes' if profile_metadata.get('has_photo') else 'No'}") # Display has_photo


    # Helper function to display content, with or without expander
    def render_section(title, content):
        if not content:
            return
        # Ensure content is a string for display; if it's a list (like skills), join it.
        display_content = ", ".join(content) if isinstance(content, list) else str(content)

        if use_inner_expanders:
            with st.expander(title):
                st.write(display_content)
        else:
            st.markdown(f"**{title}:**")
            st.write(display_content)

    render_section("Objective", profile_metadata.get('objective')) # New field
    render_section("Skills", profile_metadata.get('skills'))
    render_section("Qualifications Summary", profile_metadata.get('qualifications_summary')) # Renamed
    render_section("Experience Summary", profile_metadata.get('experience_summary')) # Renamed

    # Display duplicate status if available
    if profile_metadata.get('_duplicate_count', 1) > 1:
        status = "Master Record" if profile_metadata.get('_is_master_record') else "Associated Resume"
        st.info(f"**Duplicate Status:** This person has {profile_metadata['_duplicate_count']} associated resumes. "
                f"Group ID: `{profile_metadata.get('_duplicate_group_id', 'N/A')[:8]}` ({status}).")
        if profile_metadata.get('_associated_original_filenames'):
            st.markdown(f"**Associated Original Files:** {', '.join(profile_metadata['_associated_original_filenames'])}")

    # Add original filename and download link if available
    if 'original_file_minio_url' in profile_metadata:
        minio_url = profile_metadata['original_file_minio_url']
        original_filename = profile_metadata.get('original_filename', 'Resume')
        st.markdown(f"[[Download Original {original_filename}]]({minio_url})")

    st.markdown("---") # Separator between profiles