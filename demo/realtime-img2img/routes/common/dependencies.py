"""
Shared dependency injection functions for all route modules.

This module consolidates dependency injection functions that were previously
duplicated across multiple route files to eliminate code duplication and
centralize dependency management.
"""


def get_app_instance():
    """Dependency to get the app instance - will be injected during router registration"""
    # This will be overridden when the router is included in main.py
    pass


def get_pipeline_class():
    """Dependency to get the Pipeline class - will be injected during router registration"""
    # This will be overridden when the router is included in main.py
    pass


def get_default_settings():
    """Dependency to get the DEFAULT_SETTINGS - will be injected during router registration"""
    # This will be overridden when the router is included in main.py
    pass


def get_available_controlnets():
    """Dependency to get the AVAILABLE_CONTROLNETS - will be injected during router registration"""
    # This will be overridden when the router is included in main.py
    pass
