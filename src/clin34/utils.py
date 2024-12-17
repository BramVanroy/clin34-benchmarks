import os
import random
import re

import numpy
import torch.backends.cudnn


def is_jinja_template(text: str) -> bool:
    # Patterns for Jinja delimiters: {{ variable }}, {% control %}, and {# comment #}
    jinja_patterns = [
        r"\{\{.*?\}\}",  # Matches {{ ... }}
        r"\{%.*?%\}",  # Matches {% ... %}
        r"\{#.*?#\}",  # Matches {# ... #}
    ]

    # Check if any pattern matches
    for pattern in jinja_patterns:
        if re.search(pattern, text):
            return True

    return False
