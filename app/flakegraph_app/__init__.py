"""Backend-neutral services used by the FlakeGraph Streamlit application.

This package intentionally supports Python 3.11, the current Snowflake
Streamlit runtime baseline. It does not import the Python 3.14 processing
package; local and Kubernetes execution cross the stable CLI boundary while
Snowflake execution crosses the persisted table contract.
"""

from flakegraph_app.models import RuntimeMode

__all__ = ["RuntimeMode"]
