"""Who is looking at the application, and which graphs they may see.

Access here is enforced by the application, not by the database. Streamlit in
Snowflake runs with owner's rights: every viewer executes with the app owner's
role, so the warehouse will happily return any row the owner can read, and it is
this module's predicates that decide what a viewer is shown. A query added
elsewhere that does not pass through :func:`visible_graph_predicate` sees
everything. Enforcing this in the database instead would take a row access
policy over ``KG_GRAPH.OWNER``, which requires an account administrator to grant
the global READ SESSION privilege to the app owner role.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import streamlit as st

# Snowflake resolves the viewer of an owner's-rights app through ``st.user``.
# ``CURRENT_USER()`` deliberately is not used: inside Streamlit in Snowflake it
# returns the account that owns the application object, so every viewer would be
# read as the owner and would inherit the owner's graphs.
_VIEWER_NAME_FIELDS = ("user_name", "username", "login_name")


@dataclass(frozen=True)
class Viewer:
    """One authenticated person looking at the application."""

    user_name: str
    email: str = ""
    roles: frozenset[str] = frozenset()

    @property
    def is_identified(self) -> bool:
        """Report whether Snowflake named this viewer.

        An unidentified viewer is not treated as an owner of anything, so a
        deployment that cannot resolve identity shows private graphs to nobody
        rather than showing them to everybody.
        """

        return bool(self.user_name)


def current_viewer(roles: Sequence[str] = ()) -> Viewer:
    """Read the signed-in viewer from the Streamlit session."""

    identity = _user_mapping()
    name = ""
    for field in _VIEWER_NAME_FIELDS:
        value = identity.get(field)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    email = identity.get("email")
    return Viewer(
        user_name=name.upper(),
        email=email.strip() if isinstance(email, str) else "",
        roles=frozenset(role.strip().upper() for role in roles if role.strip()),
    )


def _user_mapping() -> Mapping[str, object]:
    """Return ``st.user`` as a plain mapping on every supported Streamlit.

    Outside Streamlit in Snowflake the object exists but carries no Snowflake
    identity, and older builds raise rather than return an empty mapping.
    """

    user = getattr(st, "user", None)
    if user is None:
        return {}
    try:
        if hasattr(user, "to_dict"):
            return dict(user.to_dict())
        return dict(user)
    except Exception:
        return {}


def visible_graph_predicate(viewer: Viewer, *, alias: str = "G") -> tuple[str, list[object]]:
    """Build the SQL a graph must satisfy to be shown to this viewer.

    A graph is visible when the viewer owns it, when it has been shared with
    them by name or through one of their roles, or when nobody owns it. The last
    case covers graphs written before ownership was recorded: hiding them would
    make existing work vanish, so they stay visible and the application labels
    them as belonging to no one.

    Seeing a graph is not permission to change it — see
    :func:`administrable_graph_predicate`.
    """

    if not viewer.is_identified:
        # Identity could not be resolved, so only unowned graphs are shown. The
        # alternative — treating an unknown viewer as everyone — would hand a
        # misconfigured deployment every private graph in the account.
        return f"{alias}.OWNER IS NULL", []
    roles = sorted(viewer.roles)
    # A share names a principal in one of Snowflake's two namespaces, and the
    # same name can exist in both: a service user and its functional role
    # usually share one. Matching on the name alone would let a share to a role
    # admit a user who was never granted it.
    role_arm = ""
    parameters: list[object] = [viewer.user_name, viewer.user_name]
    if roles:
        placeholders = ", ".join("?" for _ in roles)
        role_arm = f" OR (UPPER(A.GRANTEE_TYPE) = 'ROLE' AND UPPER(A.GRANTEE) IN ({placeholders}))"
        parameters.extend(roles)
    shared = (
        f"EXISTS (SELECT 1 FROM KG_GRAPH_ACL A WHERE A.GRAPH_ID = {alias}.GRAPH_ID "
        f"AND ((UPPER(A.GRANTEE_TYPE) = 'USER' AND UPPER(A.GRANTEE) = ?){role_arm}))"
    )
    return f"({alias}.OWNER IS NULL OR UPPER({alias}.OWNER) = ? OR {shared})", parameters


def administrable_graph_predicate(viewer: Viewer, *, alias: str = "G") -> tuple[str, list[object]]:
    """Build the SQL a graph must satisfy for this viewer to change it.

    Ownership, never a share. Cancelling a run, forgetting it, renaming a graph
    or submitting more work into one are destructive: a share exists so a
    colleague can read a graph, and letting it also authorise deleting that
    graph would make every share an irrevocable act of trust.
    """

    if not viewer.is_identified:
        return f"{alias}.OWNER IS NULL", []
    return f"({alias}.OWNER IS NULL OR UPPER({alias}.OWNER) = ?)", [viewer.user_name]


UPLOAD_NAMESPACE = "app"


def viewer_stage_prefix(viewer: Viewer, job_id: str) -> str:
    """Return the stage folder one viewer's uploads belong in.

    The application lists a Snowflake stage with the owner's privileges, so a
    prefix a viewer could choose freely would let them enumerate and ingest
    documents another viewer uploaded. Deriving it from identity keeps each
    person's corpus in a folder only their own runs address.
    """

    who = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in (viewer.user_name or "shared")
    ).strip("_")
    # A name of dots alone would form a relative path segment and address a
    # folder above this viewer's own.
    if not who or set(who) <= {"."}:
        who = "shared"
    return f"{UPLOAD_NAMESPACE}/{who}/{job_id}"


def may_read_stage_path(viewer: Viewer, path: str) -> bool:
    """Report whether a viewer may read one path inside a Snowflake stage.

    The application lists a stage with the app owner's privileges, so the check
    is on the path finally addressed rather than on any one form field: a stage
    identifier may itself carry a path, and the two are concatenated before the
    LIST runs.

    Refused: the stage root and the upload namespace itself, both of which
    contain other viewers' folders, and any other viewer's folder directly.
    Allowed: this viewer's own uploads, and anything outside the namespace the
    application uploads into — those are stages an operator curates, where
    Snowflake's own grants are the authority.
    """

    segments = [segment for segment in path.strip().strip("/").split("/") if segment]
    if not segments:
        # The root lists every folder beneath it, including other viewers'.
        return False
    if segments[0] != UPLOAD_NAMESPACE:
        return True
    mine = viewer_stage_prefix(viewer, "").rstrip("/").split("/")
    return segments[: len(mine)] == mine
