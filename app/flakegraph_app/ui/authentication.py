"""Establish who the viewer is, by whichever mechanism the deployment provides.

There are two, and a deployment uses one of them.

*The application signs the viewer in.* Streamlit's OpenID Connect support is
inert on its own: configuring ``[auth]`` in secrets grants the application the
*ability* to sign a user in, and nothing more. Until something calls
:func:`streamlit.login`, every visitor is anonymous and every page is served. A
deployment can therefore hold a complete, correct identity configuration and
still be open to whoever can reach it, which is a worse failure than having no
configuration at all - it looks protected. This module is that call.

*A gate in front signs the viewer in.* Where a deployment puts an authenticating
proxy ahead of every routed host, the application is never reached by an
anonymous request, and running a second sign-in behind the first is not extra
safety - it is a second redirect flow, with a second callback URL, that the
viewer has to complete after already having proved who they are. So when the
operator names a header the gate populates, this module reads the identity from
there and does not start a flow of its own.

Both are provider-neutral. The application asks Streamlit, or reads a header;
neither path knows about any particular vendor.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, cast

import streamlit as st

AUTH_SECTION = "auth"
# Streamlit requires these before st.login() can work. A partial section is a
# misconfiguration worth reporting rather than quietly treating as "no auth".
REQUIRED_AUTH_KEYS = ("redirect_uri", "cookie_secret", "client_id", "client_secret")

# The header an authenticating proxy sets once it has admitted the request.
# Naming it is how an operator states that a gate is in front; leaving it empty
# means the application signs viewers in itself.
FORWARDED_IDENTITY_ENVIRONMENT = "FLAKEGRAPH_APP_FORWARDED_IDENTITY_HEADER"
# Where the gate ends a session, when it offers somewhere to do that.
SIGN_OUT_URL_ENVIRONMENT = "FLAKEGRAPH_APP_SIGN_OUT_URL"


class AuthenticationNotConfigured(RuntimeError):
    """Raised when identity is required but the deployment cannot perform it."""


class _User(Protocol):
    """The part of ``st.user`` this module depends on."""

    is_logged_in: bool


def _environment_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def identity_is_delegated() -> bool:
    """Report whether a gate in front establishes identity for this deployment."""

    return bool(_environment_value(FORWARDED_IDENTITY_ENVIRONMENT))


def forwarded_identity() -> str | None:
    """Return the identity the gate admitted this request under, if any.

    The header cannot be forged by the viewer: the gate answers every request
    before the application sees it, and the proxy replaces these headers with
    what the gate returned. A request that arrives without one did not come
    through the gate at all.
    """

    header = _environment_value(FORWARDED_IDENTITY_ENVIRONMENT)
    if not header:
        return None
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(header)
    except Exception:
        return None
    if value is None:
        return None
    return str(value).strip() or None


def identity_is_configured() -> bool:
    """Report whether this deployment can establish who the viewer is.

    Reading secrets can raise when no secrets file exists at all, which is the
    ordinary case for a local checkout, so absence is not an error here.
    """

    if identity_is_delegated():
        return True
    try:
        section = st.secrets.get(AUTH_SECTION)
    except Exception:
        return False
    if not isinstance(section, dict):
        return bool(section)
    return any(str(section.get(key) or "").strip() for key in REQUIRED_AUTH_KEYS)


def _missing_auth_keys() -> tuple[str, ...]:
    """Return the required ``[auth]`` keys this deployment has not supplied."""

    try:
        section = st.secrets.get(AUTH_SECTION) or {}
    except Exception:
        return REQUIRED_AUTH_KEYS
    if not isinstance(section, dict):
        return ()
    return tuple(key for key in REQUIRED_AUTH_KEYS if not str(section.get(key) or "").strip())


def viewer_is_signed_in() -> bool:
    """Report whether this request carries an established identity."""

    if identity_is_delegated():
        return forwarded_identity() is not None
    user = cast("_User | None", getattr(st, "user", None))
    return bool(user is not None and getattr(user, "is_logged_in", False))


def require_sign_in(*, required: bool) -> bool:
    """Return whether the page may render, prompting for sign-in when it may not.

    ``required`` is the operator's decision, not this module's. An unconfigured
    deployment that demands identity would lock out a local checkout for no
    benefit; a configured one that does not demand it would leave the very
    protection it configured switched off. So the two are separate, and the
    combination that matters - required, but not established - refuses to render
    rather than falling open.
    """

    if identity_is_delegated():
        if forwarded_identity() is not None:
            return True
        # The gate is declared to be in front, and this request did not come
        # through it. `required` does not apply here: it asks whether identity
        # is demanded, and naming a gate has already answered that. Treating a
        # missing header as "identity is optional" would serve every request
        # that reaches this process by any route other than the gate - another
        # pod in the namespace, a port-forward, a Service exposed by mistake -
        # which is precisely the case the header exists to exclude.
        #
        # Sending the viewer to sign in would not help either: whatever they
        # did, they arrived by a route the gate does not cover, and a flow
        # cannot fix a route.
        raise AuthenticationNotConfigured(
            "This deployment is served behind a sign-in gate, but this request "
            f"carried no {_environment_value(FORWARDED_IDENTITY_ENVIRONMENT)} "
            "header - it did not arrive through the gate."
        )

    configured = identity_is_configured()
    if required and not configured:
        missing = ", ".join(_missing_auth_keys()) or "the [auth] section"
        raise AuthenticationNotConfigured(
            "Sign-in is required for this deployment but identity is not "
            f"configured: {missing} missing from Streamlit secrets."
        )
    if not configured or viewer_is_signed_in():
        return True
    if not required:
        # Identity is available but optional: offer it without blocking, so a
        # deployment can adopt sign-in before it enforces it.
        _render_sign_in(blocking=False)
        return True
    _render_sign_in(blocking=True)
    return False


def _render_sign_in(*, blocking: bool) -> None:
    """Show the sign-in affordance, occupying the page only when it must."""

    target: Any = st if blocking else st.sidebar
    if blocking:
        target.title("FlakeGraph")
        target.write("Sign in to continue.")
    if target.button("Sign in", type="primary" if blocking else "secondary"):
        st.login()


def render_sign_out() -> None:
    """Offer sign-out wherever this is called, for a signed-in viewer only."""

    if identity_is_delegated():
        identity = forwarded_identity()
        if identity is None:
            return
        st.sidebar.caption(f"Signed in as {identity}")
        # The gate owns the session, so it is the only thing that can end it.
        # Where the operator has not said where that is, showing a button that
        # cannot sign anyone out would be worse than showing none.
        sign_out_url = _environment_value(SIGN_OUT_URL_ENVIRONMENT)
        if sign_out_url:
            st.sidebar.link_button("Sign out", sign_out_url)
        return
    if not viewer_is_signed_in():
        return
    if st.sidebar.button("Sign out"):
        st.logout()
