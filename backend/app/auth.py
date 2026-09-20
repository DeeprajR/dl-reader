"""The optional password for a deployment that anyone can reach.

With APP_PASSWORD set, every request (the API, the images and the frontend itself) needs that
password. Without it the app is open, as on a local run.

The visitor types the password once, on the app's own sign-in page, and the browser then keeps
a cookie. There are no user names and no accounts: it is one shared password.
"""

import asyncio
import base64
import hashlib
import hmac
import os
import secrets

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

LOGIN_PATH = "/api/login"  # under /api, so the Vite dev server passes it on to the backend too
COOKIE = "licence_reader_session"
COOKIE_DAYS = 7
# Every wrong password waits this long before it is answered, which makes guessing slow.
WRONG_PASSWORD_DELAY = 1.0

router = APIRouter()


def app_password() -> str | None:
    """The configured password, or None when the app is open."""
    return os.getenv("APP_PASSWORD") or None


def session_token(password: str) -> str:
    """The cookie's value: a fingerprint of the password, never the password itself.

    Nothing is stored on the server, so a sign-in survives restarts, and changing the password
    signs everyone out.
    """
    return hmac.new(password.encode(), b"licence-reader-session", hashlib.sha256).hexdigest()


def _same(a: str, b: str) -> bool:
    """Compare in the same time whether or not the first characters match."""
    return secrets.compare_digest(a.encode(), b.encode())


def _basic_password(header: str | None) -> str | None:
    """The password from an `Authorization: Basic ...` header, which scripts and curl can send."""
    scheme, _, encoded = (header or "").partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8").partition(":")[2]
    except ValueError:  # not base64, or not text
        return None


def is_signed_in(request: Request, password: str) -> bool:
    """True when the request carries the sign-in cookie, or the password itself in a Basic header."""
    if _same(request.cookies.get(COOKIE, ""), session_token(password)):
        return True
    given = _basic_password(request.headers.get("Authorization"))
    return given is not None and _same(given, password)


async def password_gate(request: Request, call_next):
    """Let a request through only when it is signed in (registered as middleware in main.py)."""
    password = app_password()
    if not password or request.url.path == LOGIN_PATH or is_signed_in(request, password):
        return await call_next(request)
    # The frontend sends the visitor to the sign-in page when an API call answers 401.
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Password required."}, status_code=401)
    return RedirectResponse(LOGIN_PATH, status_code=303)


class LoginRequest(BaseModel):
    """The body of POST /api/login."""

    password: str


@router.get(LOGIN_PATH, response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    """The sign-in page: one password box, no user name."""
    return HTMLResponse(LOGIN_PAGE if app_password() else '<meta http-equiv="refresh" content="0; url=/">')


@router.post(LOGIN_PATH, include_in_schema=False)
async def login(body: LoginRequest, request: Request):
    """Check the password and hand the browser its sign-in cookie."""
    password = app_password()
    if password and not _same(body.password, password):
        await asyncio.sleep(WRONG_PASSWORD_DELAY)
        return JSONResponse({"error": "Wrong password."}, status_code=401)
    response = JSONResponse({"ok": True})
    if password:
        # Cloud Run and other hosts end HTTPS in front of the app and say so in this header.
        https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        response.set_cookie(
            COOKIE,
            session_token(password),
            max_age=COOKIE_DAYS * 24 * 3600,
            httponly=True,  # page scripts cannot read it
            samesite="strict",  # other sites cannot send it
            secure=https,
        )
    return response


LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Driving Licence Reader</title>
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f8fafc;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #0f172a; }
  form { width: min(22rem, calc(100vw - 2rem)); box-sizing: border-box; padding: 2rem; background: #fff;
         border: 1px solid #e2e8f0; border-radius: .75rem; box-shadow: 0 1px 3px rgb(0 0 0 / .08); }
  h1 { margin: 0 0 .25rem; font-size: 1.125rem; }
  p { margin: 0 0 1.25rem; font-size: .875rem; color: #475569; }
  label { display: block; font-size: .875rem; font-weight: 500; }
  input { width: 100%; box-sizing: border-box; margin-top: .25rem; padding: .5rem .75rem; font: inherit;
          border: 1px solid #cbd5e1; border-radius: .375rem; }
  input:focus { outline: 2px solid #bfdbfe; border-color: #3b82f6; }
  button { width: 100%; margin-top: 1rem; padding: .55rem; font: inherit; font-weight: 500; color: #fff;
           background: #1d4ed8; border: 0; border-radius: .375rem; cursor: pointer; }
  button:disabled { background: #cbd5e1; cursor: not-allowed; }
  #error { min-height: 1.25rem; margin: .75rem 0 0; color: #b91c1c; }
</style>
</head>
<body>
<form id="form">
  <h1>AI Driving Licence Reader</h1>
  <p>Enter the password to continue.</p>
  <label for="password">Password</label>
  <input id="password" type="password" autocomplete="current-password" required autofocus>
  <button id="submit" type="submit">Continue</button>
  <p id="error" role="alert"></p>
</form>
<script>
  const form = document.getElementById('form'), error = document.getElementById('error')
  const button = document.getElementById('submit'), input = document.getElementById('password')
  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    button.disabled = true
    error.textContent = ''
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: input.value }),
      })
      if (response.ok) return window.location.replace('/')
      error.textContent = response.status === 401 ? 'Wrong password.' : 'Could not sign in. Try again.'
    } catch {
      error.textContent = 'Could not reach the server.'
    }
    button.disabled = false
    input.select()
  })
</script>
</body>
</html>
"""
