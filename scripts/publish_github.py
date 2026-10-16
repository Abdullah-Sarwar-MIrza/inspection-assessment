"""Publish this prepared submission using an existing Git credential login.

Never prints credentials. Does not initiate interactive login or overwrite an
existing remote repository. Use --check to inspect only the authenticated user.
"""
import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request

OWNER = "Abdullah-Sarwar-MIrza"
REPO = "inspection-assessment"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    result = subprocess.run(["git", "-c", "safe.directory=D:/Job Assessment", "-c", "safe.directory=D:/Job Assessment", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, env=env)
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = credentials.get("password")
    if not token:
        raise SystemExit("No existing GitHub authentication is available. Connect GitHub or authenticate Git Credential Manager, then rerun this script.")

    def request(path, method="GET", body=None):
        req = urllib.request.Request("https://api.github.com" + path, method=method,
             data=json.dumps(body).encode() if body is not None else None,
             headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                      "User-Agent": "casting-inspection-assessment", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    user = request("/user")["login"]
    print("Authenticated GitHub account:", user)
    if user.lower() != OWNER.lower():
        raise SystemExit("Authenticated account differs from the requested owner; nothing published.")
    if args.check:
        return
    try:
        request(f"/repos/{OWNER}/{REPO}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise SystemExit(f"GitHub lookup failed: HTTP {exc.code}") from None
    else:
        raise SystemExit("Repository already exists. Inspect it before choosing how to publish; nothing overwritten.")
    created = request("/user/repos", "POST", {"name": REPO, "private": False,
                      "description": "Casting defect inspection: transfer learning, evaluation, FastAPI and Docker"})
    subprocess.run(["git", "-c", "safe.directory=D:/Job Assessment", "-c", "safe.directory=D:/Job Assessment", "remote", "add", "origin", created["clone_url"]], check=True)
    subprocess.run(["git", "-c", "safe.directory=D:/Job Assessment", "-c", "safe.directory=D:/Job Assessment", "push", "-u", "origin", "main"], check=True, env=env)
    print("Published:", created["html_url"])


if __name__ == "__main__":
    main()
