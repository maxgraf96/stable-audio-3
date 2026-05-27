#!/usr/bin/env bash
# notarize.sh — submit a single signed bundle to Apple's notary service,
# wait for the ticket, then staple it to the bundle so launches work
# offline forever.
#
# Usage:
#   notarize.sh <path-to-bundle>
#
# Credentials are read from plugin/.env (gitignored). Required keys:
#   APPLE_ID="you@example.com"
#   APPLE_TEAM_ID="ABCDE12345"
#   APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
# See plugin/.env.example for the template + where to generate each value.
set -euo pipefail

bundle="$1"

# Source .env next to plugin/ so the script works whether invoked from
# the repo root or anywhere else.
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$HERE/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

for v in APPLE_ID APPLE_TEAM_ID APPLE_APP_PASSWORD; do
    if [[ -z "${!v:-}" ]] || [[ "${!v}" == "FILL_ME_IN"* ]] \
                          || [[ "${!v}" == "xxxx-"* ]]      \
                          || [[ "${!v}" == "XXXXXXXXXX" ]]; then
        echo "error: $v not set (or still a placeholder) in $ENV_FILE" >&2
        exit 1
    fi
done

if [[ ! -d "$bundle" ]]; then
    echo "error: bundle not found: $bundle" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
zip_path="$tmp/$(basename "$bundle").zip"

# `ditto -c -k --keepParent` produces the exact archive layout Apple's
# notary expects (preserves resource forks + extended attributes).
echo "==> archiving for notarization: $bundle"
ditto -c -k --keepParent "$bundle" "$zip_path"

echo "==> submitting (Apple ID: $APPLE_ID, team: $APPLE_TEAM_ID) — blocks until Apple replies"
xcrun notarytool submit "$zip_path" \
    --apple-id  "$APPLE_ID"  \
    --team-id   "$APPLE_TEAM_ID" \
    --password  "$APPLE_APP_PASSWORD" \
    --wait

echo "==> stapling ticket"
xcrun stapler staple "$bundle"

echo "==> verifying"
spctl -a -vv --type install "$bundle" 2>&1 || spctl -a -vv "$bundle"
echo "notarize: ok — $bundle"
