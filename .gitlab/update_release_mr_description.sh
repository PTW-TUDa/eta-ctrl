#!/bin/bash

# This script updates the description of the open merge request from development to main.
# Remember to update the project version in pyproject.toml before running.
# Run from the project root directory.
#
# Prerequisites:
#   - A .env file in the project root with: GITLAB_ACCESS_TOKEN=<your_token>
#   - The token requires the 'api' scope (Personal Access Token or Project/Group Access Token)

## Project specific variables
CI_API_V4_URL=https://git.ptw.maschinenbau.tu-darmstadt.de/api/v4
CI_PROJECT_PATH=eta-fabrik/public/eta-ctrl
PROJECT_ID=587

# Get the commit SHA from the latest tag on main, fall back to the first commit on main if no tags exist
LAST_TAG=$(git describe --tags --abbrev=0 origin/main 2>/dev/null)
if [ -n "$LAST_TAG" ]; then
    LAST_VERSION_RELEASE_SHA=$(git rev-list -n 1 "$LAST_TAG")
else
    LAST_VERSION_RELEASE_SHA=$(git rev-list --max-parents=0 origin/main)
fi

# Escape slashes for correct url
CI_PROJECT_PATH_ESCAPED=$(echo "${CI_PROJECT_PATH}" | sed 's/\//\\\//g')

echo "Last version sha $LAST_VERSION_RELEASE_SHA"
# Source the .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Check if variable was set
if [ -z "$GITLAB_ACCESS_TOKEN" ]; then
    echo "GITLAB_ACCESS_TOKEN not found in .env file"
    exit 1
fi

# Helper function for api requests ($1: method, $2: endpoint, $3: optional JSON data)
api_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    curl -s -X $method ${data:+-d "$data"} \
         -H "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" \
         -H "Content-Type: application/json" \
         "${CI_API_V4_URL}/projects/${PROJECT_ID}${endpoint}"
}

# Verify token is valid before proceeding
AUTH_CHECK=$(curl -s -H "PRIVATE-TOKEN: ${GITLAB_ACCESS_TOKEN}" "${CI_API_V4_URL}/user")
if echo "$AUTH_CHECK" | grep -q '"error"'; then
    echo "ERROR: Authentication token is invalid or expired" >&2
    exit 1
fi

# Fetch the MR IID for the open MR from development to main
MR_IID_RESPONSE=$(api_request GET "/merge_requests?state=opened&source_branch=development&target_branch=main")

MR_IID=$(echo "$MR_IID_RESPONSE" | grep -o '"iid":[0-9]*' | head -n 1 | sed 's/"iid"://')
echo $MR_IID
if [ -z "$MR_IID" ]; then
  echo "No open merge request from development to main found."
  exit 0
fi

# Generate the Changelog from the Gitlab API
CHANGELOG_RESPONSE=$(api_request GET "/repository/changelog?from=${LAST_VERSION_RELEASE_SHA}&version=$(poetry version -s)")

if [[ "$CHANGELOG_RESPONSE" == *"Failed to generate the changelog"* ]]; then
  echo "Failed to generate changelog, has the version been updated?"
  exit 0
fi
# Extract the 'notes' field from the API response and strip 'Closes issue' suffix
CHANGELOG=$(echo "$CHANGELOG_RESPONSE" | grep -o '"notes":"[^"]*' | sed 's/"notes":"//' \
                            | sed -E 's/Closes ((#[0-9]+(, )?)+)/\1/g' )
echo $CHANGELOG
# Update the MR description
api_request PUT "/merge_requests/${MR_IID}" "{ \"description\": \"${CHANGELOG}\" }" > /dev/null

echo "Updated description for merge request ${MR_IID}."
