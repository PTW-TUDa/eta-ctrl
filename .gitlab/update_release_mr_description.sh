#!/bin/bash

# This script updates the description of the open merge request from development to main
# Remember to update the project version in pyproject.toml

# CI_DEPLOY_TOKEN needs to be supplied in .env

## Project specific variables
CI_API_V4_URL=https://git.ptw.maschinenbau.tu-darmstadt.de/api/v4
CI_PROJECT_PATH=eta-fabrik/public/eta-ctrl
PROJECT_ID=120

# Get the commit SHA from the latest tag on main
LAST_VERSION_RELEASE_SHA=$(git rev-list -n 1 $(git describe --tags --abbrev=0 origin/main))

# Escape slashes for correct url
CI_PROJECT_PATH_ESCAPED=$(echo "${CI_PROJECT_PATH}" | sed 's/\//\\\//g')


# Source the .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Check if variable was set
if [ -z "$CI_DEPLOY_TOKEN" ]; then
    echo "CI_DEPLOY_TOKEN not found in .env file"
    exit 1
fi

# Helper function for api requests
api_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    curl -s -X $method ${data:+-d "$data"} \
         -H "PRIVATE-TOKEN: ${CI_DEPLOY_TOKEN}" \
         -H "Content-Type: application/json" \
         "${CI_API_V4_URL}/projects/${PROJECT_ID}${endpoint}"
}

# Fetch the MR IID for the open MR from development to main
MR_IID=$((api_request GET "/merge_requests?state=opened&source_branch=development&target_branch=main") \
        | grep -o '"iid":[0-9]*' | head -n 1 | sed 's/"iid"://')
echo $MR_IID
if [ -z "$MR_IID" ]; then
  echo "No open merge request from development to main found."
  exit 0
fi

# Generate the Changelog from the Gitlab API
CHANGELOG_RESPONSE=$(api_request GET "/repository/changelog?from=${LAST_VERSION_RELEASE_SHA}&version=$(poetry version -s)")

if [[ "$response" == *"Failed to generate the changelog"* ]]; then
  echo "Failed to generate changelog, has the version been updated?"
  exit 0
fi
echo $CHANGELOG_RESPONSE
# Extract the 'notes' field, parse merge request references and issue references
CHANGELOG=$(echo "$CHANGELOG_RESPONSE" | grep -o '"notes":"[^"]*' | sed 's/"notes":"//' \
                            | sed -E 's/Closes ((#[0-9]+(, )?)+)/\1/g' \
                            | sed -E "s/${CI_PROJECT_PATH_ESCAPED}!([0-9]+)/!\1/g" )
echo $CHANGELOG
# Update the MR description
api_request PUT "/merge_requests/${MR_IID}" "{ \"description\": \"${CHANGELOG}\" }" > /dev/null

echo "Updated description for merge request ${MR_IID}."
