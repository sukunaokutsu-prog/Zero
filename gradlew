#!/usr/bin/env sh
# Lightweight Gradle launcher. It keeps this repository usable without committing a wrapper JAR.
set -eu
GRADLE_VERSION="8.10.2"
CACHE_DIR="${GRADLE_USER_HOME:-$HOME/.gradle}/wrapper/dists/gradle-${GRADLE_VERSION}-bin"
GRADLE_HOME="$CACHE_DIR/gradle-${GRADLE_VERSION}"
if [ ! -x "$GRADLE_HOME/bin/gradle" ]; then
  mkdir -p "$CACHE_DIR"
  ARCHIVE="$CACHE_DIR/gradle-${GRADLE_VERSION}-bin.zip"
  if [ ! -f "$ARCHIVE" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --retry 3 "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -o "$ARCHIVE"
    else
      wget -O "$ARCHIVE" "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip"
    fi
  fi
  unzip -q -o "$ARCHIVE" -d "$CACHE_DIR"
fi
exec "$GRADLE_HOME/bin/gradle" "$@"
