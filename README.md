# Zero Agent

A deliberately quiet, black-and-white personal agent for turning fuzzy thoughts into a clear next move. The app is a native Android application with an offline-first conversation workspace, small task queue, quick prompts, and a custom interrupted-zero mark.

## App identity

- **Name:** Zero Agent
- **Package:** `com.zeroagent.app`
- **Minimum Android version:** Android 7.0 (API 24)
- **Logo:** a monochrome, interrupted `0` with an off-centre square — minimal, a little odd, and designed in-app as a vector asset.

## Run locally

Android Studio can open this folder directly. Or, with Java 17 installed:

```bash
./gradlew assembleDebug
```

The debug APK is written to:

```text
app/build/outputs/apk/debug/app-debug.apk
```

## APK builds on GitHub

The included workflow, [`.github/workflows/android-apk.yml`](.github/workflows/android-apk.yml), builds both a debug APK and an unsigned release APK whenever this branch is pushed, or when triggered manually from the **Actions** tab.

After it finishes, download either artifact from the workflow run:

- `zero-agent-debug-apk` — installable for testing.
- `zero-agent-release-unsigned-apk` — production build output that needs signing before store distribution.

The workflow uses Java 17 and Gradle 8.10.2. No API key or backend is required to run the app.

## Signing a production APK

Keep signing keys out of source control. For a publishable APK, configure a signing key locally or inject one through GitHub Actions secrets, then use `./gradlew assembleRelease`.
