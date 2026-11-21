**Live app:** https://interactive-ai-voice-agent.onrender.com/

**Project demo:** https://drive.google.com/file/d/1n05drrGfnjZDWhRidq2_erRqIixkiJLu/view?usp=drivesdk

# Voice AI Chat Agent

A hands-free voice assistant that runs in the browser. It is built with **FastAPI**, **AssemblyAI** (speech-to-text), **Google Gemini** (AI replies), **Murf** (text-to-speech) and **SerpAPI** (web search).

## Features

- **Hands-free conversation:** click *Start conversation* and speak. The assistant replies after 3 seconds of silence.
- **Interruptions:** the user can speak while the assistant is talking, and it stops to listen.
- **Chat view:** the conversation is shown as chat messages in real time.
- **Personas:** Friendly Assistant, Interview Coach, Storyteller, Debate Partner, Web Researcher and Live Translator. Each has its own voice and introduces its role on the first question.
- **Emotion-aware responses:** the assistant detects the user's mood and adjusts its tone and voice.
- **Web Researcher:** searches Google live for news and current events, and lists its sources.
- **Live Translator:** translates speech in English, Spanish, French, German, Italian or Portuguese into Spanish, French, German, Hindi or English.
- **Date and time awareness:** the assistant knows the current date and time in the user's time zone.
- **Clear error messages:** short messages, with a *View issue* link for technical details.

## Get the API Keys

| Service | Used for | Get a key |
| --- | --- | --- |
| AssemblyAI | Speech-to-text | https://www.assemblyai.com/app/api-keys |
| Google Gemini | AI replies | https://aistudio.google.com/app/apikey |
| Murf | Text-to-speech | https://murf.ai/api/dashboard |
| SerpAPI | Web search (Web Researcher only, optional) | https://serpapi.com/manage-api-key |

## How to Run

The code is in the `Project-MURF_AI-30DaysofVoiceAgents-main/` folder, so run these commands from there.

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add the API keys to the `.env` file:
   ```env
   ASSEMBLYAI_API_KEY=your_key
   GEMINI_API_KEY=your_key
   MURF_API_KEY=your_key
   SERPAPI_API_KEY=your_key   # required only for the Web Researcher
   ```
3. Start the backend:
   ```bash
   python -m uvicorn main:app --port 8000
   ```
4. In a second terminal, start the frontend:
   ```bash
   python -m http.server 5500
   ```
5. Open **http://localhost:5500** in Chrome or Edge, allow microphone access, and click **Start conversation**.

Headphones are recommended for the best results.

## Deployment (Render)

1. Push the repository to GitHub.
2. On [Render](https://render.com), choose **New → Web Service** and select the repository.
3. Use these settings:

   | Setting | Value |
   | --- | --- |
   | Root Directory | `Project-MURF_AI-30DaysofVoiceAgents-main` |
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | Health Check Path | `/personas` |

4. Add the environment variables `ASSEMBLYAI_API_KEY`, `GEMINI_API_KEY`, `MURF_API_KEY`, `SERPAPI_API_KEY` (optional) and `PYTHON_VERSION` = `3.11.3`, then deploy.
5. Open the generated `https://<name>.onrender.com` address. The backend serves the web page, so no separate frontend hosting is needed.

The free plan sleeps after 15 minutes of inactivity, so the first visit after that takes about 30–50 seconds to load.

## How It Works

```
User speech → AssemblyAI (text) → Gemini (reply) → Murf (voice) → Audio playback
                                     ↑
                        SerpAPI web search (Web Researcher only)
```

All data is streamed over a WebSocket, so the reply typically begins about 2 seconds after the user stops speaking.

## Project Structure

| File | Purpose |
| --- | --- |
| `main.py` | Starts the server |
| `conversation.py` | Manages a live conversation |
| `personas.py` | Personas, voices and moods |
| `services/` | AssemblyAI, Gemini, Murf and SerpAPI integrations |
| `index.html`, `app.js`, `style.css` | Frontend |

**Note:** AssemblyAI bills for the time a conversation is open, so click **End conversation** when finished.
