/**
 * app.js — Real-time voice conversation client.
 *
 * Streams microphone audio (16 kHz PCM16) to the backend over a WebSocket, renders
 * the conversation as a live chat, plays the assistant's MP3 replies back-to-back,
 * and lets the user interrupt the assistant just by talking (barge-in).
 */

// ── Config ──────────────────────────────────────────────────────────────────

// Served by the backend itself (e.g. on Render) → same origin. Opened from the local
// dev server (port 5500) or as a file → the local backend on port 8000.
const IS_LOCAL_DEV = location.protocol === "file:" || location.port === "5500";
const API_BASE = IS_LOCAL_DEV ? "http://127.0.0.1:8000" : location.origin;
const WS_BASE = API_BASE.replace(/^http/, "ws");
const TARGET_SAMPLE_RATE = 16000;   // What AssemblyAI streaming expects.
const BARGE_IN_MIN_WORDS = 2;       // Words the user must say to interrupt the assistant.
const ECHO_THRESHOLD = 0.6;         // Share of heard words matching the reply to count as echo.
const ERROR_GRACE_MS = 3000;        // Skip the "connection lost" message right after an error.
const PLAYBACK_LEAD_S = 0.03;       // Small delay before a clip starts, to avoid clicks.
const MAX_SOURCE_CHIPS = 3;         // Web sources shown under a Web Researcher reply.

// ── State ───────────────────────────────────────────────────────────────────

const sessionId = getOrCreateSessionId();
let personas = [];

// Connection and audio
let ws = null;
let conversationActive = false;
let endingByUser = false;
let lastErrorAt = 0;
let micStream = null;
let captureCtx = null;
let workletNode = null;
let playbackCtx = null;
const playingSources = new Set();
let nextStartTime = 0;
let playChain = Promise.resolve();
let playing = false;

// Turns
let currentTurnId = null;   // Assistant turn whose audio is accepted.
let currentReply = "";      // Its text so far (used to recognise echo).
let assistantDone = false;  // The server finished sending this turn.
const assistantBubbles = new Map();
const searchSources = new Map();  // Web sources per turn, shown once the reply bubble exists.
let liveUserBubble = null;
let lastUserBubble = null;

// ── DOM ─────────────────────────────────────────────────────────────────────

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("status");
const personaSelect = document.getElementById("personaSelect");
const personaDescription = document.getElementById("personaDescription");
const languageField = document.getElementById("languageField");
const languageSelect = document.getElementById("languageSelect");
const chat = document.getElementById("chat");
const chatEmpty = document.getElementById("chatEmpty");
const issueBox = document.getElementById("issueBox");
const issueToggle = document.getElementById("issueToggle");
const issueDetails = document.getElementById("issueDetails");

// ── Session ─────────────────────────────────────────────────────────────────

/**
 * Reads the session id from the URL, or creates one and adds it to the URL.
 * @returns {string} The current session id.
 */
function getOrCreateSessionId() {
    const params = new URLSearchParams(window.location.search);
    let id = params.get("session_id");
    if (!id) {
        id = self.crypto.randomUUID();
        window.history.pushState(null, "", `?session_id=${id}`);
    }
    return id;
}

console.log(`Current Session ID: ${sessionId}`);

// ── Personas ────────────────────────────────────────────────────────────────

/**
 * Saves a preference in localStorage (best effort; storage may be unavailable).
 * @param {string} key
 * @param {string} value
 */
function savePref(key, value) {
    try { localStorage.setItem(key, value); } catch (_) { /* Storage unavailable. */ }
}

/**
 * Loads a preference from localStorage.
 * @param {string} key
 * @returns {string|null} The stored value, or null.
 */
function loadPref(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
}

/**
 * Builds the config message: the selected persona and language, plus the user's
 * time zone so the assistant knows the local date and time.
 * @returns {{type: string, persona: string, target_language: string|null, timezone: string|null, utc_offset_minutes: number}}
 */
function currentConfig() {
    let timezone = null;
    try { timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || null; } catch (_) { /* Unsupported. */ }
    return {
        type: "config",
        persona: personaSelect.value,
        target_language: personaSelect.value === "translator" ? languageSelect.value : null,
        timezone,
        utc_offset_minutes: -new Date().getTimezoneOffset(),
    };
}

/** Shows the selected persona's description and the language picker for the translator. */
function updatePersonaUI() {
    const selected = personas.find(p => p.id === personaSelect.value);
    personaDescription.textContent = selected ? selected.description : "";
    languageField.hidden = personaSelect.value !== "translator";
}

/**
 * Loads personas and translator languages from the backend and fills the dropdowns.
 * @returns {Promise<void>}
 */
async function loadPersonas() {
    try {
        const res = await fetch(`${API_BASE}/personas`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        personas = data.personas;

        personaSelect.innerHTML = "";
        for (const p of personas) personaSelect.add(new Option(p.name, p.id));
        languageSelect.innerHTML = "";
        for (const l of data.languages) languageSelect.add(new Option(l.label, l.code));

        const savedPersona = loadPref("persona");
        const savedLanguage = loadPref("targetLanguage");
        personaSelect.value = personas.some(p => p.id === savedPersona) ? savedPersona : data.default_persona;
        languageSelect.value = data.languages.some(l => l.code === savedLanguage) ? savedLanguage : data.default_language;
        personaSelect.disabled = false;
        updatePersonaUI();
    } catch (err) {
        console.error("Could not load personas:", err);
        personaSelect.innerHTML = "";
        personaSelect.add(new Option("Friendly Assistant", "assistant"));
        personaDescription.textContent = "";
        showError(
            "🔌 I can't reach the server to load personas. Please make sure the backend is running, then refresh.",
            `Loading personas failed: ${err.message} (${API_BASE}/personas)`,
        );
    }
}

/** Saves the new choice and switches persona live, even mid-conversation. */
function onConfigChange() {
    savePref("persona", personaSelect.value);
    savePref("targetLanguage", languageSelect.value);
    updatePersonaUI();
    sendJson(currentConfig());
}

// ── Errors ──────────────────────────────────────────────────────────────────

/**
 * Turns a status code and technical reason into a short, human message.
 * @param {number} status HTTP-like status code, or 0 if unknown.
 * @param {string} technical Technical error details.
 * @returns {string} A friendly message.
 */
function friendlyError(status, technical) {
    const t = (technical || "").toLowerCase();
    const has = (...words) => words.some(w => t.includes(w));

    if (status === 401 || status === 403 || has("api key", "api_key", "authentication", "unauthorized",
        "permission_denied", "not configured", "http 401", "http 403")) {
        return "🔑 There's a problem with one of the API keys. Please check your .env file and restart the backend.";
    }
    if (status === 429 || has("rate limit", "quota", "resource_exhausted", "429", "credits")) {
        return "⏳ We've hit a usage limit for now. Please wait a moment and try again.";
    }
    if (status === 504 || has("timed out", "timeout")) {
        return "🐢 That took too long to answer. Please try again.";
    }
    if (status === 400) {
        return "⚙️ That option isn't supported. Please refresh the page and try again.";
    }
    if (has("assemblyai", "transcription", "speech-to-text")) {
        return "👂 I'm having trouble hearing you right now. Please try again.";
    }
    if (has("serpapi", "web search")) {
        return "🔎 I couldn't search the web just now, so this answer may not be up to date.";
    }
    if (has("gemini")) {
        return "🤖 I couldn't come up with a reply this time. Please try again.";
    }
    if (has("murf", "audio")) {
        return "🔇 I couldn't create the voice reply. Please try again.";
    }
    return "😕 Something went wrong on my side. Please try again.";
}

/**
 * Shows a friendly message in the status line, with the technical reason behind "View issue".
 * @param {string} friendly
 * @param {string} technical
 */
function showError(friendly, technical) {
    statusText.textContent = friendly;
    showIssue(technical);
}

/**
 * Shows the collapsed "View issue" link holding the technical details.
 * @param {string} technical
 */
function showIssue(technical) {
    if (!technical) return;
    issueDetails.textContent = "❌ Error processing: " + technical;
    issueDetails.hidden = true;
    issueToggle.textContent = "View issue";
    issueToggle.setAttribute("aria-expanded", "false");
    issueBox.hidden = false;
}

/** Hides the "View issue" link and its details. */
function hideIssue() {
    issueBox.hidden = true;
    issueDetails.hidden = true;
}

/** Expands or collapses the technical details. */
function toggleIssue() {
    const open = issueDetails.hidden;
    issueDetails.hidden = !open;
    issueToggle.textContent = open ? "Hide issue" : "View issue";
    issueToggle.setAttribute("aria-expanded", String(open));
}

// ── Chat rendering ──────────────────────────────────────────────────────────

/** Scrolls the chat to the newest message. */
function scrollChat() {
    chat.scrollTop = chat.scrollHeight;
}

/**
 * Appends a chat message.
 * @param {"user"|"assistant"|"system"} role
 * @param {string} [text=""]
 * @returns {HTMLDivElement} The message row.
 */
function addMessage(role, text = "") {
    chatEmpty.hidden = true;
    const row = document.createElement("div");
    row.className = `msg msg-${role}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    const meta = document.createElement("div");
    meta.className = "meta";
    row.append(bubble, meta);
    chat.appendChild(row);
    scrollChat();
    return row;
}

/**
 * Adds a small label (mood, language, voice…) under a message.
 * @param {HTMLDivElement|null} row
 * @param {string} text
 * @param {string} [extraClass=""]
 */
function addMetaChip(row, text, extraClass = "") {
    if (!row) return;
    const chip = document.createElement("span");
    chip.className = `chip ${extraClass}`.trim();
    chip.textContent = text;
    row.querySelector(".meta").appendChild(chip);
}

/**
 * Adds clickable web-source chips under an assistant reply.
 * @param {HTMLDivElement|null} row
 * @param {{title: string, source: string, link: string}[]} sources
 */
function addSourceLinks(row, sources) {
    if (!row || !sources || !sources.length) return;
    const meta = row.querySelector(".meta");
    for (const s of sources.filter(s => /^https?:\/\//i.test(s.link || "")).slice(0, MAX_SOURCE_CHIPS)) {
        const link = document.createElement("a");
        link.className = "chip chip-source";
        link.href = s.link;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = s.title;
        link.textContent = `🔗 ${s.source}`;
        meta.appendChild(link);
    }
}

/**
 * Shows the user's words live while they are still speaking.
 * @param {string} text
 */
function setLiveUserText(text) {
    if (!liveUserBubble) {
        liveUserBubble = addMessage("user");
        liveUserBubble.classList.add("live");
    }
    liveUserBubble.querySelector(".bubble").textContent = text;
    scrollChat();
}

/**
 * Turns the live bubble into the user's final message.
 * @param {string} text
 * @param {string|null} language Detected language code.
 */
function finalizeUserBubble(text, language) {
    const row = liveUserBubble || addMessage("user");
    row.classList.remove("live");
    row.querySelector(".bubble").textContent = text;
    if (language) addMetaChip(row, language.toUpperCase(), "chip-lang");
    liveUserBubble = null;
    lastUserBubble = row;
    scrollChat();
}

/** Removes the live bubble (e.g. when the server discarded the turn as echo). */
function discardLiveBubble() {
    if (liveUserBubble) liveUserBubble.remove();
    liveUserBubble = null;
}

/**
 * Adds a centred, italic note to the chat.
 * @param {string} text
 */
function addSystemNote(text) {
    const row = addMessage("system", text);
    row.querySelector(".meta").remove();
}

/**
 * Marks an assistant reply as interrupted.
 * @param {number} turnId
 */
function markInterrupted(turnId) {
    const row = assistantBubbles.get(turnId);
    if (row && !row.classList.contains("interrupted")) {
        row.classList.add("interrupted");
        addMetaChip(row, "⏹ interrupted");
    }
}

/** Shows what the app is doing right now in the status line. */
function setListeningStatus() {
    if (!conversationActive) return;
    statusText.textContent = playing
        ? "🔊 Speaking… just start talking to interrupt me."
        : "🎤 Listening… I'll reply after 3 seconds of silence.";
}

// ── Echo detection ──────────────────────────────────────────────────────────

/**
 * Splits text into lowercase words (any script).
 * @param {string} text
 * @returns {string[]}
 */
function words(text) {
    return text.toLowerCase().match(/[\p{L}\p{N}']+/gu) || [];
}

/**
 * True if the mic mostly heard the assistant's own voice rather than the user.
 * @param {string} heard What speech-to-text picked up.
 * @param {string} spoken What the assistant is saying.
 * @returns {boolean}
 */
function looksLikeEcho(heard, spoken) {
    const heardWords = words(heard);
    if (!heardWords.length || !spoken) return false;
    const spokenWords = new Set(words(spoken));
    const hits = heardWords.filter(w => spokenWords.has(w)).length;
    return hits / heardWords.length >= ECHO_THRESHOLD;
}

// ── WebSocket ───────────────────────────────────────────────────────────────

/**
 * Sends a JSON message if the socket is open.
 * @param {object} obj
 */
function sendJson(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

/** Opens the conversation socket and wires up its events. */
function openSocket() {
    let opened = false;
    endingByUser = false;
    const socket = new WebSocket(`${WS_BASE}/ws/conversation/${sessionId}`);
    socket.binaryType = "arraybuffer";
    ws = socket;

    socket.onopen = () => {
        opened = true;
        sendJson(currentConfig());
    };

    socket.onmessage = (event) => {
        if (socket === ws && typeof event.data === "string") handleServerMessage(JSON.parse(event.data));
    };

    socket.onclose = (event) => {
        if (ws !== null && ws !== socket) return; // An old socket closing after a restart.
        const userEnded = endingByUser;
        teardown();
        if (userEnded || Date.now() - lastErrorAt < ERROR_GRACE_MS) return; // Already explained.
        if (!opened) {
            showError(
                "🔌 I can't reach the server. Please make sure the backend is running and try again.",
                `WebSocket connection failed (code ${event.code}) — ${WS_BASE}`,
            );
        } else {
            showError(
                "🔌 The connection to the server was lost. Click Start conversation to reconnect.",
                `WebSocket closed (code ${event.code}${event.reason ? ", " + event.reason : ""})`,
            );
        }
    };
}

/**
 * Handles one message from the server.
 * @param {{type: string, [key: string]: any}} msg
 */
function handleServerMessage(msg) {
    switch (msg.type) {
        case "ready":
            conversationActive = true;
            stopBtn.disabled = false;
            startCapture().catch(err => {
                showError("🎤 I couldn't start the microphone. Please try again.",
                    `Audio capture error: ${err.name}: ${err.message}`);
                endConversation(false);
            });
            setListeningStatus();
            break;

        case "partial":
            onPartial(msg.text);
            break;

        case "user_turn":
            hideIssue();
            finalizeUserBubble(msg.text, msg.language);
            statusText.textContent = "🤔 Thinking…";
            break;

        case "turn_discarded":
            discardLiveBubble();
            setListeningStatus();
            break;

        case "search":
            if (msg.status === "searching") statusText.textContent = "🔎 Searching the web…";
            else if (msg.status === "done") {
                searchSources.set(msg.turn_id, msg.sources);
                statusText.textContent = "🤔 Thinking…";
            }
            break;

        case "assistant_start": {
            // A new reply replaces whatever was playing before.
            if (currentTurnId !== null && currentTurnId !== msg.turn_id) stopPlayback();
            currentTurnId = msg.turn_id;
            currentReply = "";
            assistantDone = false;
            addMetaChip(lastUserBubble, `${msg.mood_emoji} ${msg.mood}`, `chip-mood mood-${msg.mood}`);
            const row = addMessage("assistant");
            addMetaChip(row, `🔊 ${msg.delivery || "natural"} voice`);
            addSourceLinks(row, searchSources.get(msg.turn_id));
            searchSources.delete(msg.turn_id);
            assistantBubbles.set(msg.turn_id, row);
            break;
        }

        case "assistant_delta":
            if (msg.turn_id !== currentTurnId) break;
            currentReply += msg.text;
            assistantBubbles.get(msg.turn_id).querySelector(".bubble").textContent = currentReply.trim();
            scrollChat();
            break;

        case "audio":
            if (msg.turn_id === currentTurnId) enqueueAudio(msg);
            break;

        case "assistant_end":
            if (msg.turn_id !== currentTurnId) break;
            assistantDone = true;
            if (msg.failed && !currentReply.trim()) {
                const row = assistantBubbles.get(msg.turn_id);
                if (row) row.remove();
            }
            if (!playingSources.size) onPlaybackFinished();
            break;

        case "assistant_interrupted":
            markInterrupted(msg.turn_id);
            break;

        case "history_reset":
            addSystemNote(`🔄 Switched to ${msg.persona}. Starting a fresh conversation.`);
            break;

        case "error":
            lastErrorAt = Date.now();
            showError(msg.message || friendlyError(0, msg.technical), msg.technical);
            break;
    }
}

/**
 * Shows live speech and interrupts the assistant when the user talks over it.
 * @param {string} text The partial transcript.
 */
function onPartial(text) {
    if (isPlaying()) {
        if (looksLikeEcho(text, currentReply)) return;          // Our own voice leaking into the mic.
        if (words(text).length < BARGE_IN_MIN_WORDS) return;     // Wait for a clearer interruption.
        bargeIn();
    }
    setLiveUserText(text);
    statusText.textContent = "🗣️ I'm listening…";
}

/** Stops the assistant's audio and tells the server to cancel the reply. */
function bargeIn() {
    const turnId = currentTurnId;
    stopPlayback();
    currentTurnId = null;
    sendJson({ type: "interrupt" });
    if (turnId !== null) markInterrupted(turnId);
}

// ── Audio capture ───────────────────────────────────────────────────────────

/**
 * AudioWorklet that resamples the mic to TARGET_SAMPLE_RATE (linear interpolation)
 * and posts 100 ms chunks of 16-bit PCM. Loaded from a Blob URL, so no extra file is needed.
 */
const WORKLET_CODE = `
class PcmCapture extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.target = options.processorOptions.targetRate;
        this.ratio = sampleRate / this.target;
        this.chunk = Math.round(this.target / 10);
        this.buf = new Int16Array(this.chunk);
        this.n = 0;
        this.pos = 0;
    }

    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;
        while (this.pos < ch.length) {
            const i = Math.floor(this.pos);
            const f = this.pos - i;
            const a = ch[i];
            const b = i + 1 < ch.length ? ch[i + 1] : a;
            const s = Math.max(-1, Math.min(1, a + (b - a) * f));
            this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
            if (this.n === this.chunk) {
                this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
                this.buf = new Int16Array(this.chunk);
                this.n = 0;
            }
            this.pos += this.ratio;
        }
        this.pos -= ch.length;
        return true;
    }
}
registerProcessor("pcm-capture", PcmCapture);
`;

/**
 * Starts streaming the microphone to the server.
 * @returns {Promise<void>}
 */
async function startCapture() {
    captureCtx = new AudioContext();
    const moduleUrl = URL.createObjectURL(new Blob([WORKLET_CODE], { type: "application/javascript" }));
    await captureCtx.audioWorklet.addModule(moduleUrl);
    URL.revokeObjectURL(moduleUrl);

    const source = captureCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(captureCtx, "pcm-capture", {
        processorOptions: { targetRate: TARGET_SAMPLE_RATE },
    });
    workletNode.port.onmessage = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
    source.connect(workletNode);
    workletNode.connect(captureCtx.destination); // Outputs silence; keeps the node processing.
}

/** Stops the microphone and releases the capture graph. */
function stopCapture() {
    if (workletNode) {
        workletNode.port.onmessage = null;
        workletNode.disconnect();
        workletNode = null;
    }
    if (captureCtx) {
        captureCtx.close().catch(() => {});
        captureCtx = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
}

// ── Playback ────────────────────────────────────────────────────────────────

/**
 * Decodes base64 into raw bytes.
 * @param {string} b64
 * @returns {ArrayBuffer}
 */
function base64ToArrayBuffer(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}

/**
 * @returns {boolean} True while any reply clip is playing.
 */
function isPlaying() {
    return playingSources.size > 0;
}

/**
 * Records the playback state and reports it to the server (used for echo filtering).
 * @param {boolean} value
 */
function setPlaying(value) {
    if (playing === value) return;
    playing = value;
    sendJson({ type: "playback", playing: value });
    setListeningStatus();
}

/**
 * Schedules one MP3 sentence clip right after the previous one (gapless).
 * Clips are decoded in arrival order so sentences never play out of sequence.
 * @param {{turn_id: number, data: string}} msg
 */
function enqueueAudio(msg) {
    const turnId = msg.turn_id;
    playChain = playChain.then(async () => {
        if (turnId !== currentTurnId || !playbackCtx) return;
        let buffer;
        try {
            buffer = await playbackCtx.decodeAudioData(base64ToArrayBuffer(msg.data));
        } catch (err) {
            console.warn("Could not decode audio clip:", err);
            return;
        }
        if (turnId !== currentTurnId) return; // Interrupted while decoding.

        const src = playbackCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(playbackCtx.destination);
        const startAt = Math.max(playbackCtx.currentTime + PLAYBACK_LEAD_S, nextStartTime);
        src.start(startAt);
        nextStartTime = startAt + buffer.duration;
        playingSources.add(src);
        setPlaying(true);
        src.onended = () => {
            playingSources.delete(src);
            if (!playingSources.size && assistantDone) onPlaybackFinished();
        };
    });
}

/** Called when the last clip of a reply has finished. */
function onPlaybackFinished() {
    setPlaying(false);
    setListeningStatus();
}

/** Stops all scheduled clips immediately. */
function stopPlayback() {
    for (const src of playingSources) {
        src.onended = null;
        try { src.stop(); } catch (_) { /* Already stopped. */ }
    }
    playingSources.clear();
    nextStartTime = 0;
    playChain = Promise.resolve();
    setPlaying(false);
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

/**
 * Asks for the microphone and opens the conversation.
 * @returns {Promise<void>}
 */
async function startConversation() {
    recordBtn.disabled = true;
    hideIssue();
    statusText.textContent = "⏳ Connecting…";
    try {
        // Created inside the click so the browser allows audio playback.
        playbackCtx = playbackCtx || new AudioContext();
        await playbackCtx.resume();
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
        });
    } catch (err) {
        const friendly = err.name === "NotFoundError"
            ? "🎤 I couldn't find a microphone. Please connect one and try again."
            : "🎤 I can't use your microphone. Please allow microphone access in your browser and try again.";
        showError(friendly, `Microphone error: ${err.name}: ${err.message}`);
        console.error("Error accessing microphone:", err);
        recordBtn.disabled = false;
        return;
    }
    openSocket();
}

/** Resets all audio and connection state. */
function teardown() {
    conversationActive = false;
    stopCapture();
    stopPlayback();
    discardLiveBubble();
    currentTurnId = null;
    ws = null;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
}

/**
 * Ends the conversation for good.
 * @param {boolean} [showMessage=true] Whether to show the goodbye message.
 */
function endConversation(showMessage = true) {
    endingByUser = true;
    sendJson({ type: "stop" });
    if (ws) ws.close();
    teardown();
    if (showMessage) statusText.textContent = "👋 Conversation ended. Click Start conversation to talk again.";
}

// ── Init ────────────────────────────────────────────────────────────────────

personaSelect.addEventListener("change", onConfigChange);
languageSelect.addEventListener("change", onConfigChange);
issueToggle.addEventListener("click", toggleIssue);
recordBtn.addEventListener("click", startConversation);
stopBtn.addEventListener("click", () => endConversation(true));

loadPersonas();
