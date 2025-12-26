const chatbox = document.getElementById("chatbox");
const input = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const speakBtn = document.getElementById("speakBtn");

let dropdownSelected = document.getElementById("dropdownSelected");
let dropdownMenu = document.getElementById("dropdownMenu");
let selectedLang = "en";

let lastBotAudio = null;
let audioPlayer = null;

// DROPDOWN
dropdownSelected.onclick = () => {
    let arrow = dropdownSelected.querySelector(".arrow");
    dropdownMenu.classList.toggle("show");
    arrow.style.transform = dropdownMenu.classList.contains("show") ? "rotate(180deg)" : "rotate(0deg)";
};

document.querySelectorAll(".dropdown-item").forEach(item => {
    item.onclick = () => {
        selectedLang = item.dataset.lang;
        dropdownSelected.innerHTML = `${selectedLang.toUpperCase()} <span class="arrow">▾</span>`;
        dropdownMenu.classList.remove("show");
    };
});

// ADD MESSAGE
function addMessage(text, sender) {
    let msg = document.createElement("div");
    msg.className = `msg ${sender}`;

    if (sender === "bot") {
        msg.innerHTML = `
            <img src="/static/robo.png" class="chat-avatar">
            <div class="msg-bubble">${text}</div>
        `;
    } else {
        msg.innerHTML = `
            <div class="msg-bubble user-bubble">${text}</div>
        `;
    }

    chatbox.appendChild(msg);
    chatbox.scrollTop = chatbox.scrollHeight;
}

// AUDIO HELPERS
function stopAudio() {
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
        audioPlayer = null;
    }
}

// SEND MESSAGE
async function sendMessage() {
    let text = input.value.trim();
    if (!text) return;

    stopAudio();
    addMessage(text, "user");
    input.value = "";

    let typing = document.createElement("div");
    typing.className = "typing-bubble";
    typing.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div>`;
    chatbox.appendChild(typing);

    let res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, lang: selectedLang })
    });

    let data = await res.json();
    typing.remove();

    stopAudio();
    addMessage(data.text, "bot");
    lastBotAudio = data.audio;
}

input.addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
});
sendBtn.onclick = sendMessage;

// MIC
micBtn.onclick = () => {
    let rec = new webkitSpeechRecognition();
    rec.lang = selectedLang;
    rec.onresult = e => {
        input.value = e.results[0][0].transcript;
        sendMessage();
    };
    rec.start();
};

// SPEAKER
speakBtn.onclick = () => {
    if (!lastBotAudio) return;

    if (audioPlayer && !audioPlayer.paused) {
        stopAudio();
        return;
    }

    audioPlayer = new Audio(lastBotAudio);
    audioPlayer.play();
};

// GREETING
window.onload = () => {
    addMessage("😊 Hello! I'm your AI assistant. How can I help you today?", "bot");
};
