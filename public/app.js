const form = document.querySelector("#chatForm");
const messagesEl = document.querySelector("#messages");
const promptInput = document.querySelector("#promptInput");
const imageInput = document.querySelector("#imageInput");
const imagePreviewWrap = document.querySelector("#imagePreviewWrap");
const imagePreview = document.querySelector("#imagePreview");
const removeImageButton = document.querySelector("#removeImageButton");
const sendButton = document.querySelector("#sendButton");
const newChatButton = document.querySelector("#newChatButton");
const mobileMenuButton = document.querySelector("#mobileMenuButton");
const sidebar = document.querySelector(".sidebar");
const modelSelect = document.querySelector("#modelSelect");
const modelTitle = document.querySelector("#modelTitle");
const modelDescription = document.querySelector("#modelDescription");

let selectedImage = null;
let chatMessages = [];

const modelDetails = {
  kimi: {
    title: "Kimi-K2.6",
    assistantAvatar: "K",
    placeholder: "Message Kimi...",
    description: "Ask anything. Attach an image to test multimodal support.",
  },
  "azure-claude-opus-4-7": {
    title: "Claude Opus 4.7",
    assistantAvatar: "O",
    placeholder: "Message Claude Opus...",
    description: "Ask anything with your Azure Anthropic Foundry deployment.",
  },
};

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function clearWelcome() {
  const welcomeCard = messagesEl.querySelector(".welcome-card");
  if (welcomeCard) {
    welcomeCard.remove();
  }
}

function createMessageElement(role, text, imageDataUrl, isError = false) {
  const currentModel = modelDetails[modelSelect.value] || modelDetails.kimi;
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "S" : currentModel.assistantAvatar;

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = `bubble${isError ? " error" : ""}`;

  if (imageDataUrl) {
    const img = document.createElement("img");
    img.className = "uploaded-image";
    img.src = imageDataUrl;
    img.alt = "Uploaded image";
    bubble.appendChild(img);
  }

  const textNode = document.createElement("div");
  textNode.textContent = text;
  bubble.appendChild(textNode);

  content.appendChild(bubble);

  if (role === "user") {
    article.append(content, avatar);
  } else {
    article.append(avatar, content);
  }

  return article;
}

function addMessage(role, text, imageDataUrl, isError = false) {
  clearWelcome();
  const messageEl = createMessageElement(role, text, imageDataUrl, isError);
  messagesEl.appendChild(messageEl);
  scrollToBottom();
  return messageEl;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  promptInput.disabled = isLoading;
  imageInput.disabled = isLoading;
  modelSelect.disabled = isLoading;
  sendButton.querySelector("span").textContent = isLoading ? "..." : "Send";
}

function autoresizeTextarea() {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read selected image."));
    reader.readAsDataURL(file);
  });
}

function removeSelectedImage() {
  selectedImage = null;
  imageInput.value = "";
  imagePreview.src = "";
  imagePreviewWrap.classList.add("hidden");
}

function buildUserMessage(text, imageDataUrl) {
  if (!imageDataUrl) {
    return { role: "user", content: text };
  }

  return {
    role: "user",
    content: [
      { type: "text", text },
      {
        type: "image_url",
        image_url: {
          url: imageDataUrl,
        },
      },
    ],
  };
}

async function sendMessage(text, imageDataUrl) {
  const userMessage = buildUserMessage(text, imageDataUrl);
  const requestMessages = [...chatMessages, userMessage];
  const model = modelSelect.value;

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, messages: requestMessages }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "The request failed.");
  }

  const assistantText = data.message || "No response returned.";
  chatMessages = [...requestMessages, { role: "assistant", content: assistantText }];
  return assistantText;
}

promptInput.addEventListener("input", autoresizeTextarea);

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    addMessage("assistant", "Please choose a valid image file.", null, true);
    removeSelectedImage();
    return;
  }

  selectedImage = await readFileAsDataUrl(file);
  imagePreview.src = selectedImage;
  imagePreviewWrap.classList.remove("hidden");
});

removeImageButton.addEventListener("click", removeSelectedImage);

function updateSelectedModel() {
  const currentModel = modelDetails[modelSelect.value] || modelDetails.kimi;
  modelTitle.textContent = currentModel.title;
  modelDescription.textContent = currentModel.description;
  promptInput.placeholder = currentModel.placeholder;
}

modelSelect.addEventListener("change", updateSelectedModel);

newChatButton.addEventListener("click", () => {
  chatMessages = [];
  selectedImage = null;
  messagesEl.innerHTML = `
    <div class="welcome-card">
      <div class="spark">*</div>
      <h2>How can I help?</h2>
      <p>Type a message or upload an image. Your API key stays on the local backend.</p>
    </div>
  `;
  promptInput.value = "";
  autoresizeTextarea();
  removeSelectedImage();
});

mobileMenuButton.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = promptInput.value.trim();
  const imageDataUrl = selectedImage;

  if (!text && !imageDataUrl) return;

  addMessage("user", text || "Please analyze this image.", imageDataUrl);
  promptInput.value = "";
  autoresizeTextarea();
  removeSelectedImage();
  setLoading(true);

  const loadingMessage = addMessage("assistant", "Thinking...");

  try {
    const assistantText = await sendMessage(text || "Please analyze this image.", imageDataUrl);
    loadingMessage.replaceWith(createMessageElement("assistant", assistantText));
    scrollToBottom();
  } catch (error) {
    loadingMessage.replaceWith(
      createMessageElement("assistant", error.message || "Something went wrong.", null, true)
    );
    scrollToBottom();
  } finally {
    setLoading(false);
    promptInput.focus();
  }
});

updateSelectedModel();
