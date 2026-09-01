"use strict";

document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("ready");

  document.querySelectorAll("[data-open-dialog]").forEach((button) => {
    button.addEventListener("click", () => {
      const dialog = document.getElementById(button.dataset.openDialog);
      if (dialog instanceof HTMLDialogElement) {
        dialog.showModal();
      }
    });
  });

  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog")?.close());
  });

  document.querySelectorAll("dialog").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) {
        dialog.close();
      }
    });
  });

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) {
        event.preventDefault();
      }
    });
  });

  document.querySelectorAll("button[data-confirm]").forEach((button) => {
    button.addEventListener("click", (event) => {
      if (!window.confirm(button.dataset.confirm)) {
        event.preventDefault();
      }
    });
  });

  const ttsProvider = document.getElementById("tts-provider");
  const ttsModel = document.getElementById("tts-model");
  const ttsVoice = document.getElementById("tts-voice");
  const ttsVoiceLabel = document.getElementById("tts-voice-label");
  const ttsVoiceHelp = document.getElementById("tts-voice-help");
  const ttsLanguageField = document.getElementById("tts-language-field");

  const updateTtsFields = (changedByUser = false) => {
    if (!(ttsProvider instanceof HTMLSelectElement)) return;
    const isElevenLabs = ttsProvider.value === "elevenlabs";
    if (ttsVoiceLabel) {
      ttsVoiceLabel.textContent = isElevenLabs ? "ElevenLabs Voice ID" : "Голос";
    }
    if (ttsVoiceHelp) {
      ttsVoiceHelp.textContent = isElevenLabs
        ? "Voice ID можно скопировать из Voice Library в ElevenLabs."
        : "Например, Cherry.";
    }
    if (ttsLanguageField) {
      ttsLanguageField.hidden = isElevenLabs;
    }
    if (changedByUser && ttsModel instanceof HTMLInputElement) {
      ttsModel.value = isElevenLabs ? "eleven_multilingual_v2" : "qwen3-tts-flash";
    }
    if (changedByUser && ttsVoice instanceof HTMLInputElement) {
      ttsVoice.value = isElevenLabs ? "JBFqnCBsd6RMkjVDRZzb" : "Cherry";
    }
  };

  if (ttsProvider instanceof HTMLSelectElement) {
    updateTtsFields();
    ttsProvider.addEventListener("change", () => updateTtsFields(true));
  }

  const imageProvider = document.getElementById("image-provider");
  const imageModel = document.getElementById("image-model");
  const imageProviderHelp = document.getElementById("image-provider-help");

  const updateImageProviderFields = (changedByUser = false) => {
    if (!(imageProvider instanceof HTMLSelectElement)) return;
    const isQwen = imageProvider.value === "qwen";
    if (imageProviderHelp) {
      imageProviderHelp.textContent = isQwen
        ? "Используется DashScope key и настроенный Qwen Image endpoint."
        : "Используется BytePlus API key.";
    }
    if (changedByUser && imageModel instanceof HTMLInputElement) {
      imageModel.value = isQwen ? "qwen-image-3.0" : "seedream-5-0-260128";
    }
  };

  if (imageProvider instanceof HTMLSelectElement) {
    updateImageProviderFields();
    imageProvider.addEventListener("change", () => {
      updateImageProviderFields(true);
    });
  }
});
