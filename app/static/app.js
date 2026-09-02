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

  const generateVideo = document.getElementById("generate-video");
  const projectForm = document.getElementById("project-settings-form");
  const progressBox = document.getElementById("pipeline-progress");
  const progressBar = document.getElementById("pipeline-progress-bar");
  const progressStage = document.getElementById("pipeline-stage");
  const progressPercent = document.getElementById("pipeline-percent");
  const progressMessage = document.getElementById("pipeline-message");
  const progressError = document.getElementById("pipeline-error");
  let pollingTimer = null;

  const renderJob = (job) => {
    if (progressBox) progressBox.hidden = false;
    if (progressBar instanceof HTMLProgressElement) progressBar.value = job.progress || 0;
    if (progressStage) progressStage.textContent = job.current_stage || job.status;
    if (progressPercent) progressPercent.textContent = `${job.progress || 0}%`;
    if (progressMessage) progressMessage.textContent = job.message || "";
    if (progressError) progressError.textContent = job.error || "";
    if (generateVideo instanceof HTMLButtonElement) {
      generateVideo.disabled = ["queued", "running"].includes(job.status);
      generateVideo.textContent = job.status === "failed" ? "Продолжить / повторить" : "Сгенерировать видео";
    }
  };

  const pollJob = async (jobId) => {
    if (!jobId) return;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) return;
      const job = await response.json();
      renderJob(job);
      if (["completed", "failed"].includes(job.status)) {
        window.clearInterval(pollingTimer);
        pollingTimer = null;
        if (job.status === "completed") window.location.reload();
      }
    } catch (_) {
      // A later poll can recover from a transient local connection error.
    }
  };

  if (generateVideo instanceof HTMLButtonElement && projectForm instanceof HTMLFormElement) {
    const existingJob = generateVideo.dataset.jobId;
    if (existingJob) {
      pollJob(existingJob);
      pollingTimer = window.setInterval(() => pollJob(existingJob), 1500);
    }
    generateVideo.addEventListener("click", async () => {
      generateVideo.disabled = true;
      if (progressError) progressError.textContent = "";
      const body = new URLSearchParams(new FormData(projectForm));
      const response = await fetch(`/api/projects/${generateVideo.dataset.projectId}/generate-video`, {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        if (progressBox) progressBox.hidden = false;
        if (progressError) progressError.textContent = payload.detail || "Не удалось запустить pipeline";
        generateVideo.disabled = false;
        return;
      }
      renderJob(payload);
      generateVideo.dataset.jobId = payload.id;
      if (pollingTimer) window.clearInterval(pollingTimer);
      pollingTimer = window.setInterval(() => pollJob(payload.id), 1500);
    });
  }

  const uploadStyle = document.getElementById("upload-style-reference");
  const styleFile = document.getElementById("style-reference-file");
  if (uploadStyle instanceof HTMLButtonElement && styleFile instanceof HTMLInputElement) {
    uploadStyle.addEventListener("click", async () => {
      const file = styleFile.files?.[0];
      if (!file) return;
      uploadStyle.disabled = true;
      const response = await fetch(`/api/projects/${uploadStyle.dataset.projectId}/style-reference`, {
        method: "POST",
        headers: {"Content-Type": "image/png"},
        body: file,
      });
      if (response.ok) window.location.reload();
      else {
        const payload = await response.json();
        window.alert(payload.detail || "Не удалось сохранить style reference");
        uploadStyle.disabled = false;
      }
    });
  }
});
