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

  const planningProvider = document.getElementById("planning-provider");
  const planningModel = document.getElementById("planning-model");
  const planningProviderHelp = document.getElementById("planning-provider-help");

  const updatePlanningProviderFields = (changedByUser = false) => {
    if (!(planningProvider instanceof HTMLSelectElement)) return;
    const isKimi = planningProvider.value === "kimi";
    if (planningProviderHelp) {
      planningProviderHelp.textContent = isKimi
        ? "Используются Kimi API key и модель kimi-k3."
        : "Используются DashScope API key и модель qwen-plus.";
    }
    if (changedByUser && planningModel instanceof HTMLInputElement) {
      planningModel.value = isKimi ? "kimi-k3" : "qwen-plus";
    }
  };

  if (planningProvider instanceof HTMLSelectElement) {
    updatePlanningProviderFields();
    planningProvider.addEventListener("change", () => {
      updatePlanningProviderFields(true);
    });
  }

  const imageProvider = document.getElementById("image-provider");
  const imageModel = document.getElementById("image-model");
  const imageProviderHelp = document.getElementById("image-provider-help");

  const updateImageProviderFields = (changedByUser = false) => {
    if (!(imageProvider instanceof HTMLSelectElement)) return;
    const isQwen = imageProvider.value === "qwen";
    const isZImage = imageProvider.value === "zimage";
    if (imageModel instanceof HTMLSelectElement) {
      Array.from(imageModel.options).forEach((option) => {
        option.disabled = option.dataset.provider !== imageProvider.value;
      });
    }
    if (imageProviderHelp) {
      imageProviderHelp.textContent = isQwen
        ? "Выберите Qwen Image 2.0 или 3.0. Используются DashScope key и настроенный endpoint."
        : isZImage
          ? "Z-Image через Kie.ai и Kie.ai API key. Только создание новых изображений."
          : "Используется BytePlus API key.";
    }
    if (changedByUser && (imageModel instanceof HTMLInputElement || imageModel instanceof HTMLSelectElement)) {
      imageModel.value = isQwen
        ? "qwen-image-3.0"
        : isZImage
          ? "z-image"
          : "seedream-5-0-260128";
    }
  };

  if (imageProvider instanceof HTMLSelectElement) {
    updateImageProviderFields();
    imageProvider.addEventListener("change", () => {
      updateImageProviderFields(true);
    });
  }

  const generationActions = document.getElementById("generation-actions");
  const profileButtons = Array.from(document.querySelectorAll(".generate-profile"));
  const continueOverBudget = document.getElementById("continue-over-budget");
  const retryPlanningAnyway = document.getElementById("retry-planning-anyway");
  const cancelGeneration = document.getElementById("cancel-generation");
  const projectForm = document.getElementById("project-settings-form");
  const progressBox = document.getElementById("pipeline-progress");
  const progressBar = document.getElementById("pipeline-progress-bar");
  const progressStage = document.getElementById("pipeline-stage");
  const progressPercent = document.getElementById("pipeline-percent");
  const progressMessage = document.getElementById("pipeline-message");
  const planningAttempts = document.getElementById("planning-attempts");
  const progressError = document.getElementById("pipeline-error");
  const costRun = document.getElementById("cost-run");
  const costEstimate = document.getElementById("cost-estimate");
  const costRemaining = document.getElementById("cost-remaining");
  const costTotal = document.getElementById("cost-total");
  const costQaRetries = document.getElementById("cost-qa-retries");
  const costNote = document.getElementById("cost-note");
  const budgetAmount = document.getElementById("budget-amount");
  const budgetAvailable = document.getElementById("budget-available");
  const budgetWarning = document.getElementById("budget-warning");
  let pollingTimer = null;

  const renderJob = (job) => {
    const planning = job.planning_progress;
    const isPlanning = Boolean(planning?.state) && (
      job.current_stage === "PLANNING"
      || ["PAUSED_AFTER_TIMEOUT", "PAUSED_BUDGET", "FAILED"].includes(planning.state)
    );
    if (progressBox) progressBox.hidden = false;
    if (progressBar instanceof HTMLProgressElement) {
      progressBar.hidden = isPlanning;
      progressBar.value = job.progress || 0;
    }
    if (progressPercent) {
      progressPercent.hidden = isPlanning;
      progressPercent.textContent = `${job.progress || 0}%`;
    }
    if (progressStage) progressStage.textContent = isPlanning ? "PLANNING" : (job.current_stage || job.status);
    if (progressMessage) progressMessage.textContent = isPlanning ? "" : (job.message || "");
    if (planningAttempts) {
      const stateLabels = {
        PREPARING_SCOPE: "Preparing scope",
        ESTIMATING_COST: "Estimating cost",
        WAITING_FOR_PROVIDER: "Waiting for provider",
        VALIDATING_RESPONSE: "Validating response",
        REPAIRING_PLAN: "Repairing plan",
        PAUSED_AFTER_TIMEOUT: "Planning paused",
        PAUSED_BUDGET: "Paused by planning budget",
        COMPLETED: "Completed",
        FAILED: "Failed",
      };
      const currency = planning?.cost?.currency || "";
      const money = (value) => value === null || value === undefined
        ? "—"
        : `${Number(value).toFixed(4)}${currency ? ` ${currency}` : ""}`;
      const duration = (seconds) => {
        if (seconds === null || seconds === undefined) return "—";
        const rounded = Math.max(0, Math.floor(Number(seconds)));
        return `${String(Math.floor(rounded / 60)).padStart(2, "0")}:${String(rounded % 60).padStart(2, "0")}`;
      };
      const lines = [];
      if (planning?.state) lines.push(`Status: ${stateLabels[planning.state] || planning.state}`);
      if (planning?.scope?.label) lines.push(`Scope: ${planning.scope.label}`);
      if (planning?.provider?.name || planning?.provider?.model) {
        lines.push(`Provider: ${[planning.provider.name, planning.provider.model].filter(Boolean).join(" · ")}`);
      }
      if (planning?.request?.number) {
        lines.push(`Paid request: ${planning.request.number} of maximum ${planning.request.maximum} · Type: ${planning.request.type}`);
        if (planning.request.type === "REPAIR") lines.push(`Repair attempt: ${planning.request.number} / ${planning.request.maximum}`);
        lines.push(`Elapsed: ${duration(planning.request.elapsed_seconds)}`);
      }
      if (planning?.tokens?.input_estimate !== null && planning?.tokens?.input_estimate !== undefined) {
        lines.push(`Input: ~${Number(planning.tokens.input_estimate).toLocaleString()} tokens · Max output: ${Number(planning.tokens.max_output).toLocaleString()} tokens`);
      }
      if (planning?.cost) lines.push(`Estimated max cost: ${money(planning.cost.estimated_max)} · ${planning.cost.certainty || "UNKNOWN"}`);
      if (planning?.repair?.reason) lines.push(`Repair reason: ${planning.repair.reason}`);
      const budget = planning?.cost?.budget;
      if (budget?.enabled) lines.push(`Planning budget: spent ${money(budget.spent)} · reserved unknown ${money(budget.reserved_unknown)} · remaining ${money(budget.remaining)} / ${money(budget.amount)}`);
      if (planning?.timeout?.paused) {
        lines.push(planning.timeout.message);
        lines.push(`Estimated/unknown exposure: ${money(budget?.reserved_unknown ?? planning.cost.estimated_max)}`);
      }
      planningAttempts.textContent = lines.join("\n");
    }
    if (progressError) {
      let failureMessage = job.budget_pause?.message || job.error || "";
      if (job.diagnostic && !job.budget_pause) {
        const provider = [job.diagnostic.provider, job.diagnostic.model]
          .filter(Boolean)
          .join(" / ");
        const reason = job.diagnostic.provider_error || job.diagnostic.error_type;
        const details = [provider, job.diagnostic.operation, reason]
          .filter(Boolean)
          .join(" · ");
        if (details) failureMessage = `${failureMessage} — ${details}`;
      }
      progressError.textContent = failureMessage;
    }
    if (job.cost) {
      const suffix = job.cost.currency ? ` ${job.cost.currency}` : "";
      if (costRun) costRun.textContent = job.cost.run_cost === null ? "—" : `${Number(job.cost.run_cost).toFixed(4)}${suffix}`;
      if (costTotal) costTotal.textContent = job.cost.historical_project_cost === null ? "—" : `${Number(job.cost.historical_project_cost).toFixed(4)}${suffix}`;
      if (costQaRetries) costQaRetries.textContent = `${Number(job.cost.qa_retry_cost || 0).toFixed(4)}${suffix}`;
      if (costNote && job.cost.unpriced_records) costNote.textContent = `Без цены: ${job.cost.unpriced_records} запросов. Настройте версионированные тарифы.`;
    }
    if (job.cost_estimate && job.cost_estimate.minimum !== null) {
      const suffix = job.cost_estimate.currency ? ` ${job.cost_estimate.currency}` : "";
      if (costEstimate) costEstimate.textContent = `${Number(job.cost_estimate.minimum).toFixed(4)}–${Number(job.cost_estimate.maximum).toFixed(4)}${suffix}`;
    }
    if (costRemaining) {
      const suffix = job.cost?.currency || job.cost_estimate?.currency;
      costRemaining.textContent = job.estimated_remaining === null ? "—" : `${Number(job.estimated_remaining).toFixed(4)}${suffix ? ` ${suffix}` : ""}`;
    }
    if (job.budget) {
      if (budgetAmount) budgetAmount.textContent = job.budget.enabled ? `${Number(job.budget.amount).toFixed(2)} ${job.budget.currency}` : "Не включён";
      if (budgetAvailable) budgetAvailable.textContent = job.budget.available === null ? "—" : `${Number(job.budget.available).toFixed(4)} ${job.budget.currency}`;
      if (budgetWarning) budgetWarning.textContent = job.budget.warning || "";
    }
    if (generationActions instanceof HTMLElement) {
      generationActions.dataset.jobStatus = job.status;
      generationActions.dataset.jobId = job.id;
      generationActions.dataset.jobProfile = job.production_profile || "FINAL";
      generationActions.dataset.jobScope = job.generation_scope?.type || "FULL";
      generationActions.dataset.jobScopeValue = job.generation_scope?.value ?? "";
    }
    profileButtons.forEach((button) => {
      if (button instanceof HTMLButtonElement) button.disabled = ["queued", "running"].includes(job.status);
    });
    if (continueOverBudget instanceof HTMLButtonElement) continueOverBudget.hidden = job.status !== "paused_budget";
    if (retryPlanningAnyway instanceof HTMLButtonElement) retryPlanningAnyway.hidden = job.status !== "paused_planning";
    if (cancelGeneration instanceof HTMLButtonElement) {
      cancelGeneration.hidden = !["queued", "running", "paused_planning"].includes(job.status);
      cancelGeneration.textContent = job.status === "paused_planning" ? "Stop" : "Остановить локально";
    }
  };

  const pollJob = async (jobId) => {
    if (!jobId) return;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!response.ok) return;
      const job = await response.json();
      renderJob(job);
      if (["completed", "failed", "paused_budget", "paused_planning", "cancelled"].includes(job.status)) {
        window.clearInterval(pollingTimer);
        pollingTimer = null;
        if (job.status === "completed") {
          const reloadKey = `fermayt-completed-job-reloaded:${job.id}`;
          if (window.sessionStorage.getItem(reloadKey) !== "1") {
            window.sessionStorage.setItem(reloadKey, "1");
            window.location.reload();
          }
        }
      }
    } catch (_) {
      // A later poll can recover from a transient local connection error.
    }
  };

  if (generationActions instanceof HTMLElement && projectForm instanceof HTMLFormElement) {
    const existingJob = generationActions.dataset.jobId;
    const existingJobStatus = generationActions.dataset.jobStatus;
    if (existingJob && ["queued", "running"].includes(existingJobStatus)) {
      pollingTimer = window.setInterval(() => pollJob(existingJob), 1500);
      pollJob(existingJob);
    }
    const startPipeline = async (profile, scope = "FULL", scopeValue = "", overrideBudget = false, retryPlanning = false) => {
      profileButtons.forEach((button) => { if (button instanceof HTMLButtonElement) button.disabled = true; });
      if (progressError) progressError.textContent = "";
      const body = new URLSearchParams(new FormData(projectForm));
      body.set("production_profile", profile);
      body.set("generation_scope_type", scope);
      if (scopeValue !== "") body.set("generation_scope_value", String(scopeValue));
      if (overrideBudget) body.set("budget_override", "1");
      if (retryPlanning) body.set("planning_retry_anyway", "1");
      const response = await fetch(`/api/projects/${generationActions.dataset.projectId}/generate-video`, {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        if (progressBox) progressBox.hidden = false;
        if (progressError) progressError.textContent = payload.detail || "Не удалось запустить pipeline";
        profileButtons.forEach((button) => { if (button instanceof HTMLButtonElement) button.disabled = false; });
        return;
      }
      renderJob(payload);
      generationActions.dataset.jobId = payload.id;
      generationActions.dataset.jobStatus = payload.status;
      generationActions.dataset.jobProfile = payload.production_profile;
      generationActions.dataset.jobScope = payload.generation_scope?.type || "FULL";
      generationActions.dataset.jobScopeValue = payload.generation_scope?.value ?? "";
      if (pollingTimer) window.clearInterval(pollingTimer);
      pollingTimer = window.setInterval(() => pollJob(payload.id), 1500);
    };
    profileButtons.forEach((button) => {
      if (button instanceof HTMLButtonElement) {
        button.addEventListener("click", () => startPipeline(
          button.dataset.profile || "FINAL",
          button.dataset.scope || "FULL",
          button.dataset.scopeValue || "",
          false,
        ));
      }
    });
    if (continueOverBudget instanceof HTMLButtonElement) {
      continueOverBudget.addEventListener("click", () => startPipeline(
        generationActions.dataset.jobProfile || "FINAL",
        generationActions.dataset.jobScope || "FULL",
        generationActions.dataset.jobScopeValue || "",
        true,
      ));
    }
    if (retryPlanningAnyway instanceof HTMLButtonElement) {
      retryPlanningAnyway.addEventListener("click", () => startPipeline(
        generationActions.dataset.jobProfile || "FINAL",
        generationActions.dataset.jobScope || "FULL",
        generationActions.dataset.jobScopeValue || "",
        false,
        true,
      ));
    }
    if (cancelGeneration instanceof HTMLButtonElement) {
      cancelGeneration.addEventListener("click", async () => {
        if (generationActions.dataset.jobStatus === "paused_planning") {
          cancelGeneration.hidden = true;
          return;
        }
        const jobId = generationActions.dataset.jobId;
        if (!jobId) return;
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {method: "POST"});
        if (response.ok) renderJob(await response.json());
      });
    }
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

  const masterAssets = document.querySelector("[data-master-assets]");
  if (masterAssets instanceof HTMLElement) {
    const projectId = masterAssets.dataset.projectId || "";
    masterAssets.querySelectorAll("[data-upload-master]").forEach((element) => {
      if (!(element instanceof HTMLButtonElement)) return;
      element.addEventListener("click", async () => {
        const input = document.getElementById(element.dataset.fileInput || "");
        const file = input instanceof HTMLInputElement ? input.files?.[0] : null;
        if (!file) {
          window.alert("Выберите PNG-файл");
          return;
        }
        element.disabled = true;
        const query = new URLSearchParams({master_scene_id: element.dataset.masterId || ""});
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/master-scenes?${query}`, {
          method: "POST",
          headers: {"Content-Type": "image/png"},
          body: file,
        });
        if (response.ok) window.location.reload();
        else {
          const payload = await response.json();
          window.alert(payload.detail || "Не удалось добавить мастер-картинку");
          element.disabled = false;
        }
      });
    });
    masterAssets.querySelectorAll("[data-delete-master]").forEach((element) => {
      if (!(element instanceof HTMLButtonElement)) return;
      element.addEventListener("click", async () => {
        if (!window.confirm(`Удалить мастер-картинку ${element.dataset.masterId || ""}?`)) return;
        element.disabled = true;
        const response = await fetch(
          `/api/projects/${encodeURIComponent(projectId)}/master-scenes/${encodeURIComponent(element.dataset.assetId || "")}/delete`,
          {method: "POST"},
        );
        if (response.ok) window.location.reload();
        else {
          const payload = await response.json();
          window.alert(payload.detail || "Не удалось удалить мастер-картинку");
          element.disabled = false;
        }
      });
    });
  }

  const promptSheet = document.querySelector("[data-prompt-sheet]");
  const promptDialog = document.getElementById("prompt-detail-dialog");
  if (promptSheet instanceof HTMLElement && promptDialog instanceof HTMLDialogElement) {
    const projectId = promptSheet.dataset.projectId || "";
    const search = promptSheet.querySelector("[data-prompt-search]");
    const filter = promptSheet.querySelector("[data-prompt-filter]");
    const count = promptSheet.querySelector("[data-prompt-count]");
    const providerSelect = promptDialog.querySelector("[data-preview-provider]");
    const modelInput = promptDialog.querySelector("[data-preview-model]");
    const overrideInput = promptDialog.querySelector("[data-override-input]");
    const feedback = promptDialog.querySelector("[data-prompt-feedback]");
    const generateSelected = promptSheet.querySelector("[data-generate-selected]");
    const selectedCount = promptSheet.querySelector("[data-selected-count]");
    const selectedStatus = promptSheet.querySelector("[data-selected-status]");
    const selectAll = promptSheet.querySelector("[data-select-all-beats]");
    let activeTarget = null;
    let activeDetail = null;

    const setText = (selector, value) => {
      const element = promptDialog.querySelector(selector);
      if (element) element.textContent = value ?? "—";
    };
    const makeListItem = (text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    };
    const renderKeyValues = (element, values) => {
      if (!(element instanceof HTMLElement)) return;
      element.replaceChildren();
      Object.entries(values || {}).forEach(([key, value]) => {
        const term = document.createElement("dt");
        term.textContent = key.replaceAll("_", " ");
        const description = document.createElement("dd");
        description.textContent = Array.isArray(value) ? (value.join(", ") || "—") : String(value ?? "—");
        element.append(term, description);
      });
    };
    const formatTransformation = (item) => {
      const before = item.before_length;
      const after = item.after_length;
      const lengths = before !== undefined && after !== undefined ? ` · ${before} → ${after} chars` : "";
      return `${item.type || "TRANSFORMATION"}${lengths}`;
    };
    const renderHistory = (attempts) => {
      const container = promptDialog.querySelector("[data-detail-history]");
      if (!(container instanceof HTMLElement)) return;
      container.replaceChildren();
      if (!attempts?.length) {
        const empty = document.createElement("p");
        empty.className = "helper-text";
        empty.textContent = "No generation attempts yet.";
        container.append(empty);
        return;
      }
      attempts.forEach((attempt) => {
        const snapshot = attempt.prompt_assembly || {};
        const details = document.createElement("details");
        details.className = "prompt-attempt";
        const summary = document.createElement("summary");
        summary.textContent = `Attempt ${attempt.generation_attempt} · ${snapshot.provider || "—"} / ${snapshot.model || "—"} · ${attempt.qa_status || attempt.qa_result?.result || attempt.generation_status || "—"}`;
        const body = document.createElement("div");
        const correction = document.createElement("p");
        correction.textContent = `QA correction: ${snapshot.qa_correction || "—"}`;
        const finalLabel = document.createElement("strong");
        finalLabel.textContent = "Immutable final provider prompt";
        const finalPrompt = document.createElement("pre");
        finalPrompt.textContent = snapshot.final_provider_prompt || "NO PROVIDER PROMPT";
        const asset = document.createElement("p");
        asset.textContent = `Asset: ${attempt.generated_asset || "—"}${attempt.accepted ? " · ACCEPTED" : ""}`;
        body.append(correction, finalLabel, finalPrompt, asset);
        details.append(summary, body);
        container.append(details);
      });
    };
    const renderAssembly = (detail, {full = true} = {}) => {
      activeDetail = {...(activeDetail || {}), ...detail};
      const noProvider = !detail.final_provider_prompt;
      setText("[data-detail-auto]", detail.auto_scene_prompt || "—");
      setText("[data-detail-effective]", detail.effective_scene_prompt || "—");
      setText("[data-detail-operation]", detail.operation || "—");
      setText("[data-detail-operation-instructions]", detail.operation_instructions || "—");
      setText("[data-detail-style]", detail.style_contract_version || "—");
      setText("[data-detail-style-contract]", detail.style_contract_snapshot || "—");
      setText("[data-detail-qa-correction]", detail.qa_correction || "—");
      setText("[data-detail-final]", detail.final_provider_prompt || "NO IMAGE PROVIDER CALL");
      const transformations = promptDialog.querySelector("[data-detail-transformations]");
      if (transformations) {
        transformations.replaceChildren();
        (detail.provider_transformations || []).forEach((item) => transformations.append(makeListItem(formatTransformation(item))));
        if (!(detail.provider_transformations || []).length) transformations.append(makeListItem("None"));
      }
      const references = promptDialog.querySelector("[data-detail-references]");
      if (references) {
        references.replaceChildren();
        (detail.references_used || []).forEach((item) => references.append(makeListItem(`${item.role} · ${item.reference_id}`)));
        if (!(detail.references_used || []).length) references.append(makeListItem("None"));
      }
      promptDialog.querySelectorAll("[data-provider-prompt-fields]").forEach((element) => { element.hidden = noProvider; });
      const noProviderPanel = promptDialog.querySelector("[data-no-provider]");
      if (noProviderPanel) noProviderPanel.hidden = !noProvider;
      if (overrideInput instanceof HTMLTextAreaElement) overrideInput.disabled = noProvider;
      promptDialog.querySelectorAll("[data-edit-override], [data-save-override], [data-clear-override]").forEach((button) => { button.disabled = noProvider; });
      if (!full) return;
      setText("[data-detail-type]", detail.target_type || activeTarget?.type || "TARGET");
      setText("[data-detail-id]", detail.target_id || activeTarget?.id || "—");
      setText("[data-detail-narration]", detail.target_metadata?.narration || "—");
      setText("[data-detail-source]", detail.target_metadata?.source_asset || detail.target_metadata?.source_visual_id || "—");
      setText("[data-detail-master]", detail.target_metadata?.master_scene_id || "—");
      const time = detail.target_metadata?.time_range;
      setText("[data-detail-time]", time ? `${Number(time.start).toFixed(1)}–${Number(time.end).toFixed(1)}s` : "—");
      setText("[data-detail-operation-detail]", detail.target_metadata?.operation_detail || "");
      renderKeyValues(promptDialog.querySelector("[data-detail-semantics]"), detail.semantic_requirement);
      renderHistory(detail.attempts || []);
      const stored = detail.stored_override;
      if (overrideInput instanceof HTMLTextAreaElement) overrideInput.value = stored?.scene_prompt_override || detail.manual_scene_override || "";
      setText("[data-override-state]", stored?.state || detail.override_state || "AUTO");
      const status = promptDialog.querySelector("[data-detail-status]");
      if (status) {
        status.textContent = ["STALE", "REVIEW_REQUIRED"].includes(stored?.state)
          ? `${stored.state.replaceAll("_", " ")}: this override is not active. Save it again to attach it to the current semantic target, or discard it.`
          : "";
        status.hidden = !status.textContent;
      }
      if (providerSelect instanceof HTMLSelectElement) providerSelect.value = detail.provider || promptSheet.dataset.provider || "seedream";
      if (modelInput instanceof HTMLInputElement) modelInput.value = detail.model || promptSheet.dataset.model || "";
      const accepted = promptDialog.querySelector("[data-accepted-candidate]");
      const acceptedImage = promptDialog.querySelector("[data-detail-accepted-image]");
      if (accepted instanceof HTMLElement) accepted.hidden = !detail.accepted_preview_url;
      if (acceptedImage instanceof HTMLImageElement && detail.accepted_preview_url) acceptedImage.src = detail.accepted_preview_url;
      setText("[data-detail-provider-status]", `${detail.provider || "—"}${detail.model ? ` / ${detail.model}` : ""}`);
    };
    const loadDetail = async () => {
      if (!activeTarget) return;
      if (feedback) feedback.textContent = "Loading…";
      const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/prompts/${encodeURIComponent(activeTarget.type)}/${encodeURIComponent(activeTarget.id)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Unable to load prompt detail");
      renderAssembly(payload);
      if (feedback) feedback.textContent = "";
    };
    const updateRows = () => {
      const needle = search instanceof HTMLInputElement ? search.value.trim().toLowerCase() : "";
      const mode = filter instanceof HTMLSelectElement ? filter.value : "ALL";
      let visible = 0;
      promptSheet.querySelectorAll("[data-prompt-row]").forEach((row) => {
        const matchesSearch = !needle || (row.dataset.search || "").toLowerCase().includes(needle);
        const matchesMode = mode === "ALL" || row.dataset.mode === mode;
        row.hidden = !(matchesSearch && matchesMode);
        if (!row.hidden) visible += 1;
      });
      if (count) count.textContent = `${visible} targets`;
    };
    search?.addEventListener("input", updateRows);
    filter?.addEventListener("change", updateRows);
    const openTarget = async (row) => {
      if (!(row instanceof HTMLElement)) return;
      activeTarget = {type: row.dataset.targetType || "", id: row.dataset.targetId || "", row};
      promptDialog.showModal();
      try { await loadDetail(); } catch (error) { if (feedback) feedback.textContent = error.message; }
    };
    promptSheet.querySelectorAll("[data-inspect-prompt]").forEach((button) => {
      button.addEventListener("click", () => openTarget(button.closest("[data-prompt-row]")));
    });
    promptSheet.querySelectorAll("[data-storyboard-target]").forEach((button) => {
      button.addEventListener("click", () => {
        const row = [...promptSheet.querySelectorAll("[data-prompt-row]")].find((candidate) => candidate.dataset.targetType === button.dataset.targetType && candidate.dataset.targetId === button.dataset.targetId);
        openTarget(row);
      });
    });
    const updateSelected = () => {
      const total = promptSheet.querySelectorAll("[data-select-beat]:checked").length;
      if (selectedCount) selectedCount.textContent = String(total);
      if (generateSelected instanceof HTMLButtonElement) generateSelected.disabled = total === 0;
    };
    promptSheet.querySelectorAll("[data-select-beat]").forEach((checkbox) => checkbox.addEventListener("change", updateSelected));
    selectAll?.addEventListener("change", () => {
      promptSheet.querySelectorAll("[data-prompt-row]").forEach((row) => {
        const checkbox = row.querySelector("[data-select-beat]");
        if (checkbox instanceof HTMLInputElement && !row.hidden) checkbox.checked = selectAll.checked;
      });
      updateSelected();
    });
    generateSelected?.addEventListener("click", async () => {
      const beatIds = [...promptSheet.querySelectorAll("[data-select-beat]:checked")].map((item) => item.value);
      if (!beatIds.length) return;
      generateSelected.disabled = true;
      if (selectedStatus) selectedStatus.textContent = "Starting selected visual generation…";
      const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/visual-sheet/generate-selected`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({beat_ids: beatIds}),
      });
      const payload = await response.json();
      if (!response.ok) {
        if (selectedStatus) selectedStatus.textContent = payload.detail || "Unable to generate selected visuals";
        updateSelected();
        return;
      }
      const poll = async () => {
        const jobResponse = await fetch(`/api/jobs/${encodeURIComponent(payload.id)}`);
        const job = await jobResponse.json();
        if (selectedStatus) selectedStatus.textContent = `${job.message || job.current_stage || "Generating"} · ${job.progress || 0}%`;
        if (["queued", "running"].includes(job.status)) {
          window.setTimeout(poll, 1000);
        } else if (job.status === "completed") {
          window.location.reload();
        } else {
          if (selectedStatus) selectedStatus.textContent = job.error || `Generation ${job.status}`;
          updateSelected();
        }
      };
      window.setTimeout(poll, 500);
    });
    promptDialog.querySelector("[data-refresh-preview]")?.addEventListener("click", async () => {
      if (!activeTarget) return;
      const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/prompts/${encodeURIComponent(activeTarget.type)}/${encodeURIComponent(activeTarget.id)}/preview`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({provider: providerSelect?.value, model: modelInput?.value}),
      });
      const payload = await response.json();
      if (!response.ok) { if (feedback) feedback.textContent = payload.detail || "Preview failed"; return; }
      renderAssembly(payload, {full: false});
      if (feedback) feedback.textContent = "Exact preview refreshed.";
    });
    promptDialog.querySelector("[data-generate-video]")?.addEventListener("click", async (event) => {
      if (!activeTarget || activeTarget.type !== "BEAT") {
        if (feedback) feedback.textContent = "Video can only be generated for a visual beat.";
        return;
      }
      const button = event.currentTarget;
      if (button instanceof HTMLButtonElement) button.disabled = true;
      if (feedback) feedback.textContent = "Submitting one paid video task and waiting on its persisted task ID…";
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/visual-sheet/beats/${encodeURIComponent(activeTarget.id)}/generate-video`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({}),
        });
        const payload = await response.json();
        if (!response.ok) {
          const detail = payload.detail;
          throw new Error(typeof detail === "object" ? `${detail.code}: ${detail.message}` : detail || "Video generation failed");
        }
        if (feedback) feedback.textContent = "Video task queued. The paid provider task ID will be persisted before polling.";
        const poll = async () => {
          const jobResponse = await fetch(`/api/jobs/${encodeURIComponent(payload.id)}`);
          const job = await jobResponse.json();
          if (["queued", "running"].includes(job.status)) {
            if (feedback) feedback.textContent = job.message || "Waiting for the video provider…";
            window.setTimeout(poll, 1000);
          } else if (job.status === "completed") {
            window.location.reload();
          } else {
            if (feedback) feedback.textContent = job.error || `Video generation ${job.status}`;
            if (button instanceof HTMLButtonElement) button.disabled = false;
          }
        };
        window.setTimeout(poll, 500);
      } catch (error) {
        if (feedback) feedback.textContent = error.message;
        if (button instanceof HTMLButtonElement) button.disabled = false;
      }
    });
    promptDialog.querySelector("[data-edit-override]")?.addEventListener("click", () => {
      if (overrideInput instanceof HTMLTextAreaElement) {
        overrideInput.focus();
        overrideInput.setSelectionRange(overrideInput.value.length, overrideInput.value.length);
      }
    });
    promptDialog.querySelector("[data-save-override]")?.addEventListener("click", async () => {
      if (!activeTarget || !(overrideInput instanceof HTMLTextAreaElement)) return;
      const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/prompts/${encodeURIComponent(activeTarget.type)}/${encodeURIComponent(activeTarget.id)}/override`, {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({scene_prompt_override: overrideInput.value}),
      });
      const payload = await response.json();
      if (!response.ok) { if (feedback) feedback.textContent = payload.detail || "Save failed"; return; }
      renderAssembly(payload.prompt);
      activeTarget.row.dataset.mode = "OVERRIDE";
      const mode = activeTarget.row.querySelector("[data-row-mode]");
      if (mode) { mode.textContent = "OVERRIDE"; mode.className = "prompt-mode prompt-mode-override"; }
      if (feedback) feedback.textContent = "Override saved. Image generation was not started.";
      updateRows();
    });
    promptDialog.querySelector("[data-clear-override]")?.addEventListener("click", async () => {
      if (!activeTarget) return;
      const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/prompts/${encodeURIComponent(activeTarget.type)}/${encodeURIComponent(activeTarget.id)}/override`, {method: "DELETE"});
      const payload = await response.json();
      if (!response.ok) { if (feedback) feedback.textContent = payload.detail || "Clear failed"; return; }
      await loadDetail();
      activeTarget.row.dataset.mode = "AUTO";
      const mode = activeTarget.row.querySelector("[data-row-mode]");
      if (mode) { mode.textContent = "AUTO"; mode.className = "prompt-mode prompt-mode-auto"; }
      if (feedback) feedback.textContent = "Override cleared. AUTO prompt restored.";
      updateRows();
    });
  }
});
