"use strict";
const $ = id => document.getElementById(id);
const input = $("file-input"), dropZone = $("drop-zone"), inspectButton = $("inspect-button");
let selectedFile = null, previewURL = null, result = null, busy = false, requestGeneration = 0;
let controller = null, ready = false;
const exampleButtons = [...document.querySelectorAll("[data-example]")];
function setError(message) { $("error-message").textContent = message || ""; $("error-message").hidden = !message; }
function updateButton() { inspectButton.disabled = busy || !selectedFile || !ready; }
function setBusy(value) {
 busy = value; $("processing-overlay").hidden = !value;
 $("inspect-label").textContent = value ? "Inspecting image…" : "Inspect image";
 $("result-status").textContent = value ? "Inspecting…" : result ? "Complete" : "Awaiting image";
 dropZone.setAttribute("aria-disabled", String(value)); $("remove-file").disabled = value;
 exampleButtons.forEach(button => button.disabled = value); updateButton();
}
function clearResult() { result = null; $("result-content").hidden = true; $("empty-result").hidden = false; $("result-status").textContent = "Awaiting image"; $("result-status").classList.remove("complete"); }
function reset() {
 requestGeneration++; if (controller) controller.abort(); if (previewURL) URL.revokeObjectURL(previewURL);
 previewURL = null; selectedFile = null; input.value = "";
 $("image-preview").hidden = true; $("image-preview").removeAttribute("src");
 $("empty-upload").hidden = false; $("file-info").hidden = true; $("preview-shade").hidden = true;
 dropZone.classList.remove("has-image");
 exampleButtons.forEach(button => { button.classList.remove("selected"); button.setAttribute("aria-pressed", "false"); });
 clearResult(); setError(""); setBusy(false);
}
async function selectFile(file) {
 if (busy || !file) return false;
 reset();
 if (!["image/jpeg", "image/png"].includes(file.type)) { setError("Please choose a JPG or PNG image."); return false; }
 if (!file.size || file.size > 8 * 1024 * 1024) { setError("Choose an image smaller than 8 MB. Empty files cannot be inspected."); return false; }
 const generation = requestGeneration, url = URL.createObjectURL(file), image = new Image();
 try {
  image.src = url; await image.decode();
  if (generation !== requestGeneration) { URL.revokeObjectURL(url); return false; }
  if (Math.min(image.naturalWidth, image.naturalHeight) < 32 || image.naturalWidth * image.naturalHeight > 16000000) {
   URL.revokeObjectURL(url); setError("Use an image at least 32 × 32 pixels and no larger than 16 megapixels."); return false;
  }
  selectedFile = file; previewURL = url; $("image-preview").src = url; $("image-preview").hidden = false;
  $("empty-upload").hidden = true; $("preview-shade").hidden = false; dropZone.classList.add("has-image");
  $("file-info").hidden = false; $("file-name").textContent = file.name;
  $("file-size").textContent = `${image.naturalWidth} × ${image.naturalHeight} px · ${file.size < 1048576 ? (file.size / 1024).toFixed(1) + " KB" : (file.size / 1048576).toFixed(2) + " MB"}`;
  updateButton(); return true;
 } catch (_) { URL.revokeObjectURL(url); if (generation === requestGeneration) setError("This image could not be opened. Try a different JPG or PNG file."); return false; }
}
function showResult(data) {
 result = data; const defective = data.predicted_class === "Defective";
 $("empty-result").hidden = true; $("result-content").hidden = false;
 $("predicted-class").textContent = data.predicted_class;
 $("confidence-value").textContent = (data.confidence * 100).toFixed(1);
 $("defect-score").textContent = (data.defect_probability * 100).toFixed(1) + "%";
 $("score-fill").style.width = (data.defect_probability * 100) + "%";
 $("score-fill").classList.toggle("defective", defective);
 $("threshold-marker").style.left = `calc(${data.threshold * 100}% - 1px)`;
 $("threshold-label").textContent = `Threshold ${(data.threshold * 100).toFixed(0)}%`;
 $("verdict-icon").classList.toggle("defective", defective);
 $("verdict-symbol").setAttribute("href", defective ? "#alert" : "#check");
 $("latency-value").textContent = `${Math.round(data.latency_ms)} ms`;
 $("result-description").textContent = defective ? "The model flagged this casting as potentially defective. Review the image before deciding what to reject or rework." : "The model classified this casting as normal. Small or unfamiliar defects can still be missed.";
 $("raw-result").textContent = JSON.stringify(data, null, 2);
 $("result-status").textContent = "Complete"; $("result-status").classList.add("complete");
}
async function inspect() {
 if (!selectedFile || busy || !ready) return;
 clearResult(); setError(""); setBusy(true);
 const generation = requestGeneration; controller = new AbortController();
 const timeout = setTimeout(() => controller?.abort(), 30000);
 const form = new FormData(); form.append("file", selectedFile, selectedFile.name);
 try {
  const response = await fetch("/predict", {method:"POST", body:form, signal:controller.signal});
  const data = await response.json(); if (generation !== requestGeneration) return;
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The image could not be inspected. Please try another file.");
  showResult(data);
 } catch (error) {
  if (generation !== requestGeneration) return;
  setError(error.name === "AbortError" ? "The inspection timed out. Check that the local model service is running, then try again." : error.message === "Failed to fetch" ? "The model service could not be reached. Check your server window and try again." : error.message);
 } finally { clearTimeout(timeout); if (generation === requestGeneration) { controller = null; setBusy(false); } }
}
async function refreshStatus() {
 try {
  const response = await fetch("/ui/info", {cache:"no-store", signal:AbortSignal.timeout(8000)});
  if (!response.ok) throw new Error("unavailable");
  const data = await response.json(); ready = data.ready;
  $("service-state").classList.toggle("offline", !ready); $("service-text").textContent = ready ? "Model ready" : "Model unavailable";
  $("accuracy-value").textContent = data.test_accuracy == null ? "—" : (data.test_accuracy * 100).toFixed(2) + "%";
  $("test-count").textContent = data.test_images == null ? "—" : data.test_images.toLocaleString();
 } catch (_) { ready = false; $("service-state").classList.add("offline"); $("service-text").textContent = "Service offline"; }
 updateButton();
}
dropZone.addEventListener("click", () => { if (!busy) input.click(); });
dropZone.addEventListener("keydown", event => { if (["Enter"," "].includes(event.key)) {event.preventDefault(); if (!busy) input.click();} });
input.addEventListener("change", () => selectFile(input.files[0]));
["dragenter","dragover"].forEach(name => dropZone.addEventListener(name, event => {event.preventDefault(); if (!busy) dropZone.classList.add("dragover");}));
["dragleave","drop"].forEach(name => dropZone.addEventListener(name, event => {event.preventDefault(); dropZone.classList.remove("dragover");}));
dropZone.addEventListener("drop", event => {if (!busy) selectFile(event.dataTransfer.files[0]);});
window.addEventListener("dragover", event => event.preventDefault()); window.addEventListener("drop", event => event.preventDefault());
$("remove-file").addEventListener("click", reset); $("reset-button").addEventListener("click", () => {reset(); dropZone.focus();});
inspectButton.addEventListener("click", inspect);
exampleButtons.forEach(button => {
 button.setAttribute("aria-pressed", "false");
 button.addEventListener("click", async () => {
  if (busy) return; setError("");
  try {
   const response = await fetch("/assets/examples/" + button.dataset.example);
   if (!response.ok) throw new Error("Example unavailable. Try uploading your own casting image.");
   const file = new File([await response.blob()], button.dataset.example, {type:"image/jpeg"});
   if (await selectFile(file)) {button.classList.add("selected"); button.setAttribute("aria-pressed", "true"); await inspect();}
  } catch (error) {setError(error.message);}
 });
});
$("download-result").addEventListener("click", () => {
 if (!result) return;
 const blob = new Blob([JSON.stringify({...result,input_filename:selectedFile?.name},null,2)], {type:"application/json"});
 const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href = url;
 link.download = "casting-inspection-" + result.request_id.slice(0,8) + ".json"; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
const dialog = $("help-dialog");
$("help-button").addEventListener("click", () => dialog.showModal());
$("mobile-help").addEventListener("click", () => dialog.showModal());
$("close-help").addEventListener("click", () => dialog.close()); $("help-done").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => {if (event.target === dialog) {const r=dialog.getBoundingClientRect(); if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom) dialog.close();}});
window.addEventListener("beforeunload", () => {if(previewURL) URL.revokeObjectURL(previewURL);});
refreshStatus(); setInterval(refreshStatus,15000);
