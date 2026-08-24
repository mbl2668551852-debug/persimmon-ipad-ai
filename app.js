"use strict";

const SIZE = 640;
const IOU_THRESHOLD = 0.45;
const KEYPOINT_THRESHOLD = 0.50;
const video = document.querySelector("#camera");
const overlay = document.querySelector("#overlay");
const ctx = overlay.getContext("2d");
const inputCanvas = document.querySelector("#inputCanvas");
const inputCtx = inputCanvas.getContext("2d", { willReadFrequently: true });
const statusEl = document.querySelector("#status");
const fpsEl = document.querySelector("#fps");
const emptyEl = document.querySelector("#empty");
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const confidence = document.querySelector("#confidence");
const confidenceValue = document.querySelector("#confidenceValue");

let session = null;
let stream = null;
let running = false;
let busy = false;
let lastTick = 0;

ort.env.wasm.wasmPaths = "./";
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

confidence.addEventListener("input", () => confidenceValue.value = Number(confidence.value).toFixed(2));
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
window.addEventListener("resize", resizeOverlay);

async function start() {
  try {
    startButton.disabled = true;
    statusEl.textContent = session ? "正在打开后置摄像头…" : "正在加载识别模型（首次约 10 MB）…";
    if (!session) {
      session = await ort.InferenceSession.create("best.onnx", {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
    }
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    emptyEl.hidden = true;
    running = true;
    stopButton.disabled = false;
    statusEl.textContent = "本机识别中 · 未上传画面";
    requestAnimationFrame(loop);
  } catch (error) {
    startButton.disabled = false;
    statusEl.textContent = friendlyError(error);
  }
}

function stop() {
  running = false;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  emptyEl.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  fpsEl.textContent = "-- FPS";
  statusEl.textContent = "摄像头已停止";
}

function resizeOverlay() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

async function loop(now) {
  if (!running) return;
  if (!busy && video.readyState >= 2) {
    busy = true;
    const began = performance.now();
    try { await infer(); }
    catch (error) { statusEl.textContent = `识别错误：${error.message}`; }
    busy = false;
    const elapsed = performance.now() - began;
    fpsEl.textContent = `${(1000 / elapsed).toFixed(1)} FPS`;
  }
  lastTick = now;
  requestAnimationFrame(loop);
}

async function infer() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(SIZE / vw, SIZE / vh);
  const drawW = Math.round(vw * scale), drawH = Math.round(vh * scale);
  const padX = (SIZE - drawW) / 2, padY = (SIZE - drawH) / 2;
  inputCtx.fillStyle = "#727272";
  inputCtx.fillRect(0, 0, SIZE, SIZE);
  inputCtx.drawImage(video, 0, 0, vw, vh, padX, padY, drawW, drawH);

  const rgba = inputCtx.getImageData(0, 0, SIZE, SIZE).data;
  const data = new Float32Array(3 * SIZE * SIZE);
  const plane = SIZE * SIZE;
  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    data[p] = rgba[i] / 255;
    data[plane + p] = rgba[i + 1] / 255;
    data[2 * plane + p] = rgba[i + 2] / 255;
  }

  const inputName = session.inputNames[0];
  const outputs = await session.run({ [inputName]: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]) });
  const output = outputs[session.outputNames[0]];
  const detections = decode(output.data, output.dims, Number(confidence.value));
  draw(nms(detections), scale, padX, padY, vw, vh);
}

function decode(data, dims, threshold) {
  const channels = dims[1], count = dims[2];
  if (channels < 11) throw new Error(`模型输出格式不符：${dims.join("×")}`);
  const at = (c, i) => data[c * count + i];
  const found = [];
  for (let i = 0; i < count; i++) {
    const score = at(4, i);
    if (score < threshold) continue;
    const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    found.push({
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score,
      calyx: { x: at(5, i), y: at(6, i), score: at(7, i) },
      tip: { x: at(8, i), y: at(9, i), score: at(10, i) }
    });
  }
  return found.sort((a, b) => b.score - a.score);
}

function nms(items) {
  const kept = [];
  for (const item of items) {
    if (kept.every(other => iou(item, other) < IOU_THRESHOLD)) kept.push(item);
  }
  return kept;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / (areaA + areaB - intersection + 1e-6);
}

function draw(items, scale, padX, padY, vw, vh) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const map = p => ({ x: (p.x - padX) / scale, y: (p.y - padY) / scale });
  ctx.font = `${Math.max(18, vw / 45)}px -apple-system, sans-serif`;
  ctx.lineJoin = "round";
  for (const item of items) {
    const a = map({x:item.x1,y:item.y1}), b = map({x:item.x2,y:item.y2});
    ctx.strokeStyle = "#ff8a2a";
    ctx.lineWidth = Math.max(3, vw / 300);
    ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
    ctx.fillStyle = "#ff8a2a";
    ctx.fillText(`柿子 ${(item.score * 100).toFixed(0)}%`, a.x + 4, Math.max(24, a.y - 7));
    if (item.calyx.score >= KEYPOINT_THRESHOLD && item.tip.score >= KEYPOINT_THRESHOLD) {
      const c = map(item.calyx), t = map(item.tip);
      arrow(c.x, c.y, t.x, t.y);
      point(c.x, c.y, "#4cff70", "柿蒂");
      point(t.x, t.y, "#26d7ff", "柿尖");
    }
  }
}

function point(x, y, color, label) {
  ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  ctx.fillStyle = color; ctx.fillText(label, x + 10, y - 8);
}

function arrow(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1), head = 22;
  ctx.strokeStyle = "#ffe45c"; ctx.fillStyle = "#ffe45c"; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
  ctx.closePath(); ctx.fill();
}

function friendlyError(error) {
  if (!window.isSecureContext) return "必须通过 HTTPS 打开，Safari 才允许使用摄像头";
  if (error?.name === "NotAllowedError") return "摄像头权限被拒绝，请到 iPad 设置中允许 Safari 使用摄像头";
  return `启动失败：${error?.message || error}`;
}

if ("serviceWorker" in navigator) navigator.serviceWorker.register("service-worker.js");
