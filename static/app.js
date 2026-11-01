const video = document.querySelector("#camera");
const placeholder = document.querySelector("#camera-placeholder");
const cameraButton = document.querySelector("#camera-button");
const compareButton = document.querySelector("#compare-button");
const statusMessage = document.querySelector("#status");
const resultsSection = document.querySelector("#results-section");
const resultsList = document.querySelector("#results");
const captureCanvas = document.querySelector("#capture");
const MAX_CAPTURE_DIMENSION = 640;

let cameraStream;

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("is-error", isError);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("카메라를 사용하려면 HTTPS 주소나 localhost로 접속해 주세요.", true);
    return;
  }

  cameraButton.disabled = true;
  setStatus("카메라 권한을 기다리는 중이에요.");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = cameraStream;
    placeholder.hidden = true;
    compareButton.disabled = false;
    cameraButton.textContent = "카메라 켜짐";
    setStatus("준비됐어요. 비교하기를 누르면 한 장을 촬영합니다.");
  } catch (error) {
    cameraButton.disabled = false;
    const message = error.name === "NotAllowedError"
      ? "브라우저에서 카메라 권한을 허용해 주세요."
      : "카메라를 시작하지 못했어요. 다른 앱이 사용 중인지 확인해 주세요.";
    setStatus(message, true);
  }
}

function captureFrame() {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error("카메라 화면이 준비될 때까지 잠시 기다려 주세요.");
  }

  const scale = Math.min(
    1,
    MAX_CAPTURE_DIMENSION / Math.max(video.videoWidth, video.videoHeight),
  );
  captureCanvas.width = Math.round(video.videoWidth * scale);
  captureCanvas.height = Math.round(video.videoHeight * scale);
  const context = captureCanvas.getContext("2d");
  context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  return new Promise((resolve, reject) => {
    captureCanvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("사진을 캡처하지 못했어요. 다시 시도해 주세요."));
      }
    }, "image/jpeg", 0.9);
  });
}

function showResults(recommendations) {
  resultsList.replaceChildren();
  for (const [index, recommendation] of recommendations.entries()) {
    const item = document.createElement("li");
    const rank = document.createElement("span");
    const image = document.createElement("img");
    const name = document.createElement("span");
    const score = document.createElement("span");

    rank.className = "result-rank";
    rank.textContent = index === 0 ? "가장 닮은 캐릭터" : `추천 ${index + 1}`;
    image.className = "result-image";
    image.src = recommendation.image_url;
    image.alt = `${recommendation.character} 참고 이미지`;
    image.loading = index === 0 ? "eager" : "lazy";
    name.className = "result-name";
    name.textContent = recommendation.character;
    score.className = "result-score";
    score.textContent = recommendation.score.toFixed(2);

    const details = document.createElement("div");
    details.className = "result-details";
    details.append(rank, name, score);
    item.append(image, details);
    resultsList.append(item);
  }
  resultsSection.hidden = false;
}

async function comparePhoto() {
  compareButton.disabled = true;
  cameraButton.disabled = true;
  setStatus("사진을 비교하고 있어요. 처음 실행하면 모델 준비로 시간이 걸릴 수 있습니다.");
  try {
    const photo = await captureFrame();
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: photo,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "비교에 실패했어요.");
    }
    showResults(payload.recommendations);
    setStatus("비교가 끝났어요. 다른 사진으로 다시 비교할 수 있습니다.");
  } catch (error) {
    setStatus(error.message || "비교 중 문제가 생겼어요.", true);
  } finally {
    compareButton.disabled = !cameraStream;
    cameraButton.disabled = Boolean(cameraStream);
  }
}

cameraButton.addEventListener("click", startCamera);
compareButton.addEventListener("click", comparePhoto);
window.addEventListener("pagehide", () => {
  cameraStream?.getTracks().forEach((track) => track.stop());
});
