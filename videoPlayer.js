document.addEventListener('DOMContentLoaded', () => {
function formatTimeWithMilliseconds(timeInSeconds) {
  const minutes = Math.floor(timeInSeconds / 60).toString().padStart(2, '0');
  const seconds = Math.floor(timeInSeconds % 60).toString().padStart(2, '0');
  const milliseconds = Math.floor((timeInSeconds % 1) * 1000).toFixed(0).toString().padStart(3, '0');
  return `${minutes}:${seconds}.${milliseconds}`;
}

const videoFileInput = document.getElementById('videoInput');
const videoPlayer = document.getElementById('videoPlayer');
const controls = document.getElementById('controls');
const playPauseBtn = document.getElementById('playPauseBtn');
const seekBar = document.getElementById('seekBar');
const timeDisplay = document.getElementById('timeDisplay');
const message = document.getElementById('message');

let animationId;

videoFileInput.addEventListener('change', () => {
  const files = videoFileInput.files;
  if (!files.length) return;

  const file = files[0];
  const url = URL.createObjectURL(file);

  videoPlayer.src = url;
  videoPlayer.load();
  videoPlayer.style.display = 'block';
  controls.style.display = 'inline-flex';
  message.textContent = '';

  videoPlayer.addEventListener('loadedmetadata', () => {
    seekBar.max = videoPlayer.duration;
    seekBar.value = 0;
    timeDisplay.textContent = `${formatTimeWithMilliseconds(0)} / ${formatTimeWithMilliseconds(videoPlayer.duration)}`;
    playPauseBtn.textContent = 'Play';
  });
});

playPauseBtn.addEventListener('click', () => {
  if (videoPlayer.paused || videoPlayer.ended) {
    videoPlayer.play();
    playPauseBtn.textContent = 'Pause';
    message.textContent = '';
    startUpdatingTime();
  } else {
    videoPlayer.pause();
    playPauseBtn.textContent = 'Play';
    stopUpdatingTime();
  }
});

seekBar.addEventListener('input', () => {
  videoPlayer.pause();
  playPauseBtn.textContent = 'Play';
  videoPlayer.currentTime = seekBar.value;
  updateTimeDisplay(videoPlayer.currentTime, videoPlayer.duration);
  stopUpdatingTime();
});

videoPlayer.addEventListener('ended', () => {
  playPauseBtn.textContent = 'Play';
  stopUpdatingTime();
});

function updateTimeDisplay(current, duration) {
  timeDisplay.textContent = `${formatTimeWithMilliseconds(current)} / ${formatTimeWithMilliseconds(duration)}`;
}

function updateSeekBar() {
  seekBar.value = videoPlayer.currentTime;
  updateTimeDisplay(videoPlayer.currentTime, videoPlayer.duration);
  animationId = requestAnimationFrame(updateSeekBar);
}

function startUpdatingTime() {
  if (!animationId) {
    animationId = requestAnimationFrame(updateSeekBar);
  }
}

function stopUpdatingTime() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

videoPlayer.addEventListener('play', () => {
  playPauseBtn.textContent = 'Pause';
  startUpdatingTime();
});

videoPlayer.addEventListener('pause', () => {
  playPauseBtn.textContent = 'Play';
  stopUpdatingTime();
});
});
