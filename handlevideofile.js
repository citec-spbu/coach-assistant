//Функция проверят правильное ли разрешение у видео и запускает его если разрешение правильное
function handleVideoFile(files) {
  if (!files.length) return;
  const file = files[0];
  const videoPlayer = document.getElementById('videoPlayer');
  const message = document.getElementById('message');
  const url = URL.createObjectURL(file);

  const tmpVideo = document.createElement('video');
  tmpVideo.src = url;

  tmpVideo.addEventListener('loadedmetadata', () => {
    if (tmpVideo.videoWidth === 1920 && tmpVideo.videoHeight === 1080) {
      videoPlayer.src = url;
      videoPlayer.muted = true;
      videoPlayer.style.display = 'block';
      videoPlayer.load();

      videoPlayer.oncanplay = () => {
        videoPlayer.play().catch(err => {
          console.warn("Не удалось запустить видео автоматически:", err);
          message.textContent = "Видео готово к воспроизведению. Нажмите Play для запуска.";
        });
      };

      message.textContent = "Видео загружено.";
      console.log('Duration', tmpVideo.duration);
      document.getElementById('timePickerEnd').timePicker.setTime(tmpVideo.duration);
    } else {
      alert(`Неправильное разрешение видео: ${tmpVideo.videoWidth}x${tmpVideo.videoHeight}px. Требуется 1920x1080px.`);
      message.textContent = "Ошибка: неправильное разрешение видео.";
      videoPlayer.style.display = 'none';
      URL.revokeObjectURL(url);
    }
  });

  tmpVideo.addEventListener('error', () => {
    alert("Ошибка при загрузке видео.");
    message.textContent = "Ошибка загрузки видео.";
    URL.revokeObjectURL(url);
  });
}
