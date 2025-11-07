# REST API - Инструкция

## Запуск сервера

### 1. Установите Flask
```bash
pip install flask flask-cors
```

### 2. Запустите сервер
```bash
cd FOR_GITHUB/coach-assistant
python api_server.py
```

**Сервер запустится на:** `http://localhost:5000`

---

## Использование с фронтенда

### 1. Проверка что сервер работает
```javascript
fetch('http://localhost:5000/health')
    .then(res => res.json())
    .then(data => console.log(data));

// Ответ: {"status": "ok", "message": "Server is running"}
```

### 2. Инициализация модели (один раз при старте)
```javascript
fetch('http://localhost:5000/init', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model_path: 'best_model.pth',
        metadata_path: 'metadata.json'
    })
})
.then(res => res.json())
.then(data => console.log(data));

// Ответ: {"success": true, "message": "Model loaded successfully"}
```

### 3. Классификация одного видео
```javascript
fetch('http://localhost:5000/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        poses_file: 'outputs/my_video/poses.jsonl'
    })
})
.then(res => res.json())
.then(data => {
    console.log(data.predicted_figure);  // "Fan"
    console.log(data.confidence);         // 0.63
});

// Ответ:
// {
//     "success": true,
//     "predicted_figure": "Fan",
//     "confidence": 0.63
// }
```

### 4. Классификация нескольких видео
```javascript
fetch('http://localhost:5000/classify_multiple', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        poses_files: [
            'outputs/video1/poses.jsonl',
            'outputs/video2/poses.jsonl'
        ]
    })
})
.then(res => res.json())
.then(data => console.log(data.results));

// Ответ:
// {
//     "results": [
//         {
//             "file": "outputs/video1/poses.jsonl",
//             "success": true,
//             "predicted_figure": "Fan",
//             "confidence": 0.63
//         },
//         {
//             "file": "outputs/video2/poses.jsonl",
//             "success": true,
//             "predicted_figure": "Alemana",
//             "confidence": 0.78
//         }
//     ]
// }
```

---

## Примеры для разных фреймворков

### React
```jsx
async function classifyVideo(posesFile) {
    const response = await fetch('http://localhost:5000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ poses_file: posesFile })
    });
    const data = await response.json();
    return data;
}

// Использование
const result = await classifyVideo('outputs/my_video/poses.jsonl');
console.log(`Движение: ${result.predicted_figure} (${result.confidence * 100}%)`);
```

### Vue.js
```javascript
async classifyVideo(posesFile) {
    const response = await axios.post('http://localhost:5000/classify', {
        poses_file: posesFile
    });
    return response.data;
}

// Использование
const result = await this.classifyVideo('outputs/my_video/poses.jsonl');
this.movement = result.predicted_figure;
this.confidence = result.confidence;
```

### Vanilla JavaScript
```javascript
function classifyVideo(posesFile) {
    return fetch('http://localhost:5000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ poses_file: posesFile })
    })
    .then(res => res.json());
}

// Использование
classifyVideo('outputs/my_video/poses.jsonl')
    .then(result => {
        document.getElementById('movement').textContent = result.predicted_figure;
        document.getElementById('confidence').textContent = (result.confidence * 100).toFixed(1) + '%';
    });
```

---

## Endpoints

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/health` | Проверка работы сервера |
| POST | `/init` | Инициализация модели |
| POST | `/classify` | Классификация одного видео |
| POST | `/classify_multiple` | Классификация нескольких видео |

---

## Обработка ошибок

```javascript
fetch('http://localhost:5000/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ poses_file: 'wrong_path.jsonl' })
})
.then(res => res.json())
.then(data => {
    if (data.success) {
        console.log('Движение:', data.predicted_figure);
    } else {
        console.error('Ошибка:', data.error);
    }
});
```

---

## Для продакшена

### 1. Отключите debug режим
В `api_server.py` измените:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

### 2. Используйте gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

### 3. Настройте CORS для конкретного домена
В `api_server.py`:
```python
CORS(app, origins=['https://yourdomain.com'])
```

---

## Готово!

Сервер готов к использованию! 🚀

