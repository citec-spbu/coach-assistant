const Koa = require('koa');
const path = require('path');
const serve = require('koa-static');
const Router = require('@koa/router');
const fs = require('fs-extra');
const { koaBody } = require('koa-body');

const staticDirPath = path.join(__dirname, '');
const nodeModulesDirPath = path.join(__dirname, 'node_modules');
const uploadDir = path.join(__dirname, 'uploads');

// Проверка существования 'uploads'
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

const server = new Koa();

server.use(koaBody({
  multipart: true,
  formidable: {
    uploadDir: uploadDir,
    keepExtensions: true,
    maxFileSize: 100 * 1024 * 1024, // 100 MB
  }
}));

const router = new Router();

// Для хранения информации о статусах видео
const videoStatusMap = new Map();

// Роут для загрузки видео
router.post('/upload/:filename', async (ctx) => {
  try {
    const { filename } = ctx.params;
    const file = ctx.request.files?.video;

    if (!file) {
      ctx.status = 400;
      ctx.body = { error: 'Файл не найден' };
      return;
    }

    console.log('Получен файл:', file.originalFilename || file.newFilename);
    console.log('Размер:', file.size, 'bytes');
    console.log('MIME:', file.mimetype);

    // Проверка типа файла
    if (!file.mimetype.startsWith('video/')) {
      await fs.remove(file.filepath);
      ctx.status = 400;
      ctx.body = { error: 'Файл должен быть видео' };
      return;
    }

    const destPath = path.join(uploadDir, filename);

    // Перемещение файла из временного хранилища
    await fs.move(file.filepath, destPath, { overwrite: true });

    console.log('Файл загружен:', destPath);
    
    
    const videoUrl = `http://localhost:3000/uploads/${filename}`;

    // Отправка URL видео на FastAPI для обработки
    try {
      const fastapiResponse = await axios.post('http://fastapi-server:5000/api/send', { url: videoUrl });
      console.log('FastAPI ответил:', fastapiResponse.data);
    } catch (error) {
      console.error('Ошибка отправки видео в FastAPI:', error.message);
    }

    ctx.status = 200;
    ctx.body = {
      success: true,
      filename: filename,
      size: file.size,
      path: `/uploads/${filename}`
    };

  } catch (error) {
    console.error('Ошибка загрузки:', error);
    ctx.status = 500;
    ctx.body = { error: error.message };
  }
});

// Прием URL видео от FastAPI, начало обработки
router.post('/api/send', async (ctx) => {
  const videoUrl = ctx.request.body.url;
  if (!videoUrl) {
    ctx.status = 400;
    ctx.body = { error: 'URL видео обязателен' };
    return;
  }

  videoStatusMap.set(videoUrl, { status: 'processing', download_url: null });

  ctx.status = 200;
  ctx.body = { message: 'Видео получено и в процессе обработки' };
});

// Получение статуса и информации по видео
router.get('/api/get', async (ctx) => {
  const videoUrl = ctx.query.url;
  if (!videoUrl) {
    ctx.status = 400;
    ctx.body = { error: 'URL видео обязателен' };
    return;
  }

  const info = videoStatusMap.get(videoUrl);
  if (!info) {
    ctx.status = 404;
    ctx.body = { error: 'Информация по видео не найдена' };
    return;
  }

  ctx.status = 200;
  ctx.body = info;
});

// Получение результата обработки с FastAPI
router.post('/api/result', async (ctx) => {
  const { status, upload_url, download_url } = ctx.request.body;
  if (!status || !upload_url) {
    ctx.status = 400;
    ctx.body = { error: 'Status и upload_url обязательны' };
    return;
  }

  videoStatusMap.set(upload_url, { status, download_url: download_url || null });

  ctx.status = 200;
  ctx.body = { message: 'Результат обработки принят' };
});

router.get('/result/:filename', async ctx => {});

server.use(router.routes()).use(router.allowedMethods());

server.use(serve(staticDirPath));
server.use(serve(nodeModulesDirPath));

const PORT = 3000;
server.listen(PORT, () => console.log(`Server Listening on PORT ${PORT} ..`));
