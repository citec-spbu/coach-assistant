const Koa = require('koa');
const path = require('path');
const serve = require('koa-static');
const Router = require('koa-router');

const staticDirPath = path.join(__dirname, '');
const nodeModulesDirPath = path.join(__dirname, 'node_modules');

const server = new Koa();

const router = new Router();

router.post('/upload/:filename', async ctx => {});

router.get('/result/:filename', async ctx => {});

server.use(router.routes()).use(router.allowedMethods());

server.use(serve(staticDirPath));
server.use(serve(nodeModulesDirPath));

const PORT = 3000;
server.listen(PORT, () => console.log(`Server Listening on PORT ${PORT} ..`));
