import { FFmpeg } from '/node_modules/@ffmpeg/ffmpeg/dist/esm/index.js';
import { fetchFile } from '/node_modules/@ffmpeg/util/dist/esm/index.js';

export async function trimVideo(blob, startTime, endTime) {
  console.log(blob)
  const ffmpeg = new FFmpeg();
  await ffmpeg.load();
  await ffmpeg.writeFile('input.mp4', await fetchFile(blob));
  await ffmpeg.exec([
   '-ss', startTime.toFixed(3),
  '-i', 'input.mp4',
  '-to', endTime.toFixed(3),
  '-c:v', 'copy',              
  '-c:a', 'copy',             
  '-avoid_negative_ts', 'make_zero',
  '-copyts',         
  'output_lossless.mp4'
  ]);
  const data = await ffmpeg.readFile('output_lossless.mp4');
  return new Blob([data.buffer], { type: 'video/mp4' });
}