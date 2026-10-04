import { Queue, Worker, Job } from 'bullmq';
import IORedis from 'ioredis';
import fs from 'fs';
import FormData from 'form-data';
import fetch from 'node-fetch';

const REDIS_HOST = process.env.REDIS_HOST || '127.0.0.1';
const REDIS_PORT = parseInt(process.env.REDIS_PORT || '6379', 10);
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export const redisConnection = new IORedis({
  host: REDIS_HOST,
  port: REDIS_PORT,
  maxRetriesPerRequest: null, // Required by BullMQ
  retryStrategy(times) {
    const delay = Math.min(times * 200, 2000);
    return delay;
  },
});

redisConnection.on('error', (err) => {
  console.error('[Redis Error]', err.message);
});

redisConnection.on('connect', () => {
  console.log(`[Redis] Connected to ${REDIS_HOST}:${REDIS_PORT}`);
});

export const INFERENCE_QUEUE_NAME = 'medical-inference';

export const inferenceQueue = new Queue(INFERENCE_QUEUE_NAME, {
  connection: redisConnection,
  defaultJobOptions: {
    attempts: 2,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
    removeOnComplete: {
      age: 3600, // keep completed jobs for 1 hour
      count: 100,
    },
    removeOnFail: {
      age: 3600,
      count: 100,
    },
  },
});

export interface JobData {
  filePath: string;
  originalName: string;
  fileSize: number;
}

export const inferenceWorker = new Worker<JobData>(
  INFERENCE_QUEUE_NAME,
  async (job: Job<JobData>) => {
    console.log(`[Worker] Starting job ${job.id} for file: ${job.data.originalName}`);
    await job.updateProgress(10);

    const { filePath, originalName } = job.data;

    try {
      if (!fs.existsSync(filePath)) {
        throw new Error(`File at ${filePath} not found`);
      }

      await job.updateProgress(25);
      console.log(`[Worker] Forwarding ${originalName} to ML service at ${ML_SERVICE_URL}/infer...`);

      const form = new FormData();
      form.append('file', fs.createReadStream(filePath), {
        filename: originalName,
      });

      await job.updateProgress(40);

      const response = await fetch(`${ML_SERVICE_URL}/infer`, {
        method: 'POST',
        body: form as any,
        headers: form.getHeaders(),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`ML Service responded with HTTP ${response.status}: ${errorText}`);
      }

      await job.updateProgress(85);
      const result = await response.json();
      console.log(`[Worker] Job ${job.id} finished successfully! Bone quality:`, (result as any).bone_quality?.misch_class);

      await job.updateProgress(100);
      return result;
    } finally {
      // Clean up uploaded temp file
      if (fs.existsSync(filePath)) {
        try {
          fs.unlinkSync(filePath);
          console.log(`[Worker] Cleaned up temporary file: ${filePath}`);
        } catch (cleanupErr: any) {
          console.warn(`[Worker] Failed to delete temp file ${filePath}:`, cleanupErr.message);
        }
      }
    }
  },
  {
    connection: redisConnection,
    concurrency: 1, // 1 heavy 3D inference at a time to optimize memory and CPU
  }
);

inferenceWorker.on('completed', (job) => {
  console.log(`[Worker] Job ${job.id} has completed!`);
});

inferenceWorker.on('failed', (job, err) => {
  console.error(`[Worker] Job ${job?.id} failed with error:`, err.message);
});

export async function getJobDetails(jobId: string) {
  const job = await inferenceQueue.getJob(jobId);
  if (!job) {
    return null;
  }

  const state = await job.getState();
  const progress = job.progress;

  return {
    id: job.id,
    status: state, // 'waiting' | 'active' | 'completed' | 'failed' | 'delayed'
    progress,
    result: job.returnvalue || null,
    error: job.failedReason || null,
    timestamp: job.timestamp,
    processedOn: job.processedOn,
    finishedOn: job.finishedOn,
  };
}
