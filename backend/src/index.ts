import express, { Request, Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import dotenv from 'dotenv';
import { inferenceQueue, getJobDetails, inferenceWorker, redisConnection } from './queue';

dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || '5001', 10);
const UPLOADS_DIR = path.join(__dirname, '..', 'uploads');

if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Enable CORS for frontend development
app.use(
  cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure Multer storage
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, UPLOADS_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `scan-${Date.now()}-${uuidv4()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 300 * 1024 * 1024, // 300 MB max for 3D NRRD scans
  },
});

// Health check endpoint
app.get('/health', async (_req: Request, res: Response) => {
  try {
    const redisPing = await redisConnection.ping();
    res.json({
      status: 'ok',
      service: 'implant-platform-backend',
      redis: redisPing === 'PONG' ? 'connected' : 'error',
      queue: 'medical-inference',
    });
  } catch (err: any) {
    res.status(503).json({
      status: 'degraded',
      error: err.message,
    });
  }
});

// Upload scan and enqueue inference job
app.post('/upload', upload.single('file'), async (req: Request, res: Response): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded. Please upload a .nrrd scan file under field name "file".' });
      return;
    }

    const { path: filePath, originalname, size } = req.file;

    console.log(`[Upload] Received file: ${originalname} (${(size / 1024 / 1024).toFixed(2)} MB)`);

    // Add job to BullMQ inference queue
    const job = await inferenceQueue.add(
      'infer_implant_job',
      {
        filePath,
        originalName: originalname,
        fileSize: size,
      },
      {
        jobId: `job-${uuidv4().slice(0, 8)}`,
      }
    );

    res.status(202).json({
      jobId: job.id,
      status: 'queued',
      message: 'Scan uploaded and queued for ML inference.',
    });
  } catch (err: any) {
    console.error('[Upload Error]', err);
    res.status(500).json({
      error: 'Failed to process file upload or enqueue inference job',
      details: err.message,
    });
  }
});

// Check status of inference job
app.get('/jobs/:id', async (req: Request, res: Response): Promise<void> => {
  try {
    const id = req.params.id as string;
    const jobInfo = await getJobDetails(id);

    if (!jobInfo) {
      res.status(404).json({
        error: `Job with ID ${id} not found`,
      });
      return;
    }

    res.json(jobInfo);
  } catch (err: any) {
    console.error(`[Job Query Error for ${req.params.id}]`, err);
    res.status(500).json({
      error: 'Failed to retrieve job status',
      details: err.message,
    });
  }
});

const server = app.listen(PORT, () => {
  console.log(`===============================================`);
  console.log(`🚀 Implant Platform Backend listening on port ${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/health`);
  console.log(`   Upload: POST http://localhost:${PORT}/upload`);
  console.log(`   Job Status: GET http://localhost:${PORT}/jobs/:id`);
  console.log(`===============================================`);
});

// Graceful shutdown
async function gracefulShutdown(signal: string) {
  console.log(`\n[Shutdown] Received ${signal}. Shutting down gracefully...`);
  server.close(async () => {
    console.log('[Shutdown] HTTP server closed.');
    await inferenceWorker.close();
    await inferenceQueue.close();
    await redisConnection.quit();
    console.log('[Shutdown] BullMQ worker and Redis connection closed.');
    process.exit(0);
  });
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
