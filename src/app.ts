import dotenv from 'dotenv';
import express from 'express';
import logger from 'morgan';
import helmet from 'helmet';
import healthRouter from './routes/health';

dotenv.config();

const app = express();

if (process.env.NODE_ENV !== 'development') {
  app.use(helmet());
}

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use('/health', healthRouter);

export default app;
